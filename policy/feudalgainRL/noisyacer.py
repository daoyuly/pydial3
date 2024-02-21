###############################################################################
# PyDial: Multi-domain Statistical Spoken Dialogue System Software
###############################################################################
#
# Copyright 2015 - 2019
# Cambridge University Engineering Department Dialogue Systems Group
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
###############################################################################

"""
Implementation of ACER with Noisy Networks

The algorithm is developed with Tensorflow

Author: Gellert Weisz/Christian Geishauser
"""


import numpy as np
import tensorflow as tf


# ===========================
# Actor Critic with Experience Replay
# ===========================


class NoisyACERNetwork(object):
    def __init__(self, sess, state_dim, action_dim, learning_rate, delta, c, alpha, h1_size=130, h2_size=50,
                 is_training=True, actfreq_loss=None, noisy_acer=False):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        if isinstance(action_dim, list):
            self.master_space = True
            self.master_space_dim = self.a_dim[0] - self.a_dim[2] + self.a_dim[2] * self.a_dim[1]
        else:
            self.master_space = False
        self.learning_rate = learning_rate

        self.delta = delta
        self.c = c
        self.noisy_acer = noisy_acer
        self.alpha = alpha
        self.h1_size = h1_size
        self.h2_size = h2_size
        self.is_training = is_training

        #Input and hidden layers
        self.inputs = tf.placeholder(tf.float32, [None, self.s_dim])
        self.actions = tf.placeholder(tf.float32, [None, self.a_dim])
        self.execMask = tf.placeholder(tf.float32, [None, self.a_dim])
        self.behaviour_mask = tf.placeholder_with_default(tf.zeros(tf.shape(self.actions)[0], dtype=tf.float32), shape=[None])
        self.mu_values = tf.placeholder(tf.float32, [None])
        self.mu = tf.placeholder(tf.float32, [None, self.a_dim])

        if self.noisy_acer:
            print("WE USE NOISY ACER")
            self.policy, self.q = self.construct_noisy_network()
            self.network_params = tf.trainable_variables()
            self.avg_policy, _ = self.construct_noisy_network()
            self.target_network_params = tf.trainable_variables()[len(self.network_params):]
        else:
            self.policy, self.q = self.construct_network()
            self.network_params = tf.trainable_variables()
            self.avg_policy, _ = self.construct_network()
            self.target_network_params = tf.trainable_variables()[len(self.network_params):]

        self.avg_policy = tf.stop_gradient(self.avg_policy)

        # weighted average over q-values according to current policy gives the value of the state
        self.value = tf.reduce_sum(self.q * self.policy, 1)

        self.actions_onehot = self.actions
        self.responsible_outputs = tf.reduce_sum(self.policy * self.actions_onehot, [1])
        self.responsible_q = tf.reduce_sum(self.q * self.actions_onehot, [1])

        # IS weights
        self.responsible_mu = tf.reduce_sum(self.mu * self.actions_onehot, [1])
        self.rho = self.responsible_outputs / self.responsible_mu
        self.rho_all = self.policy / self.mu
        self.rho_bar = tf.minimum(1., self.rho)
        self.rho_bar_c = tf.minimum(self.c, self.rho)

        self.q_ret = tf.placeholder(tf.float32, [None])

        # step 1 from pawel
        self.advantages_qret = self.q_ret - self.value
        self.wrt_theta_step1 = -tf.reduce_sum(tf.log(self.responsible_outputs) * tf.stop_gradient(self.rho * self.advantages_qret))

        # step 2 from pawel
        self.wrt_theta = tf.reduce_sum(
            tf.log(self.responsible_outputs) *
            tf.stop_gradient(self.rho_bar_c * self.advantages_qret) + tf.reduce_sum(tf.log(self.policy) *
                          tf.stop_gradient(tf.maximum(0., 1. - self.c / self.rho_all) * self.policy *
                                           (self.q - tf.reshape(self.value, [-1, 1]))), [1]))

        self.q_regularizer = tf.placeholder(tf.float32, [None])
        self.wrt_theta_v = tf.reduce_sum(tf.square(self.q_ret - self.responsible_q))

        self.entropy = -tf.reduce_sum(self.policy * tf.log(self.policy))
        #self.loss = self.wrt_theta_v + self.wrt_theta - self.entropy * 0.01

        self.target_v = tf.placeholder(tf.float32, [None])
        self.advantages = tf.placeholder(tf.float32, [None])
        self.advantage_qret_diff = tf.reduce_mean(tf.square(self.advantages - self. advantages_qret))

        self.q_loss = 0.5 * self.wrt_theta_v
        self.policy_loss = -self.wrt_theta
        self.entropy = - tf.reduce_sum(self.policy * tf.log(self.policy))
        self.loss = self.q_loss + self.policy_loss - 0.01 * self.entropy

        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.optimize = self.optimizer.minimize(self.loss)

        # TRPO in theta-space
        use_trpo = True  # can switch off TRPO here
        self.value_gradients = self.optimizer.compute_gradients(self.q_loss)
        self.entropy_gradients = self.optimizer.compute_gradients(-0.01 * self.entropy)
        #self.behaviour_gradients = self.optimizer.compute_gradients(self.behaviour_loss)
        self.g = self.optimizer.compute_gradients(-self.policy_loss)
        self.kl = tf.reduce_sum(tf.reduce_sum(self.avg_policy * tf.log(self.avg_policy / self.policy), [1])) # this is total KL divergence, per batch
        self.k = self.optimizer.compute_gradients(self.kl)
        self.g = [(grad, var) for grad, var in self.g if grad is not None]
        self.k = [(grad, var) for grad, var in self.k if grad is not None]
        assert len(self.g) == len(self.k)
        self.klprod = tf.reduce_sum([tf.reduce_sum(tf.reshape(k[0], [-1]) * tf.reshape(g[0], [-1])) for k, g in zip(self.k, self.g)])
        self.klen = tf.reduce_sum([tf.reduce_sum(tf.reshape(k[0], [-1]) * tf.reshape(k[0], [-1])) for k, g in zip(self.k, self.g)])
        self.trpo_scale = tf.maximum(0., (self.klprod - self.delta) / self.klen)
        self.final_gradients = []
        for i in range(len(self.g)):
            if use_trpo:
                self.final_gradients.append((-(self.g[i][0] - self.trpo_scale * self.k[i][0]), self.g[i][1])) # negative because this is loss
            else:
                self.final_gradients.append((-self.g[i][0], self.g[i][1])) # negative because this is loss

        if not self.noisy_acer:
            self.optimize = [self.optimizer.apply_gradients(self.final_gradients),
                             self.optimizer.apply_gradients(self.entropy_gradients),
                             self.optimizer.apply_gradients(self.value_gradients)
                             ]
        else:
            self.optimize = [self.optimizer.apply_gradients(self.final_gradients),
                             self.optimizer.apply_gradients(self.value_gradients)
                             ]

        self.update_avg_theta = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], 1. - self.alpha)
                                                  + tf.multiply(self.target_network_params[i], self.alpha))
             for i in range(len(self.target_network_params))]

        self.saver = tf.train.Saver()

    def construct_network(self):
        W_fc1 = tf.Variable(tf.truncated_normal([self.s_dim, self.h1_size], stddev=0.01))
        b_fc1 = tf.Variable(tf.zeros([self.h1_size]))
        h_fc1 = tf.nn.relu(tf.matmul(self.inputs, W_fc1) + b_fc1)

        W_h2 = tf.Variable(tf.truncated_normal([self.h1_size, self.h2_size], stddev=0.01))
        b_h2 = tf.Variable(tf.zeros([self.h2_size]))
        h_h2 = tf.nn.relu(tf.matmul(h_fc1, W_h2) + b_h2)

        W_q = tf.Variable(tf.truncated_normal([self.h2_size, self.a_dim], stddev=0.01))
        b_q = tf.Variable(tf.zeros([self.a_dim]))
        q = tf.matmul(h_h2, W_q) + b_q

        W_policy = tf.Variable(tf.truncated_normal([self.h2_size, self.a_dim], stddev=0.01))
        b_policy = tf.Variable(tf.zeros([self.a_dim]))
        policy = tf.nn.softmax(tf.matmul(h_h2, W_policy) + b_policy + self.execMask) + 0.00001

        return policy, q

    def construct_noisy_network(self):
        self.mean_noisy_w = []
        self.mean_noisy_b = []

        h_fc1 = self.noisy_dense_layer(self.inputs, self.s_dim, self.h1_size, activation=tf.nn.relu)

        h_h2 = self.noisy_dense_layer(h_fc1, self.h1_size, self.h2_size, activation=tf.nn.relu)
        # Q function
        q = self.noisy_dense_layer(h_h2, self.h2_size, self.a_dim)

        policy = self.noisy_dense_layer(h_h2, self.h2_size, self.a_dim)
        # prevent problem when calling log(self.policy)
        policy = tf.nn.softmax(policy + self.execMask) + 0.00001

        return policy, q

    def getPolicy(self, inputs, execMask):
        return self.sess.run([self.policy], feed_dict={
            self.inputs: inputs,
            self.execMask: execMask,
        })

    def train(self, inputs, actions, execMask, rewards, unflattened_inputs, unflattened_rewards, gamma, mu,
              discounted_rewards, advantages, mu_values=None, behaviour_mask=None, critic_regularizer_output=None):
        value, responsible_q, rho_bar, responsible_outputs = self.sess.run(
            [self.value, self.responsible_q, self.rho_bar, self.responsible_outputs], feed_dict={
            self.inputs: inputs,
            self.actions: actions,
            self.execMask: execMask,
            self.mu: mu,
        })

        q_rets, offset = [], 0
        #print >> sys.stderr, rho_bar[0], value[0], responsible_q[0]
        for j in range(0, len(unflattened_inputs)):  # todo implement retrace for lambda other than one
            q_ret, new_q_ret = [], 0
            for i in range(len(unflattened_inputs[j])-1, -1, -1):
                new_q_ret = rewards[offset+i] + gamma * new_q_ret
                q_ret.append(new_q_ret)
                new_q_ret = rho_bar[offset+i] * (new_q_ret - responsible_q[offset+i]) + value[offset+i]
                #new_q_ret = value[offset+i] # debug
            q_ret = list(reversed(q_ret))
            q_rets.append(q_ret)
            offset += len(unflattened_inputs[j])

        q_ret_flat = np.concatenate(np.array(q_rets), axis=0).tolist()

        feed_dict = {
            self.inputs: inputs,
            self.actions: actions,
            self.execMask: execMask,
            self.mu: mu,
            self.q_ret: q_ret_flat,
            self.target_v: discounted_rewards,
            self.advantages: advantages,
            #self.mu_values: mu_values,
            #self.behaviour_mask: behaviour_mask
        }

        trpo_scale, klprod, kl, diff, entropy, loss, optimize = self.sess.run([self.trpo_scale, self.klprod, self.kl, self.advantage_qret_diff, self.entropy, self.loss, self.optimize], feed_dict=feed_dict)
        update_avg_theta = self.sess.run([self.update_avg_theta], feed_dict=feed_dict)

        return loss, entropy, optimize

    def predict_policy(self, inputs, execMask):
        return self.sess.run(self.policy, feed_dict={
            self.inputs: inputs,
            self.execMask: execMask,
        })

    def compute_rho(self, inputs, actions, mu, mask):
        return self.sess.run(self.rho,
                             feed_dict={self.inputs: inputs, self.actions: actions, self.mu: mu, self.execMask: mask})

    def compute_responsible_output(self, inputs, actions, mask):
        return self.sess.run(self.responsible_outputs,
                             feed_dict={self.inputs: inputs, self.actions: actions, self.execMask: mask})

    def compute_responsible_q(self, inputs, actions, mask):
        return self.sess.run(self.responsible_q,
                             feed_dict={self.inputs: inputs, self.actions: actions, self.execMask: mask})

    def predict_value(self, inputs, execMask):
        return self.sess.run(self.value, feed_dict={
            self.inputs: inputs,
            self.execMask: execMask,
        })

    def predict_action_value(self, inputs, execMask):
        return self.sess.run([self.policy, self.value], feed_dict={
            self.inputs: inputs,
            self.execMask: execMask,
        })

    def compute_mean_noisy(self):
        return self.sess.run([self.mean_noisy_w, self.mean_noisy_b])

    def noisy_dense_layer(self, input, input_neurons, output_neurons, activation=tf.identity):

        W_mu = tf.Variable(tf.truncated_normal([input_neurons, output_neurons], stddev=0.01))
        W_sigma = tf.Variable(tf.truncated_normal([input_neurons, output_neurons], stddev=0.01))
        W_eps = tf.random_normal(shape=[input_neurons, output_neurons])
        W = W_mu + tf.multiply(W_sigma, W_eps)

        b_mu = tf.Variable(tf.zeros([output_neurons]))
        b_sigma = tf.Variable(tf.zeros([output_neurons]))
        b_eps = tf.random_normal(shape=[output_neurons])
        b = b_mu + tf.multiply(b_sigma, b_eps)

        self.mean_noisy_w.append(tf.reduce_mean(tf.abs(W_sigma)))
        self.mean_noisy_b.append(tf.reduce_mean(tf.abs(b_sigma)))

        return activation(tf.matmul(input, W) + b)

    def load_network(self, load_filename):
        self.saver = tf.train.Saver()
        if load_filename.split('.')[-3] != '0':
            try:
                self.saver.restore(self.sess, load_filename)
                print("Successfully loaded:", load_filename)
            except:
                print("Could not find old network weights")
        else:
            print('nothing loaded in first iteration')

    def save_network(self, save_filename):
        print('Saving acer-network...')
        self.saver.save(self.sess, save_filename)


class RNNACERNetwork(object):
    def __init__(self, sess, si_state_dim, sd_state_dim, action_dim, learning_rate, delta, c, alpha, h1_size = 130, h2_size = 50, is_training = True, sd_enc_size=25,
                                    si_enc_size=25, dropout_rate=0., tn='normal', slot='si'):
        self.sess = sess
        self.s_dim = si_state_dim + sd_state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.delta = delta
        self.c = c
        self.alpha = alpha
        self.h1_size = h1_size
        self.h2_size = h2_size
        self.is_training = is_training
        self.sd_dim = sd_state_dim
        self.si_dim = si_state_dim
        self.sd_enc_size = sd_enc_size

        #Input and hidden layers
        self.inputs = tf.placeholder(tf.float32, [None, self.s_dim])
        self.actions = tf.placeholder(tf.float32, [None, self.a_dim])
        self.execMask = tf.placeholder(tf.float32, [None, self.a_dim])

        keep_prob = 1 - dropout_rate
        sd_inputs, si_inputs = tf.split(self.inputs, [self.sd_dim, self.si_dim], 1)

        if slot == 'sd':
            sd_inputs = tf.reshape(sd_inputs, (tf.shape(sd_inputs)[0], 1, self.sd_dim))

            # slots encoder
            with tf.variable_scope(tn):
                # try:
                lstm_cell = tf.nn.rnn_cell.GRUCell(self.sd_enc_size)
                if keep_prob < 1:
                    lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob)
                hidden_state = lstm_cell.zero_state(tf.shape(sd_inputs)[0], tf.float32)
                _, h_sdfe = tf.nn.dynamic_rnn(lstm_cell, sd_inputs, initial_state=hidden_state)
                # except:
                #    lstm_cell = tf.contrib.rnn.GRUCell(self.sd_enc_size)
                #    hidden_state = lstm_cell.zero_state(tf.shape(sd_inputs)[0], tf.float32)
                #    _, h_sdfe = tf.contrib.rnn.dynamic_rnn(lstm_cell, sd_inputs, initial_state=hidden_state)
            h1_inputs = tf.concat((si_inputs, h_sdfe), 1)
        else:
            '''W_sdfe = tf.Variable(tf.truncated_normal([self.sd_dim, sd_enc_size], stddev=0.01))
            b_sdfe = tf.Variable(tf.zeros([sd_enc_size]))
            h_sdfe = tf.nn.relu(tf.matmul(sd_inputs, W_sdfe) + b_sdfe)
            if keep_prob < 1:
                h_sdfe = tf.nn.dropout(h_sdfe, keep_prob)'''
            h1_inputs = self.inputs

        def construct_theta():
            W_fc1 = tf.Variable(tf.truncated_normal([self.s_dim, self.h1_size], stddev=0.01))
            b_fc1 = tf.Variable(0.0 * tf.ones([self.h1_size]))
            if self.h2_size > 0:  # todo layer 2 should be shared between policy and q-function?
                W_h2 = tf.Variable(tf.truncated_normal([self.h1_size, self.h2_size], stddev=0.01))
                b_h2 = tf.Variable(0.0 * tf.ones([self.h2_size]))

                W_q = tf.Variable(tf.truncated_normal([self.h2_size, self.a_dim], stddev=0.01))
                b_q = tf.Variable(0.0 * tf.ones([self.a_dim]))
                W_policy = tf.Variable(tf.truncated_normal([self.h2_size, self.a_dim], stddev=0.01))
                b_policy = tf.Variable(0.0 * tf.ones([self.a_dim]))

                theta = [W_fc1, b_fc1, W_h2, b_h2, W_q, b_q, W_policy, b_policy]
            else:
                W_q = tf.Variable(tf.truncated_normal([self.h1_size, self.a_dim], stddev=0.01))
                b_q = tf.Variable(0.0 * tf.ones([self.a_dim]))
                W_policy = tf.Variable(tf.truncated_normal([self.h1_size, self.a_dim], stddev=0.01))
                b_policy = tf.Variable(0.0 * tf.ones([self.a_dim]))

                theta = [W_fc1, b_fc1, W_q, b_q, W_policy, b_policy]
            return theta

        self.theta = construct_theta()
        self.avg_theta = construct_theta()

        def construct_network(theta):
            if self.h2_size > 0:
                W_fc1, b_fc1, W_h2, b_h2, W_q, b_q, W_policy, b_policy = theta
            else:
                W_fc1, b_fc1, W_q, b_q, W_policy, b_policy = theta

            h_fc1 = tf.nn.relu(tf.matmul(h1_inputs, W_fc1) + b_fc1)

            if self.h2_size > 0:
                h_h2 = tf.nn.relu(tf.matmul(h_fc1, W_h2) + b_h2)
                # Q function
                q = tf.matmul(h_h2, W_q) + b_q
                # prevent problem when calling log(self.policy)
                policy = tf.nn.softmax(tf.matmul(h_h2, W_policy) + b_policy + self.execMask) + 0.00001
            else:  # 1 hidden layer
                # value function
                q = tf.matmul(h_fc1, W_q) + b_q
                # policy function
                policy = tf.nn.softmax(tf.matmul(h_fc1, W_policy) + b_policy + self.execMask) + 0.00001
            return policy, q

        self.policy, self.q = construct_network(self.theta)
        self.avg_policy, _ = construct_network(self.avg_theta)
        self.avg_policy = tf.stop_gradient(self.avg_policy)

        # weighted average over q-values according to current policy gives the value of the state
        self.value = tf.reduce_sum(self.q * self.policy, 1)

        self.actions_onehot = self.actions
        self.responsible_outputs = tf.reduce_sum(self.policy * self.actions_onehot, [1])
        self.responsible_q = tf.reduce_sum(self.q * self.actions_onehot, [1])

        # IS weights
        self.mu = tf.placeholder(tf.float32, [None, self.a_dim])
        self.responsible_mu = tf.reduce_sum(self.mu * self.actions_onehot, [1])
        self.rho = self.responsible_outputs / self.responsible_mu
        self.rho_all = self.policy / self.mu
        self.rho_bar = tf.minimum(1., self.rho)
        self.rho_bar_c = tf.minimum(self.c, self.rho)

        self.q_ret = tf.placeholder(tf.float32, [None])

        # step 1 from pawel
        self.advantages_qret = self.q_ret - self.value
        self.wrt_theta_step1 = -tf.reduce_sum(tf.log(self.responsible_outputs) * tf.stop_gradient(self.rho *  self.advantages_qret))

        # step 2 from pawel
        self.wrt_theta = tf.reduce_sum(
            tf.log(self.responsible_outputs) * tf.stop_gradient(self.rho_bar_c *  self.advantages_qret) +
            tf.reduce_sum(tf.log(self.policy) *
                          tf.stop_gradient(tf.maximum(0., 1. - self.c / self.rho_all) *
                                           self.policy * (self.q - tf.reshape(self.value, [-1, 1]))), [1]))

        self.wrt_theta_v = tf.reduce_sum(tf.square(self.q_ret - self.responsible_q))
        self.entropy = -tf.reduce_sum(self.policy * tf.log(self.policy))
        #self.loss = self.wrt_theta_v + self.wrt_theta - self.entropy * 0.01

        self.target_v = tf.placeholder(tf.float32, [None])
        self.advantages = tf.placeholder(tf.float32, [None])
        self.advantage_qret_diff = tf.reduce_mean(tf.square(self.advantages - self. advantages_qret))

        # DEBUG (A2C)
        #self.value_loss = 0.5 * tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value, [-1]))) # original a2c
        self.q_loss = 0.5 * self.wrt_theta_v
        self.policy_loss = -self.wrt_theta
        self.entropy = - tf.reduce_sum(self.policy * tf.log(self.policy))
        self.loss = self.q_loss + self.policy_loss - 0.01 * self.entropy

        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.optimize = self.optimizer.minimize(self.loss)

        # TRPO in theta-space
        use_trpo = True  # can switch off TRPO here
        self.value_gradients = self.optimizer.compute_gradients(self.q_loss)
        self.entropy_gradients = self.optimizer.compute_gradients(-0.01 * self.entropy)
        self.g = self.optimizer.compute_gradients(-self.policy_loss)
        self.kl = tf.reduce_sum(tf.reduce_sum(self.avg_policy * tf.log(self.avg_policy / self.policy), [1])) # this is total KL divergence, per batch
        self.k = self.optimizer.compute_gradients(self.kl)
        self.g = [(grad, var) for grad, var in self.g if grad is not None]
        self.k = [(grad, var) for grad, var in self.k if grad is not None]
        assert len(self.g) == len(self.k)
        self.klprod = tf.reduce_sum([tf.reduce_sum(tf.reshape(k[0], [-1]) * tf.reshape(g[0], [-1])) for k, g in zip(self.k, self.g)])
        self.klen = tf.reduce_sum([tf.reduce_sum(tf.reshape(k[0], [-1]) * tf.reshape(k[0], [-1])) for k, g in zip(self.k, self.g)])
        self.trpo_scale = tf.maximum(0., (self.klprod - self.delta) / self.klen)
        self.final_gradients = []
        for i in range(len(self.g)):
            if use_trpo:
                self.final_gradients.append((-(self.g[i][0] - self.trpo_scale * self.k[i][0]), self.g[i][1])) # negative because this is loss
            else:
                self.final_gradients.append((-self.g[i][0], self.g[i][1])) # negative because this is loss

        self.optimize = [self.optimizer.apply_gradients(self.final_gradients),
                         self.optimizer.apply_gradients(self.entropy_gradients),
                         self.optimizer.apply_gradients(self.value_gradients)]

        self.update_avg_theta = [avg_w.assign(self.alpha * avg_w + (1. - self.alpha) * w)
                                 for avg_w, w in zip(self.avg_theta, self.theta)]


    def getPolicy(self, inputs, execMask):
        return self.sess.run([self.policy], feed_dict={
            self.inputs: inputs,
            self.execMask: execMask,
        })

    def train(self, inputs, actions, execMask, rewards, unflattened_inputs, unflattened_rewards, gamma, mu, discounted_rewards, advantages):
        value, responsible_q, rho_bar, responsible_outputs = self.sess.run(
            [self.value, self.responsible_q, self.rho_bar, self.responsible_outputs], feed_dict={
            self.inputs: inputs,
            self.actions: actions,
            self.execMask: execMask,
            self.mu: mu,
        })

        q_rets, offset = [], 0
        #print >> sys.stderr, rho_bar[0], value[0], responsible_q[0]
        for j in range(0, len(unflattened_inputs)):  # todo implement retrace for lambda other than one
            q_ret, new_q_ret = [], 0
            for i in range(len(unflattened_inputs[j])-1, -1, -1):
                new_q_ret = rewards[offset+i] + gamma * new_q_ret
                q_ret.append(new_q_ret)
                new_q_ret = rho_bar[offset+i] * (new_q_ret - responsible_q[offset+i]) + value[offset+i]
                #new_q_ret = value[offset+i] # debug
            q_ret = list(reversed(q_ret))
            q_rets.append(q_ret)
            offset += len(unflattened_inputs[j])

        q_ret_flat = np.concatenate(np.array(q_rets), axis=0).tolist()

        feed_dict = {
            self.inputs: inputs,
            self.actions: actions,
            self.execMask: execMask,
            self.mu: mu,
            self.q_ret: q_ret_flat,
            self.target_v: discounted_rewards,
            self.advantages: advantages,
        }

        trpo_scale, klprod, kl, diff, entropy, loss, optimize = self.sess.run([self.trpo_scale, self.klprod, self.kl, self.advantage_qret_diff, self.entropy, self.loss, self.optimize], feed_dict=feed_dict)
        update_avg_theta = self.sess.run([self.update_avg_theta], feed_dict=feed_dict)

        return loss, entropy, optimize

    def predict_policy(self, inputs, execMask):
        return self.sess.run(self.policy, feed_dict={
            self.inputs: inputs,
            self.execMask: execMask,
        })

    def predict_value(self, inputs, execMask):
        return self.sess.run(self.value, feed_dict={
            self.inputs: inputs,
            self.execMask: execMask,
        })

    def predict_action_value(self, inputs, execMask):
        return self.sess.run([self.policy, self.value], feed_dict={
            self.inputs: inputs,
            self.execMask: execMask,
        })

    def load_network(self, load_filename):
        self.saver = tf.train.Saver()
        if load_filename.split('.')[-3] != '0':
            try:
                self.saver.restore(self.sess, load_filename)
                print("Successfully loaded:", load_filename)
            except:
                print("Could not find old network weights")
        else:
            print('nothing loaded in first iteration')

    def save_network(self, save_filename):
        print('Saving sacer-network...')
        #self.saver = tf.train.Saver()
        self.saver.save(self.sess, save_filename)

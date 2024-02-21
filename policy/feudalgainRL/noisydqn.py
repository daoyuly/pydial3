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
Implementation of DQN -  Deep Q Network

The algorithm is developed with tflearn + Tensorflow

Author: Pei-Hao Su
"""
import tensorflow as tf


class NNFDeepQNetwork(object):
    """
    Input to the network is the state and action, output is Q(s,a).
    """
    def __init__(self, sess, si_state_dim, sd_state_dim, action_dim, learning_rate, tau, num_actor_vars, minibatch_size=64,
                 architecture='duel', h1_size=130, h2_size=50, sd_enc_size=40, si_enc_size=80, dropout_rate=0.):
        #super(NNFDeepQNetwork, self).__init__(sess, si_state_dim + sd_state_dim, action_dim, learning_rate, tau, num_actor_vars,
        #                                      minibatch_size=64, architecture='duel', h1_size=130, h2_size=50)
        self.sess = sess
        self.si_dim = si_state_dim
        self.sd_dim = sd_state_dim
        self.s_dim = self.si_dim + self.sd_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau
        self.architecture = architecture
        self.h1_size = h1_size
        self.h2_size = h2_size
        self.minibatch_size = minibatch_size
        self.sd_enc_size = sd_enc_size
        self.si_enc_size = si_enc_size
        self.dropout_rate = dropout_rate

        # Create the deep Q network
        self.inputs, self.action, self.Qout = \
                        self.create_nnfdq_network(self.h1_size, self.h2_size, self.sd_enc_size, self.si_enc_size, self.dropout_rate)
        self.network_params = tf.trainable_variables()

        # Target Network
        self.target_inputs, self.target_action, self.target_Qout = \
                        self.create_nnfdq_network(self.h1_size, self.h2_size, self.sd_enc_size, self.si_enc_size, self.dropout_rate)
        self.target_network_params = tf.trainable_variables()[len(self.network_params):]

        # Op for periodically updating target network
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau)
                                                  + tf.multiply(self.target_network_params[i], 1. - self.tau))
             for i in range(len(self.target_network_params))]

        # Network target (y_i)
        self.sampled_q = tf.placeholder(tf.float32, [None, 1])

        # Predicted Q given state and chosed action
        actions_one_hot = self.action

        if architecture != 'dip':
            self.pred_q = tf.reshape(tf.reduce_sum(self.Qout * actions_one_hot, axis=1, name='q_acted'),
                                 [-1, 1])
        else:
            self.pred_q = self.Qout

        # Define loss and optimization Op
        self.diff = self.sampled_q - self.pred_q
        self.loss = tf.reduce_mean(self.clipped_error(self.diff), name='loss')

        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.optimize = self.optimizer.minimize(self.loss)

    def create_nnfdq_network(self, h1_size=130, h2_size=50, sd_enc_size=40, si_enc_size=80, dropout_rate=0.):

        keep_prob = 1 - dropout_rate
        inputs = tf.placeholder(tf.float32, [None, self.s_dim])
        action = tf.placeholder(tf.float32, [None, self.a_dim])

        if self.architecture == 'duel':
            print("WE USE THE DUELING ARCHITECTURE!")
            W_fc1 = tf.Variable(tf.truncated_normal([self.s_dim, h1_size], stddev=0.01))
            b_fc1 = tf.Variable(tf.zeros([h1_size]))
            h_fc1 = tf.nn.relu(tf.matmul(inputs, W_fc1) + b_fc1)

            # value function
            W_value = tf.Variable(tf.truncated_normal([h1_size, h2_size], stddev=0.01))
            b_value = tf.Variable(tf.zeros([h2_size]))
            h_value = tf.nn.relu(tf.matmul(h_fc1, W_value) + b_value)

            W_value = tf.Variable(tf.truncated_normal([h2_size, 1], stddev=0.01))
            b_value = tf.Variable(tf.zeros([1]))
            value_out = tf.matmul(h_value, W_value) + b_value

            # advantage function
            W_advantage = tf.Variable(tf.truncated_normal([h1_size, h2_size], stddev=0.01))
            b_advantage = tf.Variable(tf.zeros([h2_size]))
            h_advantage = tf.nn.relu(tf.matmul(h_fc1, W_advantage) + b_advantage)

            W_advantage = tf.Variable(tf.truncated_normal([h2_size, self.a_dim], stddev=0.01))
            b_advantage = tf.Variable(tf.zeros([self.a_dim]))
            Advantage_out = tf.matmul(h_advantage, W_advantage) + b_advantage

            Qout = value_out + (Advantage_out - tf.reduce_mean(Advantage_out, axis=1, keep_dims=True))

        elif self.architecture == 'noisy_duel':
            print("WE USE THE NOISY DUELING ARCHITECTURE!")
            self.mean_noisy_w = []
            self.mean_noisy_b = []
            h_fc1 = self.noisy_dense_layer(inputs, self.s_dim, h1_size, activation=tf.nn.relu)
            # value function
            h_value = self.noisy_dense_layer(h_fc1, h1_size, h2_size, activation=tf.nn.relu)
            value_out = self.noisy_dense_layer(h_value, h2_size, 1)

            # advantage function
            h_advantage = self.noisy_dense_layer(h_fc1, h1_size, h2_size, activation=tf.nn.relu)
            Advantage_out = self.noisy_dense_layer(h_advantage, h2_size, self.a_dim)

            Qout = value_out + (Advantage_out - tf.reduce_mean(Advantage_out, axis=1, keep_dims=True))

        else:
            inputs = tf.placeholder(tf.float32, [None, self.sd_dim + self.si_dim])
            keep_prob = 1 - dropout_rate
            sd_inputs, si_inputs = tf.split(inputs, [self.sd_dim, self.si_dim], 1)
            action = tf.placeholder(tf.float32, [None, self.a_dim])

            W_sdfe = tf.Variable(tf.truncated_normal([self.sd_dim, sd_enc_size], stddev=0.01))
            b_sdfe = tf.Variable(tf.zeros([sd_enc_size]))
            h_sdfe = tf.nn.relu(tf.matmul(sd_inputs, W_sdfe) + b_sdfe)
            if keep_prob < 1:
                h_sdfe = tf.nn.dropout(h_sdfe, keep_prob)

            W_sife = tf.Variable(tf.truncated_normal([self.si_dim, si_enc_size], stddev=0.01))
            b_sife = tf.Variable(tf.zeros([si_enc_size]))
            h_sife = tf.nn.relu(tf.matmul(si_inputs, W_sife) + b_sife)
            if keep_prob < 1:
                h_sife = tf.nn.dropout(h_sife, keep_prob)

            W_fc1 = tf.Variable(tf.truncated_normal([sd_enc_size+si_enc_size, h1_size], stddev=0.01))
            b_fc1 = tf.Variable(tf.zeros([h1_size]))
            h_fc1 = tf.nn.relu(tf.matmul(tf.concat((h_sdfe, h_sife), 1), W_fc1) + b_fc1)

            W_fc2 = tf.Variable(tf.truncated_normal([h1_size, h2_size], stddev=0.01))
            b_fc2 = tf.Variable(tf.zeros([h2_size]))
            h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

            W_out = tf.Variable(tf.truncated_normal([h2_size, self.a_dim], stddev=0.01))
            b_out = tf.Variable(tf.zeros([self.a_dim]))
            Qout = tf.matmul(h_fc2, W_out) + b_out

        return inputs, action, Qout

    def predict(self, inputs):
        return self.sess.run(self.Qout, feed_dict={ #inputs where a single flat_bstate
            self.inputs: inputs
        })

    def predict_dip(self, inputs, action):
        return self.sess.run(self.Qout, feed_dict={ #inputs and action where array of 64 (batch size)
            self.inputs: inputs,
            self.action: action
        })

    def predict_target(self, inputs):
        return self.sess.run(self.target_Qout, feed_dict={ #inputs where a single flat_bstate
            self.target_inputs: inputs
        })

    def predict_target_dip(self, inputs, action):
        return self.sess.run(self.target_Qout, feed_dict={ #inputs and action where array of 64 (batch size)
            self.target_inputs: inputs,
            self.target_action: action
        })

    def train(self, inputs, action, sampled_q):
        return self.sess.run([self.pred_q, self.optimize, self.loss], feed_dict={ #all the inputs are arrays of 64
            self.inputs: inputs,
            self.action: action,
            self.sampled_q: sampled_q
        })

    def compute_loss(self, inputs, action, sampled_q):

        return self.sess.run(self.loss, feed_dict={  # yes, needs to be changed too
            self.inputs: inputs,
            self.action: action,
            self.sampled_q: sampled_q
        })

    def clipped_error(self, x):
        return tf.where(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5) # condition, true, false

    def save_network(self, save_filename):
        print('Saving deepq-network...')
        self.saver.save(self.sess, './' +  save_filename)

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def load_network(self, load_filename):
        self.saver = tf.train.Saver()
        if load_filename.split('.')[-3] != '0':
            try:
                self.saver.restore(self.sess, './' + load_filename)
                print("Successfully loaded:", load_filename)
            except:
                print("Could not find old network weights")
        else:
            print('nothing loaded in first iteration')

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

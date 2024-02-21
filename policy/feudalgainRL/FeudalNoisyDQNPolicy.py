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

'''
DQNPolicy.py - deep Q network policy
==================================================

Copyright CUED Dialogue Systems Group 2015 - 2017

.. seealso:: CUED Imports/Dependencies: 

    import :class:`Policy`
    import :class:`utils.ContextLogger`

.. warning::
        Documentation not done.


************************

'''

import copy
import os
import sys
import json
import numpy as np
import pickle as pickle
from itertools import product
from scipy.stats import entropy
import utils
#from pydial import log_dir
from utils.Settings import config as cfg
from utils import ContextLogger, DiaAct, DialogueState

import ontology.FlatOntologyManager as FlatOnt
import tensorflow as tf
from policy.DRL.replay_buffer import ReplayBuffer
from policy.DRL.replay_prioritised import ReplayPrioritised
import policy.feudalgainRL.noisydqn as dqn
import policy.Policy
import policy.DQNPolicy
import policy.SummaryAction
from policy.Policy import TerminalAction, TerminalState
from policy.feudalgainRL.DIP_parametrisation import DIP_state, padded_state


logger = utils.ContextLogger.getLogger('')


class FeudalDQNPolicy(policy.DQNPolicy.DQNPolicy):
    '''Derived from :class:`DQNPolicy`
    '''

    def __init__(self, in_policy_file, out_policy_file, domainString='CamRestaurants', is_training=False,
                 action_names=None, slot=None, sd_state_dim=50, js_threshold=0, info_reward=0.0, jsd_reward=False,
                 jsd_function=None):
        super(FeudalDQNPolicy, self).__init__(in_policy_file, out_policy_file, domainString, is_training)

        tf.reset_default_graph()

        self.domainString = domainString
        self.sd_state_dim = sd_state_dim
        self.domainUtil = FlatOnt.FlatDomainOntology(self.domainString)
        self.in_policy_file = in_policy_file
        self.out_policy_file = out_policy_file
        self.is_training = is_training
        self.accum_belief = []
        self.info_reward = info_reward
        self.js_threshold = js_threshold
        self.jsd_reward = jsd_reward
        self.jsd_function = jsd_function
        self.log_path = cfg.get('exec_config', 'logfiledir')
        self.log_path = self.log_path + f"/{in_policy_file.split('/')[-1].split('.')[0]}-seed{self.randomseed}.txt"

        if self.jsd_function is not None:
            print("We use the JSD-function", self.jsd_function)
        if self.js_threshold != 1.0 and not self.jsd_reward:
            print("We use Information Gain with JS-divergence, threshold =", self.js_threshold)
        if self.jsd_reward:
            print("We train with raw JSD reward.")
        self.slots = slot
        self.features = 'dip'
        if cfg.has_option('feudalpolicy', 'features'):
            self.features = cfg.get('feudalpolicy', 'features')
        self.actfreq_ds = False
        if cfg.has_option('feudalpolicy', 'actfreq_ds'):
            self.actfreq_ds = cfg.getboolean('feudalpolicy', 'actfreq_ds')

        self.use_pass = False
        if cfg.has_option('feudalpolicy', 'use_pass'):
            self.use_pass = cfg.getboolean('feudalpolicy', 'use_pass')

        if self.use_pass:
            print("We work with pass action in DQN training")
        else:
            print("We work without pass action in DQN training")

        self.domainUtil = FlatOnt.FlatDomainOntology(self.domainString)
        self.prev_state_check = None

        self.max_k = 5
        if cfg.has_option('dqnpolicy', 'max_k'):
            self.max_k = cfg.getint('dqnpolicy', 'max_k')

        self.capacity *= 5  # capacity for episode methods, multiply it to adjust to turn based methods

        # init session
        self.sess = tf.Session()
        with tf.device("/cpu:0"):

            np.random.seed(self.randomseed)
            tf.set_random_seed(self.randomseed)

            # initialise a replay buffer
            if self.replay_type == 'vanilla':
                self.episodes[self.domainString] = ReplayBuffer(self.capacity, self.minibatch_size*4, self.randomseed)
            elif self.replay_type == 'prioritized':
                self.episodes[self.domainString] = ReplayPrioritised(self.capacity, self.minibatch_size,
                                                                     self.randomseed)
            self.samplecount = 0
            self.episodecount = 0

            # construct the models
            self.summaryaction = policy.SummaryAction.SummaryAction(domainString)
            self.action_names = action_names
            self.action_dim = len(self.action_names)
            action_bound = len(self.action_names)
            self.stats = [0 for _ in range(self.action_dim)]

            if self.features == 'learned' or self.features == 'rnn':
                si_state_dim = 73
                if self.actfreq_ds:
                    if self.domainString == 'CamRestaurants':
                        si_state_dim += 9#16
                    elif self.domainString == 'SFRestaurants':
                        si_state_dim += 9#25
                    elif self.domainString == 'Laptops11':
                        si_state_dim += 9#40
                self.sd_enc_size = 50
                self.si_enc_size = 25
                self.dropout_rate = 0.
                if cfg.has_option('feudalpolicy', 'sd_enc_size'):
                    self.sd_enc_size = cfg.getint('feudalpolicy', 'sd_enc_size')
                if cfg.has_option('feudalpolicy', 'si_enc_size'):
                    self.si_enc_size = cfg.getint('feudalpolicy', 'si_enc_size')
                if cfg.has_option('dqnpolicy', 'dropout_rate') and self.is_training:
                    self.dropout_rate = cfg.getfloat('feudalpolicy', 'dropout_rate')
                if cfg.has_option('dqnpolicy', 'dropout_rate') and self.is_training:
                    self.dropout_rate = cfg.getfloat('feudalpolicy', 'dropout_rate')

                self.state_dim = si_state_dim + sd_state_dim
                if self.features == 'learned':
                    self.dqn = dqn.NNFDeepQNetwork(self.sess, si_state_dim, sd_state_dim, self.action_dim,
                                            self.learning_rate, self.tau, action_bound, self.minibatch_size,
                                            self.architecture, self.h1_size, self.h2_size, sd_enc_size=self.sd_enc_size,
                                               si_enc_size=self.si_enc_size, dropout_rate=self.dropout_rate)

                elif self.features == 'rnn':
                    self.dqn = dqn.RNNFDeepQNetwork(self.sess, si_state_dim, sd_state_dim, self.action_dim,
                                                   self.learning_rate, self.tau, action_bound, self.minibatch_size,
                                                   self.architecture, self.h1_size, self.h2_size,
                                                   sd_enc_size=self.sd_enc_size, si_enc_size=self.si_enc_size,
                                                   dropout_rate=self.dropout_rate, slot=self.slot)
            else: # self.features = 'dip'
                if self.actfreq_ds:
                    if self.domainString == 'CamRestaurants':
                        self.state_dim += 9#16
                    elif self.domainString == 'SFRestaurants':
                        self.state_dim += 9#25
                    elif self.domainString == 'Laptops11':
                        self.state_dim += 9#40
                self.dqn = dqn.DeepQNetwork(self.sess, self.state_dim, self.action_dim,
                                            self.learning_rate, self.tau, action_bound, self.minibatch_size,
                                            self.architecture, self.h1_size,
                                            self.h2_size, dropout_rate=self.dropout_rate)

            # when all models are defined, init all variables (this might to be sent to the main policy too)
            init_op = tf.global_variables_initializer()
            self.sess.run(init_op)

            self.loadPolicy(self.in_policy_file)
            print('loaded replay size: ', self.episodes[self.domainString].size())

            self.dqn.update_target_network()

    def record(self, reward, domainInControl=None, weight=None, state=None, action=None, exec_mask=None):
        if domainInControl is None:
            domainInControl = self.domainString
        if self.actToBeRecorded is None:
            self.actToBeRecorded = self.summaryAct

        if state is None:
            state = self.prevbelief
        if action is None:
            action = self.actToBeRecorded

        cState, cAction = state, action
        # normalising total return to -1~1
        reward /= 20.0

        if self.replay_type == 'vanilla':
            self.episodes[domainInControl].record(state=cState, \
                                                  state_ori=state, action=cAction, reward=reward)

        self.actToBeRecorded = None
        self.samplecount += 1

    def finalizeRecord(self, reward, domainInControl=None):
        if domainInControl is None:
            domainInControl = self.domainString
        if self.episodes[domainInControl] is None:
            logger.warning("record attempted to be finalized for domain where nothing has been recorded before")
            return

        reward /= 20.0

        terminal_state, terminal_action = self.convertStateAction(TerminalState(), TerminalAction())

        if self.replay_type == 'vanilla':
            self.episodes[domainInControl].record(state=terminal_state, \
                                                  state_ori=TerminalState(), action=terminal_action, reward=reward,
                                                  terminal=True)
        elif self.replay_type == 'prioritized':
            self.episodes[domainInControl].record(state=terminal_state, \
                                                      state_ori=TerminalState(), action=terminal_action, reward=reward, \
                                                      Q_s_t_a_t_=0.0, gamma_Q_s_tplu1_maxa_=0.0, uniform=False,
                                                      terminal=True)
            print('total TD', self.episodes[self.domainString].tree.total())

    def convertStateAction(self, state, action):
        '''

        '''
        if isinstance(state, TerminalState):
            return [0] * 89, action
        else:
            if self.features == 'learned' or self.features == 'rnn':
                dip_state = padded_state(state.domainStates[state.currentdomain], self.domainString)
            else:
                dip_state = DIP_state(state.domainStates[state.currentdomain], self.domainString)
            action_name = self.actions.action_names[action]
            act_slot = 'general'
            for slot in dip_state.slots:
                if slot in action_name:
                    act_slot = slot
            flat_belief = dip_state.get_beliefStateVec(act_slot)
            self.prev_state_check = flat_belief

            return flat_belief, action

    def nextAction(self, beliefstate):
        '''
        select next action

        :param beliefstate: already converted to dipstatevec of the specific slot (or general)
        :returns: (int) next summary action
        '''

        if self.exploration_type == 'e-greedy' and self.architecture != 'noisy_duel':
            # epsilon greedy
            if self.is_training and utils.Settings.random.rand() < self.epsilon:
                action_Q = np.random.rand(len(self.action_names))
            else:
                if len(beliefstate.shape) == 1:
                    action_Q = self.dqn.predict(np.reshape(beliefstate, (1, -1)))
                else:
                    action_Q = self.dqn.predict(beliefstate)
                # add current max Q to self.episode_ave_max_q
                self.episode_ave_max_q.append(np.max(action_Q))
        elif self.architecture == 'noisy_duel':
            if len(beliefstate.shape) == 1:
                action_Q = self.dqn.predict(np.reshape(beliefstate, (1, -1)))
            else:
                action_Q = self.dqn.predict(beliefstate)
            # add current max Q to self.episode_ave_max_q
            self.episode_ave_max_q.append(np.max(action_Q))

        #return the Q vect, the action will be converted in the feudal policy
        return action_Q

    def train(self):
        '''
        call this function when the episode ends
        '''

        if not self.is_training:
            logger.info("Not in training mode")
            return
        else:
            logger.info("Update dqn policy parameters.")

        self.episodecount += 1
        logger.info("Sample Num so far: %s" % (self.samplecount))
        logger.info("Episode Num so far: %s" % (self.episodecount))

        s_batch_new, s_batch_beliefstate, s_batch_chosen_slot, s2_batch_dipstate, s2_batch_beliefstate, t_batch_new, r_batch_new = \
            [], [], [], [], [], [], []

        if self.samplecount >= self.minibatch_size * 10 and self.episodecount % self.training_frequency == 0:
            logger.info('start training...')

            a_batch_one_hot_new = None
            #since in a batch we can take only non-pass() actions, we have to loop a bit until we get enough samples

            if self.js_threshold < 1.0 or not self.use_pass:
                while len(s_batch_new) < self.minibatch_size:

                    s_batch, s_ori_batch, a_batch, r_batch, s2_batch, s2_ori_batch, t_batch, idx_batch, _ = \
                        self.episodes[self.domainString].sample_batch()

                    a_batch_one_hot = np.eye(self.action_dim, self.action_dim)[a_batch]
                    #we only wanna update state-action pairs, where action != pass()
                    valid_steps = [action[-1] != 1 for action in a_batch_one_hot]
                    a_batch_one_hot = a_batch_one_hot[valid_steps]

                    s_batch_new += [s[0] for i, s in enumerate(s_batch) if valid_steps[i]]
                    s_batch_beliefstate += [s[1] for i, s in enumerate(s_batch) if valid_steps[i]]
                    s_batch_chosen_slot += [s[2] for i, s in enumerate(s_batch) if valid_steps[i]]

                    s2_batch_dipstate += [s[3] for s, valid in zip(s2_batch, valid_steps) if valid]
                    s2_batch_beliefstate += [s[1] for s, valid in zip(s2_batch, valid_steps) if valid]

                    r_batch_new += [r for r, valid in zip(r_batch, valid_steps) if valid]
                    t_batch_new += [t for t, valid in zip(t_batch, valid_steps) if valid]

                    if a_batch_one_hot_new is None:
                        a_batch_one_hot_new = a_batch_one_hot
                    else:
                        a_batch_one_hot_new = np.vstack((a_batch_one_hot_new, a_batch_one_hot))

                s_batch_new = np.vstack(s_batch_new)
                s2_batch_dipstate = np.vstack(s2_batch_dipstate)

            else:
                s_batch, s_ori_batch, a_batch, r_batch, s2_batch, s2_ori_batch, t_batch, idx_batch, _ = \
                    self.episodes[self.domainString].sample_batch()

                a_batch_one_hot_new = np.eye(self.action_dim, self.action_dim)[a_batch]
                s_batch_new = np.vstack([s[0] for s in s_batch])
                r_batch_new = r_batch
                s2_batch_dipstate = np.vstack([s[3] for s in s2_batch])
                t_batch_new = t_batch

            if self.js_threshold < 1.0:
                js_divergence_batch = []
                for belief, belief2, slot in zip(s_batch_beliefstate, s2_batch_beliefstate, s_batch_chosen_slot):
                    if slot != "None":
                        keys = belief['beliefs'][slot].keys()

                        b = [belief['beliefs'][slot]['**NONE**']] + \
                            [belief['beliefs'][slot][value] for value in list(keys) if value != '**NONE**']

                        b_2 = [belief2['beliefs'][slot]['**NONE**']] + \
                              [belief2['beliefs'][slot][value] for value in list(keys) if value != '**NONE**']

                        js_divergence = self.compute_js_divergence(b, b_2)
                        js_divergence_batch.append(js_divergence)
                    else:
                        js_divergence_batch.append(0.0)
            else:
                js_divergence_batch = [0] * len(r_batch_new)

            if self.js_threshold < 1.0:
                # normalizing bound to [0, 2] and then /20
                js_divergence_batch = [2/20 * int(x > self.js_threshold) for x in js_divergence_batch]

            action_q = self.dqn.predict_dip(s2_batch_dipstate, a_batch_one_hot_new)
            target_q = self.dqn.predict_target_dip(s2_batch_dipstate, a_batch_one_hot_new)

            action_q = np.reshape(action_q, (s_batch_new.shape[0], -1, self.action_dim))
            target_q = np.reshape(target_q, (s_batch_new.shape[0], -1, self.action_dim))

            y_i = []
            for k in range(min(s_batch_new.shape[0], self.episodes[self.domainString].size())):
                Q_bootstrap_label = 0
                if t_batch_new[k]:
                    Q_bootstrap_label = r_batch_new[k]
                else:
                    if self.q_update == 'single':
                        action_Q = target_q[k]
                        Q_bootstrap_label = r_batch_new[k] + js_divergence_batch[k] + self.gamma * np.max(action_Q)
                    elif self.q_update == 'double':
                        action_Q = action_q[k]
                        argmax_tuple = np.unravel_index(np.argmax(action_Q, axis=None), action_Q.shape)
                        value_Q = target_q[k][argmax_tuple]
                        Q_bootstrap_label = r_batch_new[k] + js_divergence_batch[k] + self.gamma * value_Q

                y_i.append(Q_bootstrap_label)

                if self.replay_type == 'prioritized':
                    # update the sum-tree
                    # update the TD error of the samples in the minibatch
                    currentQ_s_a_ = action_q[k][a_batch[k]]
                    error = abs(currentQ_s_a_ - Q_bootstrap_label)
                    self.episodes[self.domainString].update(idx_batch[k], error)

            reshaped_yi = np.vstack([np.expand_dims(x, 0) for x in y_i])

            predicted_q_value, _, currentLoss = self.dqn.train(s_batch_new, a_batch_one_hot_new, reshaped_yi)

            self.log_loss()

            if self.episodecount % 1 == 0:
                # Update target networks
                self.dqn.update_target_network()

        self.savePolicyInc()

    def log_loss(self):

        s_batch_new, s_batch_beliefstate, s_batch_chosen_slot, s2_batch_dipstate, s2_batch_beliefstate, t_batch_new, r_batch_new = \
            [], [], [], [], [], [], []

        if self.samplecount >= self.minibatch_size * 8 and self.episodecount % self.training_frequency == 0:
            logger.info('start training...')

            a_batch_one_hot_new = None
            #updating only states where the action is not "pass()" complicates things :/
            #since in a batch we can take only non-pass() actions, we have to loop a bit until we get enough samples

            while len(s_batch_new) < 512:

                s_batch, s_ori_batch, a_batch, r_batch, s2_batch, s2_ori_batch, t_batch, idx_batch, _ = \
                    self.episodes[self.domainString].sample_batch()

                a_batch_one_hot = np.eye(self.action_dim, self.action_dim)[a_batch]
                #we only wanna update state-action pairs, where action != pass()
                valid_steps = [action[-1] != 1 for action in a_batch_one_hot]
                a_batch_one_hot = a_batch_one_hot[valid_steps]

                s_batch_new += [s[0] for i, s in enumerate(s_batch) if valid_steps[i]]
                s_batch_beliefstate += [s[1] for i, s in enumerate(s_batch) if valid_steps[i]]
                s_batch_chosen_slot += [s[2] for i, s in enumerate(s_batch) if valid_steps[i]]

                s2_batch_dipstate += [s[3] for s, valid in zip(s2_batch, valid_steps) if valid]
                s2_batch_beliefstate += [s[1] for s, valid in zip(s2_batch, valid_steps) if valid]

                r_batch_new += [r for r, valid in zip(r_batch, valid_steps) if valid]
                t_batch_new += [t for t, valid in zip(t_batch, valid_steps) if valid]

                if a_batch_one_hot_new is None:
                    a_batch_one_hot_new = a_batch_one_hot
                else:
                    a_batch_one_hot_new = np.vstack((a_batch_one_hot_new, a_batch_one_hot))

            s_batch_new = np.vstack(s_batch_new)
            s2_batch_dipstate = np.vstack(s2_batch_dipstate)

            if self.js_threshold < 1.0 or self.jsd_reward:
                #TODO: This is highly inefficient
                js_divergence_batch = []
                for belief, belief2, slot in zip(s_batch_beliefstate, s2_batch_beliefstate, s_batch_chosen_slot):
                    if slot != "None":
                        keys = belief['beliefs'][slot].keys()

                        b = [belief['beliefs'][slot]['**NONE**']] + \
                            [belief['beliefs'][slot][value] for value in list(keys) if value != '**NONE**']

                        b_2 = [belief2['beliefs'][slot]['**NONE**']] + \
                              [belief2['beliefs'][slot][value] for value in list(keys) if value != '**NONE**']

                        js_divergence = self.compute_js_divergence(b, b_2)
                        js_divergence_batch.append(js_divergence)
                    else:
                        js_divergence_batch.append(0.0)
            else:
                js_divergence_batch = [0] * len(r_batch_new)

            tanh_n = np.tanh(1)
            if self.jsd_reward:
                if self.jsd_function == 'tanh':
                    js_divergence_batch = np.tanh(np.array(js_divergence_batch)) / tanh_n
                #normalize jsd between -1 and 1
                js_divergence_batch = (-1 + 2 * np.array(js_divergence_batch)).tolist()
            elif self.js_threshold < 1.0:
                # normalizing bound to [0, 2] and then /20
                js_divergence_batch = [2/20 * int(x > self.js_threshold) for x in js_divergence_batch]

            action_q = self.dqn.predict_dip(s2_batch_dipstate, a_batch_one_hot_new)
            target_q = self.dqn.predict_target_dip(s2_batch_dipstate, a_batch_one_hot_new)

            action_q = np.reshape(action_q, (s_batch_new.shape[0], -1, self.action_dim))
            target_q = np.reshape(target_q, (s_batch_new.shape[0], -1, self.action_dim))

            y_i = []
            for k in range(s_batch_new.shape[0]):
                Q_bootstrap_label = 0
                if t_batch_new[k]:
                    Q_bootstrap_label = r_batch_new[k]
                else:
                    if self.q_update == 'single':
                        action_Q = target_q[k]
                        if self.jsd_reward:
                            Q_bootstrap_label = js_divergence_batch[k] + self.gamma * np.max(action_Q)
                        else:
                            Q_bootstrap_label = r_batch_new[k] + js_divergence_batch[k] + self.gamma * np.max(action_Q)
                    elif self.q_update == 'double':
                        action_Q = action_q[k]
                        argmax_tuple = np.unravel_index(np.argmax(action_Q, axis=None), action_Q.shape)
                        value_Q = target_q[k][argmax_tuple]
                        if not self.jsd_reward:
                            Q_bootstrap_label = r_batch_new[k] + js_divergence_batch[k] + self.gamma * value_Q
                        else:
                            Q_bootstrap_label = js_divergence_batch[k] + self.gamma * value_Q

                y_i.append(Q_bootstrap_label)

            reshaped_yi = np.vstack([np.expand_dims(x, 0) for x in y_i])

            currentLoss = self.dqn.compute_loss(s_batch_new, a_batch_one_hot_new, reshaped_yi)

            with open(self.log_path, 'a') as file:
                file.write(str(currentLoss) + "\n")

    def compute_js_divergence(self, P, Q):

        M = [p + q for p, q in zip(P, Q)]
        return 0.5 * (entropy(P, M, base=2) + entropy(Q, M, base=2))

# END OF FILE

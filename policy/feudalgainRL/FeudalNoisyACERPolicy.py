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

import copy
import numpy as np
import scipy
import scipy.signal
import pickle as pickle
import utils
import ontology.FlatOntologyManager as FlatOnt
import tensorflow as tf
import policy.feudalgainRL.noisyacer as noisy_acer
import policy.Policy
import policy.SummaryAction

from policy.feudalgainRL.NoisyACERPolicy import NoisyACERPolicy
from scipy.stats import entropy
from utils.Settings import config as cfg
from utils import ContextLogger
from policy.DRL.replay_buffer_episode_acer import ReplayBufferEpisode
from policy.DRL.replay_prioritised_episode import ReplayPrioritisedEpisode
from policy.Policy import TerminalAction, TerminalState
from policy.feudalgainRL.DIP_parametrisation import DIP_state, padded_state

logger = utils.ContextLogger.getLogger('')

# Discounting function used to calculate discounted returns.
def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


class FeudalNoisyACERPolicy(NoisyACERPolicy):
    '''Derived from :class:`Policy`
    '''
    def __init__(self, in_policy_file, out_policy_file, domainString='CamRestaurants', is_training=False,
                 action_names=None, slot=None, sd_state_dim=50, load_policy=True):
        super(FeudalNoisyACERPolicy, self).__init__(in_policy_file, out_policy_file, domainString, is_training)

        tf.reset_default_graph()

        self.in_policy_file = in_policy_file
        self.out_policy_file = out_policy_file
        self.is_training = is_training
        self.accum_belief = []
        self.prev_state_check = None
        self.sd_state_dim = sd_state_dim

        self.domainString = domainString
        self.domainUtil = FlatOnt.FlatDomainOntology(self.domainString)

        self.features = 'dip'
        self.sd_enc_size = 80
        self.si_enc_size = 40
        self.dropout_rate = 0.
        if cfg.has_option('feudalpolicy', 'features'):
            self.features = cfg.get('feudalpolicy', 'features')
        if cfg.has_option('feudalpolicy', 'sd_enc_size'):
            self.sd_enc_size = cfg.getint('feudalpolicy', 'sd_enc_size')
        if cfg.has_option('feudalpolicy', 'si_enc_size'):
            self.si_enc_size = cfg.getint('feudalpolicy', 'si_enc_size')
        if cfg.has_option('dqnpolicy', 'dropout_rate') and self.is_training:
            self.dropout_rate = cfg.getfloat('feudalpolicy', 'dropout_rate')
        if cfg.has_option('dqnpolicy', 'dropout_rate') and self.is_training:
            self.dropout_rate = cfg.getfloat('feudalpolicy', 'dropout_rate')
        self.actfreq_ds = False
        if cfg.has_option('feudalpolicy', 'actfreq_ds'):
            self.actfreq_ds = cfg.getboolean('feudalpolicy', 'actfreq_ds')
        self.noisy_acer = False
        if cfg.has_option('policy', 'noisy_acer'):
            self.noisy_acer = cfg.getboolean('policy', 'noisy_acer')

        self.sample_argmax = False
        if cfg.has_option('policy', 'sample_argmax'):
            self.sample_argmax = cfg.getboolean('policy', 'sample_argmax')

        if self.sample_argmax:
            print("We sample argmax")

        self.load_policy = load_policy

        # init session
        self.sess = tf.Session()
        with tf.device("/cpu:0"):

            np.random.seed(self.randomseed)
            tf.set_random_seed(self.randomseed)

            # initialise an replay buffer
            if self.replay_type == 'vanilla':
                self.episodes[self.domainString] = ReplayBufferEpisode(self.capacity, self.minibatch_size, self.randomseed)
            elif self.replay_type == 'prioritized':
                self.episodes[self.domainString] = ReplayPrioritisedEpisode(self.capacity, self.minibatch_size, self.randomseed)

            self.samplecount = 0
            self.episodecount = 0

            # construct the models
            self.state_dim = 89  # current DIP state dim
            self.summaryaction = policy.SummaryAction.SummaryAction(domainString)
            self.action_names = action_names
            self.action_dim = len(self.action_names)
            self.stats = [0 for _ in range(self.action_dim)]

            self.global_mu = [0. for _ in range(self.action_dim)]

            si_state_dim = 73
            if self.actfreq_ds:
                if self.domainString == 'CamRestaurants':
                    si_state_dim += 9#16
                elif self.domainString == 'SFRestaurants':
                    si_state_dim += 9#25
                elif self.domainString == 'Laptops11':
                    si_state_dim += 9#40

            self.state_dim = si_state_dim
            self.sacer = noisy_acer.NoisyACERNetwork(self.sess, self.state_dim, self.action_dim,
                                                self.critic_lr, self.delta, self.c, self.alpha, self.h1_size,
                                                self.h2_size, self.is_training,
                                                noisy_acer=self.noisy_acer)

            # when all models are defined, init all variables
            init_op = tf.global_variables_initializer()
            self.sess.run(init_op)

            if self.load_policy:
                self.loadPolicy(self.in_policy_file)
                print('loaded replay size: ', self.episodes[self.domainString].size())
            else:
                print("We do not load a previous policy.")

            #self.acer.update_target_network()

    # def record() has been handled...

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

    def record(self, reward, domainInControl=None, weight=None, state=None, action=None):
        if domainInControl is None:
            domainInControl = self.domainString
        if self.actToBeRecorded is None:
            self.actToBeRecorded = self.summaryAct

        if state is None:
            state = self.prevbelief
        if action is None:
            action = self.actToBeRecorded
        mu_weight = self.prev_mu
        mask = self.prev_mask
        if action == self.action_dim-1: # pass action was taken
            mask = np.zeros(self.action_dim)
            mu_weight = np.ones(self.action_dim)/self.action_dim

        cState, cAction = state, action

        reward /= 20.0

        value = self.sacer.predict_value([cState[0]], [mask])

        if self.replay_type == 'vanilla':
            self.episodes[domainInControl].record(state=cState, \
                    state_ori=state, action=cAction, reward=reward, value=value[0], distribution=mu_weight, mask=mask)
        elif self.replay_type == 'prioritized':
            self.episodes[domainInControl].record(state=cState, \
                    state_ori=state, action=cAction, reward=reward, value=value[0], distribution=mu_weight, mask=mask)

        self.actToBeRecorded = None
        self.samplecount += 1
        return

    def finalizeRecord(self, reward, domainInControl=None):
        if domainInControl is None:
            domainInControl = self.domainString
        if self.episodes[domainInControl] is None:
            logger.warning("record attempted to be finalized for domain where nothing has been recorded before")
            return

        # normalising total return to -1~1
        reward /= 20.0

        terminal_state, terminal_action = self.convertStateAction(TerminalState(), TerminalAction())
        value = 0.0 # not effect on experience replay

        def calculate_discountR_advantage(r_episode, v_episode):
            #########################################################################
            # Here we take the rewards and values from the rollout, and use them to
            # generate the advantage and discounted returns.
            # The advantage function uses "Generalized Advantage Estimation"
            bootstrap_value = 0.0
            self.r_episode_plus = np.asarray(r_episode + [bootstrap_value])
            discounted_r_episode = discount(self.r_episode_plus,self.gamma)[:-1]
            self.v_episode_plus = np.asarray(v_episode + [bootstrap_value])
            advantage = r_episode + self.gamma * self.v_episode_plus[1:] - self.v_episode_plus[:-1]
            advantage = discount(advantage,self.gamma)
            #########################################################################
            return discounted_r_episode, advantage

        self.episodes[domainInControl].record(state=terminal_state, \
                state_ori=TerminalState(), action=terminal_action, reward=reward, value=value, terminal=True, distribution=None)

    def compute_responsible_q(self, inputs, actions, mask):
        return self.sacer.compute_responsible_q(inputs, actions, mask)

    def nextAction(self, beliefstate, execMask):
        '''
        select next action

        :param beliefstate:
        :param hyps:
        :returns: (int) next summarye action
        '''

        action_prob = self.sacer.predict_policy(np.reshape(beliefstate, (1, len(beliefstate))),
                                                np.reshape(execMask, (1, len(execMask))))[0]

        if (self.exploration_type == 'e-greedy' or not self.is_training) and not self.noisy_acer:

            if not self.sample_argmax:
                epsilon = self.epsilon if self.is_training else 0.
                eps_prob = np.ones(len(action_prob)) / len(action_prob)

                best_index = np.argmax(action_prob)
                best_prob = [1. if i == best_index else 0. for i in range(len(action_prob))]

                # we sample a random action with probability epsilon and sample from the policy distribution with probability 1-epsilon
                action_prob = epsilon * np.array(eps_prob) + (1. - epsilon) * action_prob

                if not self.is_training:
                    # take the greedy action during evaluation
                    action_prob = np.array(best_prob)
            else:
                if self.is_training and utils.Settings.random.rand() < self.epsilon:
                    action_prob = np.random.rand(len(self.action_names))

        if not self.is_training:
            # take the greedy action during evaluation
            best_index = np.argmax(action_prob)
            best_prob = [1. if i == best_index else 0. for i in range(len(action_prob))]
            action_prob = np.array(best_prob)

        if not self.sample_argmax:
            nextaIdex = np.random.choice(len(action_prob), p=action_prob / sum(action_prob))
        else:
            nextaIdex = np.argmax(action_prob)
        mu = action_prob / sum(action_prob)

        self.prev_mu = mu
        self.prev_mask = execMask

        return np.array([1. if i == nextaIdex else 0. for i in range(len(action_prob))])

    def train(self, critic_regularizer=None):
        '''
        call this function when the episode ends
        '''
        USE_GLOBAL_MU = False
        self.episode_ct += 1

        if not self.is_training:
            logger.info("Not in training mode")
            return
        else:
            logger.info("Update acer policy parameters.")

        self.episodecount += 1
        logger.info("Sample Num so far: %s" % (self.samplecount))
        logger.info("Episode Num so far: %s" % (self.episodecount))
        if self.samplecount >= self.minibatch_size * 3 and self.episodecount % self.training_frequency == 0:
            logger.info('start trainig...')

            for _ in range(self.train_iters_per_episode):

                if self.replay_type == 'vanilla' or self.replay_type == 'prioritized':
                    s_batch_full, s_ori_batch, a_batch, r_batch, s2_batch_full, s2_ori_batch, t_batch, idx_batch, v_batch, mu_policy, mask_batch = \
                        self.episodes[self.domainString].sample_batch()
                    if USE_GLOBAL_MU:
                        mu_sum = sum(self.global_mu)
                        mu_normalised = np.array([c / mu_sum for c in self.global_mu])
                        mu_policy = [[mu_normalised for _ in range(len(mu_policy[i]))] for i in range(len(mu_policy))]
                else:
                    assert False  # not implemented yet

                s_batch = [[state_tuple[0] for state_tuple in epi] for epi in s_batch_full]

                discounted_r_batch = []
                advantage_batch = []
                def calculate_discountR_advantage(r_episode, v_episode):
                    #########################################################################
                    # Here we take the rewards and values from the rolloutv, and use them to
                    # generate the advantage and discounted returns.
                    # The advantage function uses "Generalized Advantage Estimation"
                    bootstrap_value = 0.0
                    # r_episode rescale by rhos?
                    self.r_episode_plus = np.asarray(r_episode + [bootstrap_value])
                    discounted_r_episode = discount(self.r_episode_plus, self.gamma)[:-1]
                    self.v_episode_plus = np.asarray(v_episode + [bootstrap_value])
                    # change sth here
                    advantage = r_episode + self.gamma * self.v_episode_plus[1:] - self.v_episode_plus[:-1]
                    advantage = discount(advantage, self.gamma)
                    #########################################################################
                    return discounted_r_episode, advantage

                if self.replay_type == 'prioritized':
                    for item_r, item_v, item_idx in zip(r_batch, v_batch, idx_batch):
                        # r, a = calculate_discountR_advantage(item_r, np.concatenate(item_v).ravel().tolist())
                        r, a = calculate_discountR_advantage(item_r, item_v)

                        # flatten nested numpy array and turn it into list
                        discounted_r_batch += r.tolist()
                        advantage_batch += a.tolist()

                        # update the sum-tree
                        # update the TD error of the samples (episode) in the minibatch
                        episodic_TD_error = np.mean(np.absolute(a))
                        self.episodes[self.domainString].update(item_idx, episodic_TD_error)
                else:
                    for item_r, item_v in zip(r_batch, v_batch):
                        # r, a = calculate_discountR_advantage(item_r, np.concatenate(item_v).ravel().tolist())
                        r, a = calculate_discountR_advantage(item_r, item_v)

                        # flatten nested numpy array and turn it into list
                        discounted_r_batch += r.tolist()
                        advantage_batch += a.tolist()

                batch_size = len(s_batch)

                a_batch_one_hot = np.eye(self.action_dim)[np.concatenate(a_batch, axis=0).tolist()]

                r_batch_concatenated = np.concatenate(np.array(r_batch), axis=0)

                loss, entropy, optimize = \
                    self.sacer.train(np.concatenate(np.array(s_batch), axis=0).tolist(), a_batch_one_hot,
                                     np.concatenate(np.array(mask_batch), axis=0).tolist(),
                                     r_batch_concatenated, s_batch, r_batch, self.gamma,
                                     np.concatenate(np.array(mu_policy), axis=0),
                                     discounted_r_batch, advantage_batch)

                ent, norm_loss = entropy/float(batch_size), loss/float(batch_size)

            self.savePolicyInc()

    def savePolicy(self, FORCE_SAVE=False):
        """
        Does not use this, cause it will be called from agent after every episode.
        we want to save the policy only periodically.
        """
        pass

    def compute_js_divergence(self, P, Q):

        M = [p + q for p, q in zip(P, Q)]
        return 0.5 * (entropy(P, M, base=2) + entropy(Q, M, base=2))

    def savePolicyInc(self, FORCE_SAVE=False):
        """
        save model and replay buffer
        """
        if self.episodecount % self.save_step == 0:
            #save_path = self.saver.save(self.sess, self.out_policy_file+'.ckpt')
            self.sacer.save_network(self.out_policy_file+'.acer.ckpt')

            f = open(self.out_policy_file+'.episode', 'wb')
            for obj in [self.samplecount, self.episodes[self.domainString], self.global_mu]:
                pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.close()
            #logger.info("Saving model to %s and replay buffer..." % save_path)

    def loadPolicy(self, filename):
        """
        load model and replay buffer
        """
        # load models
        self.sacer.load_network(filename+'.acer.ckpt')

        # load replay buffer
        if self.load_buffer:
            try:
                print('load from: ', filename)
                f = open(filename+'.episode', 'rb')
                loaded_objects = []
                for i in range(2): # load nn params and collected data
                    loaded_objects.append(pickle.load(f))
                self.samplecount = int(loaded_objects[0])
                self.episodes[self.domainString] = copy.deepcopy(loaded_objects[1])
                self.global_mu = loaded_objects[2]
                logger.info("Loading both model from %s and replay buffer..." % filename)
                f.close()
            except:
                logger.info("Loading only models...")
        else:
            print("We do not load the buffer!")

    def restart(self):
        self.summaryAct = None
        self.lastSystemAction = None
        self.prevbelief = None
        self.prev_mu = None
        self.prev_mask = None
        self.actToBeRecorded = None
        self.epsilon = self.epsilon_start - (self.epsilon_start - self.epsilon_end) * float(self.episodeNum+self.episodecount) / float(self.maxiter)
        self.episode_ave_max_q = []


#END OF FILE

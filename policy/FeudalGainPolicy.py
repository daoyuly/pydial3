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
FeudalGainPolicy.py - Information Gain for FeudalRL policies
==================================================

Copyright 2019-2021 HHU Dialogue Systems and Machine Learning Group

The implementation of the FeudalGain algorithm that incorporates information gain as intrinsic reward in order to update a Feudal policy.
Information gain is defined as the change in probability distributions between consecutive turns in the belief state. The distribution change is measured using the Jensen-Shannon divergence. FeudalGain builds upon the Feudal Dialogue Management architecture and optimises the information-seeking policy to maximise information gain. If the information-seeking policy for instance requests the area of a restaurant, the information gain reward is calculated by the Jensen-Shannon divergence of the value distributions for area before and after the request.


The details can be found here: https://arxiv.org/abs/2109.07129

'''


import numpy as np
import random
import utils
from utils.Settings import config as cfg
from utils import ContextLogger, DiaAct

import ontology.FlatOntologyManager as FlatOnt
from ontology import Ontology
from policy import Policy
from policy import SummaryAction
from policy.feudalgainRL.DIP_parametrisation import DIP_state, padded_state
from policy.feudalgainRL.FeudalNoisyDQNPolicy import FeudalDQNPolicy
from policy.feudalgainRL.FeudalNoisyACERPolicy import FeudalNoisyACERPolicy
from policy.feudalgainRL.feudalUtils import get_feudalAC_masks

logger = utils.ContextLogger.getLogger('')


class FeudalGainPolicy(Policy.Policy):
    '''Derived from :class:`Policy`
    '''

    def __init__(self, in_policy_file, out_policy_file, domainString='CamRestaurants', is_training=False):
        super(FeudalGainPolicy, self).__init__(domainString, is_training)

        self.domainString = domainString
        self.domainUtil = FlatOnt.FlatDomainOntology(self.domainString)
        self.in_policy_file = in_policy_file
        self.out_policy_file = out_policy_file
        self.is_training = is_training
        self.accum_belief = []

        self.prev_state_check = None
        #feudalRL variables
        self.prev_sub_policy = None
        self.prev_master_act = None
        self.prev_master_belief = None
        self.prev_child_act = None
        self.prev_child_belief = None

        self.slots = list(Ontology.global_ontology.get_informable_slots(domainString))

        if 'price' in self.slots:
            self.slots.remove('price')  # remove price from SFR ont, its not used
        if 'name' in self.slots:
            self.slots.remove('name')

        self.features = 'dip'
        if cfg.has_option('feudalpolicy', 'features'):
            self.features = cfg.get('feudalpolicy', 'features')
        self.si_policy_type = 'dqn'
        if cfg.has_option('feudalpolicy', 'si_policy_type'):
            self.si_policy_type = cfg.get('feudalpolicy', 'si_policy_type')
        self.sd_policy_type = 'dqn'
        if cfg.has_option('feudalpolicy', 'sd_policy_type'):
            self.sd_policy_type = cfg.get('feudalpolicy', 'sd_policy_type')
        self.probability_max = 50
        if cfg.has_option('feudalpolicy', 'probability_max'):
            self.probability_max = cfg.get('feudalpolicy', 'probability_max')
        self.info_reward = 0.0
        if cfg.has_option('feudalpolicy', 'info_reward'):
            self.info_reward = cfg.getfloat('feudalpolicy', 'info_reward')
        self.js_threshold = 1.0
        if cfg.has_option('feudalpolicy', 'js_threshold'):
            self.js_threshold = cfg.getfloat('feudalpolicy', 'js_threshold')
        self.jsd_reward = False
        if cfg.has_option('feudalpolicy', 'jsd_reward'):
            self.jsd_reward = cfg.getboolean('feudalpolicy', 'jsd_reward')
        self.jsd_function = None
        if cfg.has_option('feudalpolicy', 'jsd_function'):
            self.jsd_function = cfg.get('feudalpolicy', 'jsd_function')
        self.info_reward_master = 0.0
        if cfg.has_option('feudalpolicy', 'info_reward_master'):
            self.info_reward_master = cfg.getfloat('feudalpolicy', 'info_reward_master')
            print("Master policy trains with info_gain reward")
        self.only_master = False
        if cfg.has_option('feudalpolicy', 'only_master'):
            self.only_master = cfg.getboolean('feudalpolicy', 'only_master')
        if self.only_master:
            print("We train with merged master!")

        self.bye_mask = False
        if cfg.has_option('summaryacts', 'byemask'):
            self.bye_mask = cfg.getboolean('summaryacts', 'byemask')

        self.randomseed = 1234
        if cfg.has_option('GENERAL', 'seed'):
            self.randomseed = cfg.getint('GENERAL', 'seed')

        self.load_master_policy = True
        if cfg.has_option('policy', 'bootstrap_master_policy'):
            self.load_master_policy = cfg.getboolean('policy', 'bootstrap_master_policy')
            print("FeudalAC: BOOTSTRAP MASTER Policy: ", self.load_master_policy)

        # Create the feudal structure (including feudal masks)

        self.summaryaction = SummaryAction.SummaryAction(domainString)
        self.full_action_list = self.summaryaction.action_names
        self.slot_independent_actions = ["inform",
                                            "inform_byname",
                                            "inform_alternatives",
                                            "reqmore",
                                         'bye',
                                         'pass'
                                         ]

        self.slot_specific_actions = ["request",
                                        "confirm",
                                        "select",
                                        'pass']

        self.master_actions = ['slot_ind', 'slot_dep']

        self.chosen = False

        if self.only_master:
            print("Using merged policy pi_mg")
            self.master_actions = self.slot_independent_actions[:-1] + ['slot_dep']
            self.master_policy = FeudalNoisyACERPolicy(self._modify_policyfile('master', in_policy_file),
                                                   self._modify_policyfile('master', out_policy_file),
                                                   domainString=self.domainString, is_training=self.is_training,
                                                   action_names=self.master_actions, sd_state_dim=self.probability_max,
                                                   slot='si', load_policy=self.load_master_policy)

        elif self.si_policy_type == 'acer':
            print("Using policies pi_m and pi_g")
            self.master_policy = FeudalNoisyACERPolicy(self._modify_policyfile('master', in_policy_file),
                                                  self._modify_policyfile('master', out_policy_file),
                                                  domainString=self.domainString, is_training=self.is_training,
                                                  action_names=self.master_actions, sd_state_dim=self.probability_max,
                                                  slot='si')
            self.give_info_policy = FeudalNoisyACERPolicy(self._modify_policyfile('gi', in_policy_file),
                                                     self._modify_policyfile('gi', out_policy_file),
                                                     domainString=self.domainString, is_training=self.is_training,
                                                     action_names=self.slot_independent_actions, slot='si',
                                                     sd_state_dim=self.probability_max)

        self.request_info_policy = FeudalDQNPolicy(self._modify_policyfile('ri', in_policy_file),
                                                   self._modify_policyfile('ri', out_policy_file),
                                                   domainString=self.domainString, is_training=self.is_training,
                                                   action_names=self.slot_specific_actions, slot='sd',
                                                   sd_state_dim=self.probability_max,
                                                   js_threshold=self.js_threshold, info_reward=self.info_reward,
                                                   jsd_reward=self.jsd_reward, jsd_function=self.jsd_function)
        self.critic_regularizer = None

    def _modify_policyfile(self, mod, policyfile):
        pf_split = policyfile.split('/')
        pf_split[-1] = mod + '_' + pf_split[-1]
        return '/'.join(pf_split)

    def act_on(self, state, hyps=None):
        if self.lastSystemAction is None and self.startwithhello:
            systemAct, nextaIdex = 'hello()', -1
            self.chosen_slot_ = None
        else:
            systemAct, nextaIdex = self.nextAction(state)
        self.lastSystemAction = systemAct
        self.summaryAct = nextaIdex
        self.prevbelief = state

        systemAct = DiaAct.DiaAct(systemAct)
        return systemAct

    def record(self, reward, domainInControl=None, weight=None, state=None, action=None):
        self.record_master(reward)
        self.record_childs(reward)

    def finalizeRecord(self, reward, domainInControl=None):
        if domainInControl is None:
            domainInControl = self.domainString
        self.master_policy.finalizeRecord(reward)
        if not self.only_master:
            self.give_info_policy.finalizeRecord(reward)
        self.request_info_policy.finalizeRecord(reward)

        #print("DIALOGUE FINISHED")
        #print("REWARD:", reward)
        #print("\n")

    def record_master(self, reward):
        if self.only_master or self.si_policy_type == 'acer':
            self.master_policy.record(reward, domainInControl=self.domainString,
                                      state=[self.prev_master_belief, self.beliefstate, self.chosen_slot],
                                      action=self.prev_master_act)
        else:
            self.master_policy.record(reward, domainInControl=self.domainString,
                                      state=self.prev_master_belief, action=self.prev_master_act)

    def record_childs(self, reward):
        if self.prev_sub_policy == 'si':
            if not self.only_master:
                self.give_info_policy.record(reward, domainInControl=self.domainString,
                                             state=[self.prev_master_belief, 0 , 0],
                                             action=self.prev_child_act)

            state_for_pi_d = np.concatenate([np.zeros(self.probability_max), self.prev_master_belief])
            state_for_pi_d[0] = 1.0

            self.request_info_policy.record(reward, domainInControl=self.domainString,
                                            state=[state_for_pi_d,
                                                   self.beliefstate, self.chosen_slot, self.dipstatevec_slots],
                                            action=len(self.slot_specific_actions) - 1)
        elif self.prev_sub_policy == 'sd':
            self.request_info_policy.record(reward, domainInControl=self.domainString,
                                            state=[self.prev_child_belief, self.beliefstate, self.chosen_slot, self.dipstatevec_slots],
                                            action=self.prev_child_act)
            if not self.only_master:
                self.give_info_policy.record(reward, domainInControl=self.domainString,
                                             state=[self.prev_master_belief, 0 , 0],
                                             action=len(self.slot_independent_actions) - 1)

    def convertStateAction(self, state, action):
        pass

    def nextAction(self, beliefstate):
        '''
        select next action

        :param beliefstate:
        :returns: (int) next summary action
        '''

        # compute main belief

        if self.features == 'learned' or self.features == 'rnn':
            dipstate = padded_state(beliefstate, domainString=self.domainString, probability_max=self.probability_max)
        else:
            dipstate = DIP_state(beliefstate,domainString=self.domainString)
        dipstatevec = dipstate.get_beliefStateVec('general')

        non_exec = self.summaryaction.getNonExecutable(beliefstate.domainStates[beliefstate.currentdomain], self.lastSystemAction)
        masks = get_feudalAC_masks(non_exec, self.slots, self.slot_independent_actions, self.slot_specific_actions,
                                   only_master=self.only_master)

        master_Q_values = self.master_policy.nextAction(dipstatevec, masks["master"])
        #TODO: MASTER ACTIONS ARE NOT MASKED, ONLY COMPLETELY VALID FOR ENV4 ATM
        master_decision = np.argmax(master_Q_values)
        self.prev_master_act = master_decision
        self.prev_master_belief = dipstatevec
        self.beliefstate = beliefstate.domainStates[beliefstate.currentdomain]

        self.dipstatevec_slots, self.maskvec_slots = self.get_dipstate_vec_slots_and_masks(dipstate, masks)
        self.slot_beliefs = self.get_slot_beliefs(dipstate)

        if self.master_actions[master_decision] != 'slot_dep':
            # drop to give_info policy
            self.prev_sub_policy = 'si'
            if not self.only_master:
                child_Q_values = self.give_info_policy.nextAction(dipstatevec, masks['give_info'])
                child_Q_values = np.add(child_Q_values, masks['give_info'])
                #TODO: sample from the distribution instead of argmax..
                child_decision = np.argmax(child_Q_values)
                summaryAct = self.slot_independent_actions[child_decision]
                self.prev_child_act = child_decision
                self.prev_child_belief = dipstatevec
            else:
                summaryAct = self.master_actions[master_decision]
            self.chosen_slot = "None"
        else:
            self.prev_sub_policy = 'sd'

            child_Q_values = self.request_info_policy.nextAction(self.dipstatevec_slots)
            #if we chose randomly, child_Q_values is of shape len(actions), else shape=(number_slots, len(actions))
            if len(child_Q_values.shape) == 1:
                #we chose a random action, now we need a random slot to it
                random_slot = random.choice(self.slots)
                child_Q_values = np.add(child_Q_values, masks['req_info'][random_slot])
                child_decision = np.argmax(child_Q_values)
                self.prev_child_act = child_decision
                self.prev_child_belief = dipstate.get_beliefStateVec(random_slot)
                self.chosen_slot = random_slot
                summaryAct = self.slot_specific_actions[child_decision] + "_" + random_slot
            else:
                child_Q_values = np.add(child_Q_values, self.maskvec_slots)
                child_decision = np.unravel_index(np.argmax(child_Q_values, axis=None), child_Q_values.shape)
                #child_decision is tuple of length 2!
                chosen_slot = child_decision[0]
                chosen_action = child_decision[1]
                self.chosen_slot = self.slots[chosen_slot]
                self.chosen_slot_ = self.slots[chosen_slot]
                self.prev_child_act = chosen_action
                self.prev_child_belief = dipstate.get_beliefStateVec(self.slots[chosen_slot])
                summaryAct = self.slot_specific_actions[chosen_action] + "_" + self.slots[chosen_slot]
                self.chosen = True

        #if self.chosen_slot_:
        #    print(self.chosen_slot_)
        #    keys = self.beliefstate['beliefs'][self.chosen_slot_].keys()
        #    b = [self.beliefstate['beliefs'][self.chosen_slot_]['**NONE**']] + \
        #        [self.beliefstate['beliefs'][self.chosen_slot_][value] for value in list(keys) if value != '**NONE**']
        #    print(f"DISTRIBUTION FOR SLOT {self.chosen_slot_}:", b)

        beliefstate = beliefstate.getDomainState(self.domainUtil.domainString)
        masterAct = self.summaryaction.Convert(beliefstate, summaryAct, self.lastSystemAction)
        nextaIdex = self.full_action_list.index(summaryAct)

        return masterAct, nextaIdex

    def train(self):
        '''
        call this function when the episode ends
        '''
        self.master_policy.train(self.critic_regularizer)
        if not self.only_master:
            self.give_info_policy.train()
        self.request_info_policy.train()

    def get_slot_beliefs(self, dipstate):

        slot_beliefs = []
        for slot in self.slots:
            slot_dependent_vec = dipstate.get_beliefStateVec(slot)
            slot_beliefs.append(slot_dependent_vec)
        return np.concatenate(slot_beliefs, axis=0)

    def get_dipstate_vec_slots_and_masks(self, dipstate, masks):

        dipstatevec_slots = []
        maskvec_slots = []
        for slot in self.slots:
            slot_dependent_vec = dipstate.get_beliefStateVec(slot)
            dipstatevec_slots.append(slot_dependent_vec)
            maskvec_slots.append(masks['req_info'][slot])
        dipstatevec_slots = np.vstack(dipstatevec_slots)
        maskvec_slots = np.asarray(maskvec_slots)

        return dipstatevec_slots, maskvec_slots

    def savePolicy(self, FORCE_SAVE=False):
        """
        Does not use this, cause it will be called from agent after every episode.
        we want to save the policy only periodically.
        """
        pass

    def savePolicyInc(self, FORCE_SAVE=False):
        """
        save model and replay buffer
        """
        # just save each sub-policy
        self.master_policy.savePolicyInc()
        if not self.only_master:
            self.give_info_policy.savePolicyInc()
        self.request_info_policy.savePolicyInc()

    def loadPolicy(self, filename):
        """
        load model and replay buffer
        """
        # load policy models one by one
        pass

    def restart(self):
        self.summaryAct = None
        self.lastSystemAction = None
        self.prevbelief = None
        self.actToBeRecorded = None
        self.master_policy.restart()
        if not self.only_master:
            self.give_info_policy.restart()
        self.request_info_policy.restart()

# END OF FILE

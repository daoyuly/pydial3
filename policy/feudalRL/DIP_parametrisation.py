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
Class to convert belief states into DIP parametrisations
'''

import numpy as np
import copy
from itertools import product
from scipy.stats import entropy

from policy.Policy import Policy, Action, State, TerminalAction, TerminalState
from ontology import Ontology
from utils import Settings, ContextLogger, DialogueState
logger = ContextLogger.getLogger('')

class DIP_state(State):
    def __init__(self, belief, domainString=None, action_freq=None):
        #params
        self.domainString = domainString
        self.N_bins = 10
        self.slots = list(Ontology.global_ontology.get_informable_slots(domainString))
        if 'price' in self.slots:
            self.slots.remove('price') #remove price from SFR ont, its not used

        if 'name' in self.slots:
            self.slots.remove('name')
        self.DIP_state = {'general':None, 'joint':None}
        for slot in self.slots:
            self.DIP_state[slot]=None

        # convert belief state into DIP params
        if action_freq is not None:
            self.DIP_state['general'] = np.concatenate((action_freq,self.convert_general_b(belief)))
        else:
            self.DIP_state['general'] = self.convert_general_b(belief)
        self.DIP_state['joint'] = self.convert_joint_slot_b(belief)
        for slot in self.slots:
            self.DIP_state[slot] = self.convert_slot_b(belief, slot)

        # create DIP vector and masks
        self.get_DIP_vector()
        self.beliefStateVec = None #for compatibility with GP sarsa implementation

    def get_DIP_vector(self):
        """
        convert the DIP state into a numpy vector and a set of masks per slot
        :return:
        """
        pad_v = np.zeros(len(self.DIP_state[self.slots[0]]))
        slot_len = len(pad_v)
        general_len = len(self.DIP_state['general']) + len(self.DIP_state['joint'])
        pad_v[0] = 1.
        self.DIP_vector = [pad_v]
        self.DIP_masks = {}
        mask_template = [False] * (slot_len * (len(self.slots) + 1)) + [True] * general_len
        i = 1
        for slot in self.slots:
            self.DIP_vector.append(self.DIP_state[slot])
            self.DIP_masks[slot] = np.array(mask_template)
            self.DIP_masks[slot][slot_len*i:slot_len*(i+1)] = True
            i += 1
        self.DIP_vector.append(self.DIP_state['general'])
        self.DIP_vector.append(self.DIP_state['joint'])
        self.DIP_masks['general'] = np.array(mask_template)
        self.DIP_masks['general'][:slot_len] = True

        self.DIP_vector = np.concatenate(self.DIP_vector)

    def get_beliefStateVec(self, slot):
        return self.DIP_vector[self.DIP_masks[slot]]

    def get_DIP_state(self, slot):
        return np.array([self.DIP_state['general'] + self.DIP_state['joint'] + self.DIP_state[slot]])

    def get_full_DIP_state(self):
        full_slot_bstate = []
        for slot in self.slots:
            full_slot_bstate += self.DIP_state[slot]
        full_DIP_state = np.array([full_slot_bstate + self.DIP_state['general'] + self.DIP_state['joint']])
        DIP_mask = [True]*(len(self.DIP_state['general']) + len(self.DIP_state['joint'])) + [False] * len(full_slot_bstate)
        return full_DIP_state, DIP_mask

    def convert_general_b(self, belief):
        """
        Extracts from the belief state the DIP vector corresponding to the general features (e.g. method, user act...)
        :param belief: The full belief state
        :return: The DIP general vector
        """
        if type(belief) == DialogueState.DialogueState:
            belief = belief.domainStates[belief.currentdomain]

        dial_act = list(belief['beliefs']['discourseAct'].values())

        requested = self._get_DIP_requested_vector(belief)
        method = list(belief['beliefs']['method'].values())
        features = [int(belief['features']['offerHappened']), int(belief['features']['lastActionInformNone']), int(bool(belief['features']['lastInformedVenue']))]
        discriminable = [int(x) for x in belief['features']['inform_info']]
        slot_n = 1/len(self.slots)
        val_n = []
        for slot in self.slots:
            val_n.append(len(Ontology.global_ontology.get_informable_slot_values(self.domainString, slot)))
        avg_value_n = 1/np.mean(val_n)


        return dial_act + requested + method + features + discriminable + [slot_n, avg_value_n]


    def _get_DIP_requested_vector(self, belief):
        n_requested = sum([x>0.5 for x in list(belief['beliefs']['requested'].values())])
        ret_vec = [0] * 5
        if n_requested > 4:
            n_requested = 4
        ret_vec[n_requested] = 1.
        return ret_vec

    def convert_joint_slot_b(self, belief):
        """
        Extracts the features for the joint DIP vector for all the slots
        :param belief: The full belief state
        :return: The DIP joint slot vector
        """
        if type(belief) == DialogueState.DialogueState:
            belief = belief.domainStates[belief.currentdomain]

        joint_beliefs = []
        joint_none = 1.
        informable_beliefs = [copy.deepcopy(belief['beliefs'][x]) for x in list(belief['beliefs'].keys()) if x in self.slots] # this might be inneficent
        for i, b in enumerate(informable_beliefs):
            joint_none *= b['**NONE**']
            del b['**NONE**'] # should I put **NONE** prob mass to dontcare?
            informable_beliefs[i] = sorted([x for x in list(b.values()) if x != 0], reverse=True)[:2]
            while len(informable_beliefs[i]) < 2:
                informable_beliefs[i].append(0.)
        for probs in product(*informable_beliefs):
            joint_beliefs.append(np.prod(probs))
        j_top = joint_beliefs[0]
        j_2nd = joint_beliefs[1]
        j_3rd = joint_beliefs[2]
        first_joint_beliefs = joint_beliefs[:8]
        if sum(first_joint_beliefs) == 0:
            first_joint_beliefs = np.ones(len(first_joint_beliefs)) / len(first_joint_beliefs)
        else:
            first_joint_beliefs = np.array(first_joint_beliefs) / sum(first_joint_beliefs) # why normalise?

        # difference between 1st and 2dn values
        j_ent = entropy(first_joint_beliefs)
        j_dif = joint_beliefs[0] - joint_beliefs[1]
        j_dif_bin = [0.] * 5
        idx = int((j_dif) * 5)
        if idx == 5:
            idx = 4
        j_dif_bin[idx] = 1

        # number of slots which are not **NONE**
        n = 0
        for key in belief['beliefs']:
            if key in self.slots:
                none_val = belief['beliefs'][key]['**NONE**']
                top_val = np.max([belief['beliefs'][key][value] for value in list(belief['beliefs'][key].keys()) if value != '**NONE**'])
                if top_val > none_val:
                    n += 1
        not_none = [0.] * 5
        if n > 4:
            n = 4
        not_none[n] = 1.

        return [j_top, j_2nd, j_3rd, joint_none, j_ent, j_dif] + j_dif_bin + not_none

    def convert_slot_b(self, belief, slot):
        """
        Extracts the slot DIP features.
        :param belief: The full belief state
        :return: The slot DIP vector
        """
        if type(belief) == DialogueState.DialogueState:
            belief = belief.domainStates[belief.currentdomain]
        b = [belief['beliefs'][slot]['**NONE**']] + sorted([belief['beliefs'][slot][value] for value in list(belief['beliefs'][slot].keys()) if value != '**NONE**'], reverse=True)
        b_top = b[1]
        b_2nd = b[2]
        b_3rd = b[3]
        b_ent = entropy(b)
        b_none = b[0]
        b_dif = b[1] - b[2]
        b_dif_bin = [0.] * 5
        idx = int((b_dif) * 5)
        if idx == 5:
            idx = 4
        b_dif_bin[idx] = 1
        non_zero_rate = [x != 0 for x in b[1:]]
        non_zero_rate = sum(non_zero_rate) / len(non_zero_rate)
        requested_prob = belief['beliefs']['requested'][slot]

        # Ontology and DB based features
        V_len = len(Ontology.global_ontology.get_informable_slot_values(self.domainString, slot))
        norm_N_values = 1 / V_len
        v_len_bin_vector = [0.] * self.N_bins
        v_len_bin_vector[int(np.log2(V_len))] = 1.
        #ocurr_prob, not_occur_prob, first_prob, second_prob, later_prob = self._get_importance_and_priority(slot) # this was manually set in the original DIP paper, I think it can be learned from the other features
        val_dist_in_DB = self._get_val_dist_in_DB(slot)
        # potential_contr_to_DB_search = self._get_potential_contr_to_DB_search(slot, belief)
        #potential_contr_to_DB_search = [0, 0, 0, 0] # the implementation of this method is too slow right now, dont knwo how useful these features are (but they seem quite useful)
        return [0, b_top, b_2nd, b_3rd, b_ent, b_none, non_zero_rate, requested_prob, norm_N_values, val_dist_in_DB] + b_dif_bin + v_len_bin_vector

    def _get_val_dist_in_DB(self, slot):
        # The entropy of the normalised histogram (|DB(s=v)|/|DB|) \forall v \in V_s
        values = Ontology.global_ontology.get_informable_slot_values(self.domainString, slot)
        entities = Ontology.global_ontology.entity_by_features(self.domainString, {})
        val_dist = np.zeros(len(values))
        n = 0
        for ent in entities:
            if ent[slot] != 'not available':
                val_dist[values.index(ent[slot])] += 1
                n += 1
        return entropy(val_dist/n)


class padded_state(State):
    def __init__(self, belief, domainString=None, action_freq=None):
        #params
        self.domainString = domainString
        self.sortbelief = False
        #self.action_freq = False
        if Settings.config.has_option('feudalpolicy', 'sortbelief'):
            self.sortbelief = Settings.config.getboolean('feudalpolicy', 'sortbelief')
        #if Settings.config.has_option('feudalpolicy', 'action_freq'):
        #    self.action_freq = Settings.config.getboolean('feudalpolicy', 'action_freq')
        self.slots = list(Ontology.global_ontology.get_informable_slots(domainString))
        if 'price' in self.slots:
            self.slots.remove('price') #remove price from SFR ont, its not used

        if 'name' in self.slots:
            self.slots.remove('name')

        slot_values = Ontology.global_ontology.get_informable_slots_and_values(domainString)
        self.max_v = np.max([len(slot_values[s]) for s in self.slots]) + 3 # (+**NONE**+dontcare+pad)
        self.max_v = 158
        self.si_size = 72 # size of general plus joint vectors
        self.sd_size = self.max_v

        self.DIP_state = {'general':None, 'joint':None}
        for slot in self.slots:
            self.DIP_state[slot]=None

        # convert belief state into DIP params
        if action_freq is not None:
            self.DIP_state['general'] = np.concatenate((action_freq,self.convert_general_b(belief)))
        else:
            self.DIP_state['general'] = self.convert_general_b(belief)
        self.DIP_state['joint'] = self.convert_joint_slot_b(belief)
        for slot in self.slots:
            self.DIP_state[slot] = self.convert_slot_b(belief, slot)

        # create vector and masks
        self.get_DIP_vector()
        self.beliefStateVec = None #for compatibility with GP sarsa implementation

    def get_DIP_vector(self):
        """
        convert the state into a numpy vector and a set of masks per slot
        :return:
        """
        pad_v = np.zeros(len(self.DIP_state[self.slots[0]]))
        slot_len = len(pad_v)
        general_len = len(self.DIP_state['general']) + len(self.DIP_state['joint'])
        pad_v[0] = 1.
        self.DIP_vector = [pad_v]
        self.DIP_masks = {}
        mask_template = [False] * (slot_len * (len(self.slots) + 1)) + [True] * general_len
        i = 1
        for slot in self.slots:
            self.DIP_vector.append(self.DIP_state[slot])
            self.DIP_masks[slot] = np.array(mask_template)
            self.DIP_masks[slot][slot_len*i:slot_len*(i+1)] = True
            i += 1
        self.DIP_vector.append(self.DIP_state['general'])
        self.DIP_vector.append(self.DIP_state['joint'])
        self.DIP_masks['general'] = np.array(mask_template)
        self.DIP_masks['general'][:slot_len] = True

        self.DIP_vector = np.concatenate(self.DIP_vector)

    def get_beliefStateVec(self, slot):
        return self.DIP_vector[self.DIP_masks[slot]]

    def get_DIP_state(self, slot):
        return np.array([self.DIP_state['general'] + self.DIP_state['joint'] + self.DIP_state[slot]])

    def get_full_DIP_state(self):
        full_slot_bstate = []
        for slot in self.slots:
            full_slot_bstate += self.DIP_state[slot]
        full_DIP_state = np.array([full_slot_bstate + self.DIP_state['general'] + self.DIP_state['joint']])
        DIP_mask = [True]*(len(self.DIP_state['general']) + len(self.DIP_state['joint'])) + [False] * len(full_slot_bstate)
        return full_DIP_state, DIP_mask

    def convert_general_b(self, belief):
        """
        Extracts from the belief state the vector corresponding to the general features (e.g. method, user act...)
        :param belief: The full belief state
        :return: The general vector
        """
        if type(belief) == DialogueState.DialogueState:
            belief = belief.domainStates[belief.currentdomain]

        dial_act = list(belief['beliefs']['discourseAct'].values())

        requested = self._get_requested_vector(belief)
        method = list(belief['beliefs']['method'].values())
        features = [int(belief['features']['offerHappened']), int(belief['features']['lastActionInformNone']),
                    int(bool(belief['features']['lastInformedVenue']))]
        discriminable = [int(x) for x in belief['features']['inform_info']]

        return dial_act + requested + method + features + discriminable

    def _get_requested_vector(self, belief):
        n_requested = sum([x>0.5 for x in list(belief['beliefs']['requested'].values())])
        ret_vec = [0] * 5
        if n_requested > 4:
            n_requested = 4
        ret_vec[n_requested] = 1.
        return ret_vec

    def convert_joint_slot_b(self, belief):
        """
            Extracts the features for the joint vector of all the slots
            :param belief: The full belief state
            :return: The joint slot vector
            """
        #ic340 note: this should probably be done with an rnn encoder
        if type(belief) == DialogueState.DialogueState:
            belief = belief.domainStates[belief.currentdomain]

        joint_beliefs = []
        joint_none = 1.
        informable_beliefs = [copy.deepcopy(belief['beliefs'][x]) for x in list(belief['beliefs'].keys()) if
                              x in self.slots]  # this might be inneficent
        for i, b in enumerate(informable_beliefs):
            joint_none *= b['**NONE**']
            del b['**NONE**']  # should I put **NONE** prob mass to dontcare?
            informable_beliefs[i] = sorted([x for x in list(b.values()) if x != 0], reverse=True)[:2]
            while len(informable_beliefs[i]) < 2:
                informable_beliefs[i].append(0.)
        for probs in product(*informable_beliefs):
            joint_beliefs.append(np.prod(probs))
        first_joint_beliefs = -np.ones(20)
        joint_beliefs = joint_beliefs[:20]
        len_joint_beliefs = len(joint_beliefs)
        first_joint_beliefs[:len_joint_beliefs] = joint_beliefs

        if sum(first_joint_beliefs) == 0:
            first_joint_beliefs = list(np.ones(len(first_joint_beliefs)) / len(first_joint_beliefs))
        else:
            first_joint_beliefs = list(np.array(first_joint_beliefs) / sum(first_joint_beliefs))  # why normalise?

        # number of slots which are not **NONE**
        n = 0
        for key in belief['beliefs']:
            if key in self.slots:
                none_val = belief['beliefs'][key]['**NONE**']
                top_val = np.max(
                    [belief['beliefs'][key][value] for value in list(belief['beliefs'][key].keys()) if value != '**NONE**'])
                if top_val > none_val:
                    n += 1
        not_none = [0.] * 5
        if n > 4:
            n = 4
        not_none[n] = 1.

        return [joint_none] + first_joint_beliefs + not_none

    def convert_slot_b(self, belief, slot):
        """
        Extracts the slot features by padding the distribution vector with -1s.
        :param belief: The full belief state
        :return: The slot DIP vector
        """
        if type(belief) == DialogueState.DialogueState:
            belief = belief.domainStates[belief.currentdomain]
        if self.sortbelief is True:
            b = [belief['beliefs'][slot]['**NONE**']] + sorted(
                [belief['beliefs'][slot][value] for value in list(belief['beliefs'][slot].keys()) if value != '**NONE**'],
                reverse=True) # sorted values

        else:
            b = [belief['beliefs'][slot]['**NONE**']] + \
                [belief['beliefs'][slot][value] for value in list(belief['beliefs'][slot].keys()) if value != '**NONE**'] # unsorted values

        assert len(b) <= self.max_v -1, 'length of bstate ({}) is longer than self.max_v ({})'.format(len(b), self.max_v-1)
        padded_b = -np.ones(self.max_v)
        padded_b[0] = 0.
        padded_b[1:len(b)+1] = b
        return padded_b

    def _get_val_dist_in_DB(self, slot):
        # The entropy of the normalised histogram (|DB(s=v)|/|DB|) \forall v \in V_s
        values = Ontology.global_ontology.get_informable_slot_values(self.domainString, slot)
        entities = Ontology.global_ontology.entity_by_features(self.domainString, {})
        val_dist = np.zeros(len(values))
        n = 0
        for ent in entities:
            if ent[slot] != 'not available':
                val_dist[values.index(ent[slot])] += 1
                n += 1
        return entropy(val_dist/n)


def get_test_beliefs():
    b1 = {'beliefs': {'allowedforkids': {'**NONE**': 0.0,
   '0': 0.0,
   '1': 0.0,
   'dontcare': 1.0},
  'area': {'**NONE**': 1.0,
   'alamo square': 0.0,
   'amanico ergina village': 0.0,
   'anza vista': 0.0,
   'ashbury heights': 0.0,
   'balboa terrace': 0.0,
   'bayview district': 0.0,
   'bayview heights': 0.0,
   'bernal heights': 0.0,
   'bernal heights north': 0.0,
   'bernal heights south': 0.0,
   'buena vista park': 0.0,
   'castro': 0.0,
   'cathedral hill': 0.0,
   'cayuga terrace': 0.0,
   'central richmond': 0.0,
   'central sunset': 0.0,
   'central waterfront': 0.0,
   'chinatown': 0.0,
   'civic center': 0.0,
   'clarendon heights': 0.0,
   'cole valley': 0.0,
   'corona heights': 0.0,
   'cow hollow': 0.0,
   'crocker amazon': 0.0,
   'diamond heights': 0.0,
   'doelger city': 0.0,
   'dogpatch': 0.0,
   'dolores heights': 0.0,
   'dontcare': 0.0,
   'downtown': 0.0,
   'duboce triangle': 0.0,
   'embarcadero': 0.0,
   'eureka valley': 0.0,
   'eureka valley dolores heights': 0.0,
   'excelsior': 0.0,
   'financial district': 0.0,
   'financial district south': 0.0,
   'fishermans wharf': 0.0,
   'forest hill': 0.0,
   'forest hill extension': 0.0,
   'forest knolls': 0.0,
   'fort mason': 0.0,
   'fort winfield scott': 0.0,
   'frederick douglass haynes gardens': 0.0,
   'friendship village': 0.0,
   'glen park': 0.0,
   'glenridge': 0.0,
   'golden gate heights': 0.0,
   'golden gate park': 0.0,
   'haight ashbury': 0.0,
   'hayes valley': 0.0,
   'hunters point': 0.0,
   'india basin': 0.0,
   'ingleside': 0.0,
   'ingleside heights': 0.0,
   'ingleside terrace': 0.0,
   'inner mission': 0.0,
   'inner parkside': 0.0,
   'inner richmond': 0.0,
   'inner sunset': 0.0,
   'inset': 0.0,
   'jordan park': 0.0,
   'laguna honda': 0.0,
   'lake': 0.0,
   'lake shore': 0.0,
   'lakeside': 0.0,
   'laurel heights': 0.0,
   'lincoln park': 0.0,
   'lincoln park lobos': 0.0,
   'little hollywood': 0.0,
   'little italy': 0.0,
   'little osaka': 0.0,
   'little russia': 0.0,
   'lone mountain': 0.0,
   'lower haight': 0.0,
   'lower nob hill': 0.0,
   'lower pacific heights': 0.0,
   'malcolm x square': 0.0,
   'marcus garvey square': 0.0,
   'marina district': 0.0,
   'martin luther king square': 0.0,
   'mastro': 0.0,
   'merced heights': 0.0,
   'merced manor': 0.0,
   'midtown terrace': 0.0,
   'miraloma park': 0.0,
   'mission bay': 0.0,
   'mission district': 0.0,
   'mission dolores': 0.0,
   'mission terrace': 0.0,
   'monterey heights': 0.0,
   'mount davidson manor': 0.0,
   'nob hill': 0.0,
   'noe valley': 0.0,
   'noma': 0.0,
   'north beach': 0.0,
   'north panhandle': 0.0,
   'north park': 0.0,
   'north waterfront': 0.0,
   'oceanview': 0.0,
   'opera plaza': 0.0,
   'outer mission': 0.0,
   'outer parkside': 0.0,
   'outer richmond': 0.0,
   'outer sunset': 0.0,
   'outset': 0.0,
   'pacific heights': 0.0,
   'panhandle': 0.0,
   'park merced': 0.0,
   'parkmerced': 0.0,
   'parkside': 0.0,
   'pine lake park': 0.0,
   'portola': 0.0,
   'potrero flats': 0.0,
   'potrero hill': 0.0,
   'presidio': 0.0,
   'presidio heights': 0.0,
   'richmond district': 0.0,
   'russian hill': 0.0,
   'saint francis wood': 0.0,
   'san francisco airport': 0.0,
   'san francisco state university': 0.0,
   'sea cliff': 0.0,
   'sherwood forest': 0.0,
   'showplace square': 0.0,
   'silver terrace': 0.0,
   'somisspo': 0.0,
   'south basin': 0.0,
   'south beach': 0.0,
   'south of market': 0.0,
   'st francis square': 0.0,
   'st francis wood': 0.0,
   'stonestown': 0.0,
   'sunnydale': 0.0,
   'sunnyside': 0.0,
   'sunset district': 0.0,
   'telegraph hill': 0.0,
   'tenderloin': 0.0,
   'thomas paine square': 0.0,
   'transmission': 0.0,
   'treasure island': 0.0,
   'twin peaks': 0.0,
   'twin peaks west': 0.0,
   'upper market': 0.0,
   'van ness': 0.0,
   'victoria mews': 0.0,
   'visitacion valley': 0.0,
   'vista del monte': 0.0,
   'west of twin peaks': 0.0,
   'west portal': 0.0,
   'western addition': 0.0,
   'westlake and olympic': 0.0,
   'westwood highlands': 0.0,
   'westwood park': 0.0,
   'yerba buena island': 0.0,
   'zion district': 0.0},
  'discourseAct': {'ack': 0.0,
   'bye': 0.0,
   'hello': 0.0,
   'none': 1.0,
   'repeat': 0.0,
   'silence': 0.0,
   'thankyou': 0.0},
  'food': {'**NONE**': 0.0,
   'afghan': 0.0,
   'arabian': 0.0,
   'asian': 0.0,
   'basque': 0.0,
   'brasseries': 0.0,
   'brazilian': 0.0,
   'buffets': 0.0,
   'burgers': 0.0,
   'burmese': 0.0,
   'cafes': 0.0,
   'cambodian': 0.0,
   'cantonese': 1.0,
   'chinese': 0.0,
   'comfort food': 0.0,
   'creperies': 0.0,
   'dim sum': 0.0,
   'dontcare': 0.0,
   'ethiopian': 0.0,
   'ethnic food': 0.0,
   'french': 0.0,
   'gluten free': 0.0,
   'himalayan': 0.0,
   'indian': 0.0,
   'indonesian': 0.0,
   'indpak': 0.0,
   'italian': 0.0,
   'japanese': 0.0,
   'korean': 0.0,
   'kosher': 0.0,
   'latin': 0.0,
   'lebanese': 0.0,
   'lounges': 0.0,
   'malaysian': 0.0,
   'mediterranean': 0.0,
   'mexican': 0.0,
   'middle eastern': 0.0,
   'modern european': 0.0,
   'moroccan': 0.0,
   'new american': 0.0,
   'pakistani': 0.0,
   'persian': 0.0,
   'peruvian': 0.0,
   'pizza': 0.0,
   'raw food': 0.0,
   'russian': 0.0,
   'sandwiches': 0.0,
   'sea food': 0.0,
   'shanghainese': 0.0,
   'singaporean': 0.0,
   'soul food': 0.0,
   'spanish': 0.0,
   'steak': 0.0,
   'sushi': 0.0,
   'taiwanese': 0.0,
   'tapas': 0.0,
   'thai': 0.0,
   'traditionnal american': 0.0,
   'turkish': 0.0,
   'vegetarian': 0.0,
   'vietnamese': 0.0},
  'goodformeal': {'**NONE**': 0.0,
   'breakfast': 0.0,
   'brunch': 0.0,
   'dinner': 0.0,
   'dontcare': 1.0,
   'lunch': 0.0},
  'method': {'byalternatives': 0.0,
   'byconstraints': 0.0,
   'byname': 0.9285714285714286,
   'finished': 0.0,
   'none': 0.0714285714285714,
   'restart': 0.0},
  'name': {'**NONE**': 0.0,
   'a 16': 0.0,
   'a la turca restaurant': 0.0,
   'abacus': 0.0,
   'alamo square seafood grill': 0.0,
   'albona ristorante istriano': 0.0,
   'alborz persian cuisine': 0.0,
   'allegro romano': 0.0,
   'amarena': 0.0,
   'amber india': 0.0,
   'ame': 0.0,
   'ananda fuara': 0.0,
   'anchor oyster bar': 0.0,
   'angkor borei restaurant': 0.0,
   'aperto restaurant': 0.0,
   'ar roi restaurant': 0.0,
   'arabian nights restaurant': 0.0,
   'assab eritrean restaurant': 0.0,
   'atelier crenn': 0.0,
   'aux delices restaurant': 0.0,
   'aziza': 0.0,
   'b star bar': 0.0,
   'bar crudo': 0.0,
   'beijing restaurant': 0.0,
   'bella trattoria': 0.0,
   'benu': 0.0,
   'betelnut': 0.0,
   'bistro central parc': 0.0,
   'bix': 0.0,
   'borgo': 0.0,
   'borobudur restaurant': 0.0,
   'bouche': 0.0,
   'boulevard': 0.0,
   'brothers restaurant': 0.0,
   'bund shanghai restaurant': 0.0,
   'burma superstar': 0.0,
   'butterfly': 0.0,
   'cafe claude': 0.0,
   'cafe jacqueline': 0.0,
   'campton place restaurant': 0.0,
   'canteen': 0.0,
   'canto do brasil restaurant': 0.0,
   'capannina': 0.0,
   'capital restaurant': 0.0,
   'chai yo thai restaurant': 0.0,
   'chaya brasserie': 0.0,
   'chenery park': 0.0,
   'chez maman': 0.0,
   'chez papa bistrot': 0.0,
   'chez spencer': 0.0,
   'chiaroscuro': 0.0,
   'chouchou': 0.0,
   'chow': 0.0,
   'city view restaurant': 0.0,
   'claudine': 0.0,
   'coi': 0.0,
   'colibri mexican bistro': 0.0,
   'coqueta': 0.0,
   'crustacean restaurant': 0.0,
   'da flora a venetian osteria': 0.0,
   'darbar restaurant': 0.0,
   'delancey street restaurant': 0.0,
   'delfina': 0.0,
   'dong baek restaurant': 0.0,
   'dontcare': 0.0,
   'dosa on fillmore': 0.0,
   'dosa on valencia': 0.0,
   'eiji': 0.0,
   'enjoy vegetarian restaurant': 0.0,
   'espetus churrascaria': 0.0,
   'fang': 0.0,
   'farallon': 0.0,
   'fattoush restaurant': 0.0,
   'fifth floor': 0.0,
   'fino restaurant': 0.0,
   'firefly': 0.0,
   'firenze by night ristorante': 0.0,
   'fleur de lys': 0.0,
   'fog harbor fish house': 0.0,
   'forbes island': 0.0,
   'foreign cinema': 0.0,
   'frances': 0.0,
   'franchino': 0.0,
   'franciscan crab restaurant': 0.0,
   'frascati': 0.0,
   'fresca': 0.0,
   'fringale': 0.0,
   'fujiyama ya japanese restaurant': 0.0,
   'gajalee': 0.0,
   'gamine': 0.0,
   'garcon restaurant': 0.0,
   'gary danko': 0.0,
   'gitane': 0.0,
   'golden era restaurant': 0.0,
   'gracias madre': 0.0,
   'great eastern restaurant': 1.0,
   'hakka restaurant': 0.0,
   'hakkasan': 0.0,
   'han second kwan': 0.0,
   'heirloom cafe': 0.0,
   'helmand palace': 0.0,
   'hi dive': 0.0,
   'hillside supper club': 0.0,
   'hillstone': 0.0,
   'hong kong clay pot restaurant': 0.0,
   'house of nanking': 0.0,
   'house of prime rib': 0.0,
   'hunan homes restaurant': 0.0,
   'incanto': 0.0,
   'isa': 0.0,
   'jannah': 0.0,
   'jasmine garden': 0.0,
   'jitlada thai cuisine': 0.0,
   'kappa japanese restaurant': 0.0,
   'kim thanh restaurant': 0.0,
   'kirin chinese restaurant': 0.0,
   'kiss seafood': 0.0,
   'kokkari estiatorio': 0.0,
   'la briciola': 0.0,
   'la ciccia': 0.0,
   'la folie': 0.0,
   'la mediterranee': 0.0,
   'la traviata': 0.0,
   'lahore karahi': 0.0,
   'lavash': 0.0,
   'le charm': 0.0,
   'le colonial': 0.0,
   'le soleil': 0.0,
   'lime tree southeast asian kitchen': 0.0,
   'little delhi': 0.0,
   'little nepal': 0.0,
   'luce': 0.0,
   'lucky creation restaurant': 0.0,
   'luella': 0.0,
   'lupa': 0.0,
   'm y china': 0.0,
   'maki restaurant': 0.0,
   'mangia tutti ristorante': 0.0,
   'manna': 0.0,
   'marlowe': 0.0,
   'marnee thai': 0.0,
   'maverick': 0.0,
   'mela tandoori kitchen': 0.0,
   'mescolanza': 0.0,
   'mezes': 0.0,
   'michael mina restaurant': 0.0,
   'millennium': 0.0,
   'minako organic japanese restaurant': 0.0,
   'minami restaurant': 0.0,
   'mission chinese food': 0.0,
   'mochica': 0.0,
   'modern thai': 0.0,
   'mona lisa restaurant': 0.0,
   'mozzeria': 0.0,
   'muguboka restaurant': 0.0,
   'my tofu house': 0.0,
   'nicaragua restaurant': 0.0,
   'nob hill cafe': 0.0,
   'nopa': 0.0,
   'old jerusalem restaurant': 0.0,
   'old skool cafe': 0.0,
   'one market restaurant': 0.0,
   'orexi': 0.0,
   'original us restaurant': 0.0,
   'osha thai': 0.0,
   'oyaji restaurant': 0.0,
   'ozumo': 0.0,
   'pad thai restaurant': 0.0,
   'panta rei restaurant': 0.0,
   'park tavern': 0.0,
   'pera': 0.0,
   'piperade': 0.0,
   'ploy 2': 0.0,
   'poc chuc': 0.0,
   'poesia': 0.0,
   'prospect': 0.0,
   'quince': 0.0,
   'radius san francisco': 0.0,
   'range': 0.0,
   'red door cafe': 0.0,
   'restaurant ducroix': 0.0,
   'ristorante bacco': 0.0,
   'ristorante ideale': 0.0,
   'ristorante milano': 0.0,
   'ristorante parma': 0.0,
   'rn74': 0.0,
   'rue lepic': 0.0,
   'saha': 0.0,
   'sai jai thai restaurant': 0.0,
   'salt house': 0.0,
   'san tung chinese restaurant': 0.0,
   'san wang restaurant': 0.0,
   'sanjalisco': 0.0,
   'sanraku': 0.0,
   'seasons': 0.0,
   'seoul garden': 0.0,
   'seven hills': 0.0,
   'shangri la vegetarian restaurant': 0.0,
   'singapore malaysian restaurant': 0.0,
   'skool': 0.0,
   'so': 0.0,
   'sotto mare': 0.0,
   'source': 0.0,
   'specchio ristorante': 0.0,
   'spruce': 0.0,
   'straits restaurant': 0.0,
   'stroganoff restaurant': 0.0,
   'sunflower potrero hill': 0.0,
   'sushi bistro': 0.0,
   'taiwan restaurant': 0.0,
   'tanuki restaurant': 0.0,
   'tataki': 0.0,
   'tekka japanese restaurant': 0.0,
   'thai cottage restaurant': 0.0,
   'thai house express': 0.0,
   'thai idea vegetarian': 0.0,
   'thai time restaurant': 0.0,
   'thanh long': 0.0,
   'the big 4 restaurant': 0.0,
   'the blue plate': 0.0,
   'the house': 0.0,
   'the richmond': 0.0,
   'the slanted door': 0.0,
   'the stinking rose': 0.0,
   'thep phanom thai restaurant': 0.0,
   'tommys joynt': 0.0,
   'toraya japanese restaurant': 0.0,
   'town hall': 0.0,
   'trattoria contadina': 0.0,
   'tu lan': 0.0,
   'tuba restaurant': 0.0,
   'u lee restaurant': 0.0,
   'udupi palace': 0.0,
   'venticello ristorante': 0.0,
   'vicoletto': 0.0,
   'yank sing': 0.0,
   'yummy yummy': 0.0,
   'z and y restaurant': 0.0,
   'zadin': 0.0,
   'zare at fly trap': 0.0,
   'zarzuela': 0.0,
   'zen yai thai restaurant': 0.0,
   'zuni cafe': 0.0,
   'zushi puzzle': 0.0},
  'near': {'**NONE**': 0.0,
   'bayview hunters point': 0.0,
   'dontcare': 1.0,
   'haight': 0.0,
   'japantown': 0.0,
   'marina cow hollow': 0.0,
   'mission': 0.0,
   'nopa': 0.0,
   'north beach telegraph hill': 0.0,
   'soma': 0.0,
   'union square': 0.0},
  'price': {'**NONE**': 1.0,
   '10 dollar': 0.0,
   '10 euro': 0.0,
   '11 euro': 0.0,
   '15 euro': 0.0,
   '18 euro': 0.0,
   '20 euro': 0.0,
   '22 euro': 0.0,
   '25 euro': 0.0,
   '26 euro': 0.0,
   '29 euro': 0.0,
   '37 euro': 0.0,
   '6': 0.0,
   '7': 0.0,
   '9': 0.0,
   'between 0 and 15 euro': 0.0,
   'between 10 and 13 euro': 0.0,
   'between 10 and 15 euro': 0.0,
   'between 10 and 18 euro': 0.0,
   'between 10 and 20 euro': 0.0,
   'between 10 and 23 euro': 0.0,
   'between 10 and 30 euro': 0.0,
   'between 11 and 15 euro': 0.0,
   'between 11 and 18 euro': 0.0,
   'between 11 and 22 euro': 0.0,
   'between 11 and 25 euro': 0.0,
   'between 11 and 29 euro': 0.0,
   'between 11 and 35 euro': 0.0,
   'between 13 and 15 euro': 0.0,
   'between 13 and 18 euro': 0.0,
   'between 13 and 24 euro': 0.0,
   'between 15 and 18 euro': 0.0,
   'between 15 and 22 euro': 0.0,
   'between 15 and 26 euro': 0.0,
   'between 15 and 29 euro': 0.0,
   'between 15 and 33 euro': 0.0,
   'between 15 and 44 euro': 0.0,
   'between 15 and 58 euro': 0.0,
   'between 18 and 26 euro': 0.0,
   'between 18 and 29 euro': 0.0,
   'between 18 and 44 euro': 0.0,
   'between 18 and 55 euro': 0.0,
   'between 18 and 58 euro': 0.0,
   'between 18 and 73 euro': 0.0,
   'between 18 and 78 euro': 0.0,
   'between 2 and 15 euro': 0.0,
   'between 20 and 30 euro': 0.0,
   'between 21 and 23 euro': 0.0,
   'between 22 and 29 euro': 0.0,
   'between 22 and 30 dollar': 0.0,
   'between 22 and 37 euro': 0.0,
   'between 22 and 58 euro': 0.0,
   'between 22 and 73 euro': 0.0,
   'between 23 and 29': 0.0,
   'between 23 and 29 euro': 0.0,
   'between 23 and 37 euro': 0.0,
   'between 23 and 58': 0.0,
   'between 23 and 58 euro': 0.0,
   'between 26 and 33 euro': 0.0,
   'between 26 and 34 euro': 0.0,
   'between 26 and 37 euro': 0.0,
   'between 29 and 37 euro': 0.0,
   'between 29 and 44 euro': 0.0,
   'between 29 and 58 euro': 0.0,
   'between 29 and 73 euro': 0.0,
   'between 30 and 58': 0.0,
   'between 30 and 58 euro': 0.0,
   'between 31 and 50 euro': 0.0,
   'between 37 and 110 euro': 0.0,
   'between 37 and 44 euro': 0.0,
   'between 37 and 58 euro': 0.0,
   'between 4 and 22 euro': 0.0,
   'between 4 and 58 euro': 0.0,
   'between 5 an 30 euro': 0.0,
   'between 5 and 10 euro': 0.0,
   'between 5 and 11 euro': 0.0,
   'between 5 and 15 dollar': 0.0,
   'between 5 and 20 euro': 0.0,
   'between 5 and 25 euro': 0.0,
   'between 6 and 10 euro': 0.0,
   'between 6 and 11 euro': 0.0,
   'between 6 and 15 euro': 0.0,
   'between 6 and 29 euro': 0.0,
   'between 7 and 11 euro': 0.0,
   'between 7 and 13 euro': 0.0,
   'between 7 and 15 euro': 0.0,
   'between 7 and 37 euro': 0.0,
   'between 8 and 22 euro': 0.0,
   'between 9 and 13 dolllar': 0.0,
   'between 9 and 15 euro': 0.0,
   'between 9 and 58 euro': 0.0,
   'bteween 11 and 15 euro': 0.0,
   'bteween 15 and 22 euro': 0.0,
   'bteween 22 and 37': 0.0,
   'bteween 30 and 58 euro': 0.0,
   'bteween 51 and 73 euro': 0.0,
   'netween 20 and 30 euro': 0.0},
  'pricerange': {'**NONE**': 1.0,
   'cheap': 0.0,
   'dontcare': 0.0,
   'expensive': 0.0,
   'moderate': 0.0},
  'requested': {'addr': 1.0,
   'allowedforkids': 0.0,
   'area': 0.0,
   'food': 0.0,
   'goodformeal': 0.0,
   'name': 0.0,
   'near': 0.0,
   'phone': 1,
   'postcode': 0.0,
   'price': 0.0,
   'pricerange': 0.0}},
 'features': {'inform_info': [False,
   False,
   True,
   False,
   True,
   False,
   False,
   True,
   False,
   True,
   False,
   False,
   True,
   False,
   True,
   False,
   False,
   True,
   False,
   True,
   False,
   False,
   True,
   False,
   True],
  'informedVenueSinceNone': ['great eastern restaurant',
   'great eastern restaurant'],
  'lastActionInformNone': False,
  'lastInformedVenue': 'great eastern restaurant',
  'offerHappened': False},
 'userActs': [('request(name="great eastern restaurant",phone)', 1.0)]}
    b2 = {'beliefs': {'allowedforkids': {'**NONE**': 0.014367834316388661,
   '0': 0.009175995595522114,
   '1': 0.9579333306577846,
   'dontcare': 0.01852283943030468},
  'area': {'**NONE**': 0.9753165718480455,
   'alamo square': 0.0,
   'amanico ergina village': 0.0,
   'anza vista': 0.0,
   'ashbury heights': 0.0,
   'balboa terrace': 0.0,
   'bayview district': 0.0,
   'bayview heights': 0.0,
   'bernal heights': 0.0,
   'bernal heights north': 0.0,
   'bernal heights south': 0.0,
   'buena vista park': 0.0,
   'castro': 0.0,
   'cathedral hill': 0.0,
   'cayuga terrace': 0.0,
   'central richmond': 0.0,
   'central sunset': 0.0,
   'central waterfront': 0.0,
   'chinatown': 0.0,
   'civic center': 0.0,
   'clarendon heights': 0.0,
   'cole valley': 0.0,
   'corona heights': 0.0,
   'cow hollow': 0.0,
   'crocker amazon': 0.0,
   'diamond heights': 0.0,
   'doelger city': 0.0,
   'dogpatch': 0.0,
   'dolores heights': 0.0,
   'dontcare': 0.0,
   'downtown': 0.0,
   'duboce triangle': 0.0,
   'embarcadero': 0.0,
   'eureka valley': 0.0,
   'eureka valley dolores heights': 0.0,
   'excelsior': 0.0,
   'financial district': 0.0,
   'financial district south': 0.0,
   'fishermans wharf': 0.0,
   'forest hill': 0.0,
   'forest hill extension': 0.0,
   'forest knolls': 0.0,
   'fort mason': 0.0,
   'fort winfield scott': 0.0,
   'frederick douglass haynes gardens': 0.0,
   'friendship village': 0.0,
   'glen park': 0.0,
   'glenridge': 0.0,
   'golden gate heights': 0.0,
   'golden gate park': 0.0,
   'haight ashbury': 0.0,
   'hayes valley': 0.0,
   'hunters point': 0.0,
   'india basin': 0.0,
   'ingleside': 0.0,
   'ingleside heights': 0.0,
   'ingleside terrace': 0.0,
   'inner mission': 0.0,
   'inner parkside': 0.0,
   'inner richmond': 0.0,
   'inner sunset': 0.0,
   'inset': 0.0,
   'jordan park': 0.0,
   'laguna honda': 0.0,
   'lake': 0.0,
   'lake shore': 0.0,
   'lakeside': 0.0,
   'laurel heights': 0.0,
   'lincoln park': 0.0,
   'lincoln park lobos': 0.0,
   'little hollywood': 0.0,
   'little italy': 0.0,
   'little osaka': 0.0,
   'little russia': 0.0,
   'lone mountain': 0.0,
   'lower haight': 0.0,
   'lower nob hill': 0.0,
   'lower pacific heights': 0.0,
   'malcolm x square': 0.0,
   'marcus garvey square': 0.0,
   'marina district': 0.0,
   'martin luther king square': 0.0,
   'mastro': 0.0,
   'merced heights': 0.0,
   'merced manor': 0.0,
   'midtown terrace': 0.0,
   'miraloma park': 0.0,
   'mission bay': 0.0,
   'mission district': 0.0,
   'mission dolores': 0.0,
   'mission terrace': 0.0,
   'monterey heights': 0.0,
   'mount davidson manor': 0.0,
   'nob hill': 0.0,
   'noe valley': 0.0,
   'noma': 0.0,
   'north beach': 0.0,
   'north panhandle': 0.0,
   'north park': 0.0,
   'north waterfront': 0.0,
   'oceanview': 0.0,
   'opera plaza': 0.0,
   'outer mission': 0.0,
   'outer parkside': 0.0,
   'outer richmond': 0.0,
   'outer sunset': 0.0,
   'outset': 0.0,
   'pacific heights': 0.0,
   'panhandle': 0.0,
   'park merced': 0.0,
   'parkmerced': 0.0,
   'parkside': 0.0,
   'pine lake park': 0.0,
   'portola': 0.0,
   'potrero flats': 0.0,
   'potrero hill': 0.0,
   'presidio': 0.0,
   'presidio heights': 0.0,
   'richmond district': 0.0,
   'russian hill': 0.0,
   'saint francis wood': 0.0,
   'san francisco airport': 0.0,
   'san francisco state university': 0.0,
   'sea cliff': 0.0,
   'sherwood forest': 0.0,
   'showplace square': 0.0,
   'silver terrace': 0.0,
   'somisspo': 0.0,
   'south basin': 0.0,
   'south beach': 0.0,
   'south of market': 0.0,
   'st francis square': 0.0,
   'st francis wood': 0.0,
   'stonestown': 0.024683428151954484,
   'sunnydale': 0.0,
   'sunnyside': 0.0,
   'sunset district': 0.0,
   'telegraph hill': 0.0,
   'tenderloin': 0.0,
   'thomas paine square': 0.0,
   'transmission': 0.0,
   'treasure island': 0.0,
   'twin peaks': 0.0,
   'twin peaks west': 0.0,
   'upper market': 0.0,
   'van ness': 0.0,
   'victoria mews': 0.0,
   'visitacion valley': 0.0,
   'vista del monte': 0.0,
   'west of twin peaks': 0.0,
   'west portal': 0.0,
   'western addition': 0.0,
   'westlake and olympic': 0.0,
   'westwood highlands': 0.0,
   'westwood park': 0.0,
   'yerba buena island': 0.0,
   'zion district': 0.0},
  'discourseAct': {'ack': 0.0,
   'bye': 0.0,
   'hello': 0.0,
   'none': 0.9999999999999998,
   'repeat': 0.0,
   'silence': 0.0,
   'thankyou': 0.0},
  'food': {'**NONE**': 1.0,
   'afghan': 0.0,
   'arabian': 0.0,
   'asian': 0.0,
   'basque': 0.0,
   'brasseries': 0.0,
   'brazilian': 0.0,
   'buffets': 0.0,
   'burgers': 0.0,
   'burmese': 0.0,
   'cafes': 0.0,
   'cambodian': 0.0,
   'cantonese': 0.0,
   'chinese': 0.0,
   'comfort food': 0.0,
   'creperies': 0.0,
   'dim sum': 0.0,
   'dontcare': 0.0,
   'ethiopian': 0.0,
   'ethnic food': 0.0,
   'french': 0.0,
   'gluten free': 0.0,
   'himalayan': 0.0,
   'indian': 0.0,
   'indonesian': 0.0,
   'indpak': 0.0,
   'italian': 0.0,
   'japanese': 0.0,
   'korean': 0.0,
   'kosher': 0.0,
   'latin': 0.0,
   'lebanese': 0.0,
   'lounges': 0.0,
   'malaysian': 0.0,
   'mediterranean': 0.0,
   'mexican': 0.0,
   'middle eastern': 0.0,
   'modern european': 0.0,
   'moroccan': 0.0,
   'new american': 0.0,
   'pakistani': 0.0,
   'persian': 0.0,
   'peruvian': 0.0,
   'pizza': 0.0,
   'raw food': 0.0,
   'russian': 0.0,
   'sandwiches': 0.0,
   'sea food': 0.0,
   'shanghainese': 0.0,
   'singaporean': 0.0,
   'soul food': 0.0,
   'spanish': 0.0,
   'steak': 0.0,
   'sushi': 0.0,
   'taiwanese': 0.0,
   'tapas': 0.0,
   'thai': 0.0,
   'traditionnal american': 0.0,
   'turkish': 0.0,
   'vegetarian': 0.0,
   'vietnamese': 0.0},
  'goodformeal': {'**NONE**': 1.0,
   'breakfast': 0.0,
   'brunch': 0.0,
   'dinner': 0.0,
   'dontcare': 0.0,
   'lunch': 0.0},
  'method': {'byalternatives': 0.0,
   'byconstraints': 0.7725475751076113,
   'byname': 0.0,
   'finished': 0.0,
   'none': 0.0,
   'restart': 0.0},
  'name': {'**NONE**': 1.0,
   'a 16': 0.0,
   'a la turca restaurant': 0.0,
   'abacus': 0.0,
   'alamo square seafood grill': 0.0,
   'albona ristorante istriano': 0.0,
   'alborz persian cuisine': 0.0,
   'allegro romano': 0.0,
   'amarena': 0.0,
   'amber india': 0.0,
   'ame': 0.0,
   'ananda fuara': 0.0,
   'anchor oyster bar': 0.0,
   'angkor borei restaurant': 0.0,
   'aperto restaurant': 0.0,
   'ar roi restaurant': 0.0,
   'arabian nights restaurant': 0.0,
   'assab eritrean restaurant': 0.0,
   'atelier crenn': 0.0,
   'aux delices restaurant': 0.0,
   'aziza': 0.0,
   'b star bar': 0.0,
   'bar crudo': 0.0,
   'beijing restaurant': 0.0,
   'bella trattoria': 0.0,
   'benu': 0.0,
   'betelnut': 0.0,
   'bistro central parc': 0.0,
   'bix': 0.0,
   'borgo': 0.0,
   'borobudur restaurant': 0.0,
   'bouche': 0.0,
   'boulevard': 0.0,
   'brothers restaurant': 0.0,
   'bund shanghai restaurant': 0.0,
   'burma superstar': 0.0,
   'butterfly': 0.0,
   'cafe claude': 0.0,
   'cafe jacqueline': 0.0,
   'campton place restaurant': 0.0,
   'canteen': 0.0,
   'canto do brasil restaurant': 0.0,
   'capannina': 0.0,
   'capital restaurant': 0.0,
   'chai yo thai restaurant': 0.0,
   'chaya brasserie': 0.0,
   'chenery park': 0.0,
   'chez maman': 0.0,
   'chez papa bistrot': 0.0,
   'chez spencer': 0.0,
   'chiaroscuro': 0.0,
   'chouchou': 0.0,
   'chow': 0.0,
   'city view restaurant': 0.0,
   'claudine': 0.0,
   'coi': 0.0,
   'colibri mexican bistro': 0.0,
   'coqueta': 0.0,
   'crustacean restaurant': 0.0,
   'da flora a venetian osteria': 0.0,
   'darbar restaurant': 0.0,
   'delancey street restaurant': 0.0,
   'delfina': 0.0,
   'dong baek restaurant': 0.0,
   'dosa on fillmore': 0.0,
   'dosa on valencia': 0.0,
   'eiji': 0.0,
   'enjoy vegetarian restaurant': 0.0,
   'espetus churrascaria': 0.0,
   'fang': 0.0,
   'farallon': 0.0,
   'fattoush restaurant': 0.0,
   'fifth floor': 0.0,
   'fino restaurant': 0.0,
   'firefly': 0.0,
   'firenze by night ristorante': 0.0,
   'fleur de lys': 0.0,
   'fog harbor fish house': 0.0,
   'forbes island': 0.0,
   'foreign cinema': 0.0,
   'frances': 0.0,
   'franchino': 0.0,
   'franciscan crab restaurant': 0.0,
   'frascati': 0.0,
   'fresca': 0.0,
   'fringale': 0.0,
   'fujiyama ya japanese restaurant': 0.0,
   'gajalee': 0.0,
   'gamine': 0.0,
   'garcon restaurant': 0.0,
   'gary danko': 0.0,
   'gitane': 0.0,
   'golden era restaurant': 0.0,
   'gracias madre': 0.0,
   'great eastern restaurant': 0.0,
   'hakka restaurant': 0.0,
   'hakkasan': 0.0,
   'han second kwan': 0.0,
   'heirloom cafe': 0.0,
   'helmand palace': 0.0,
   'hi dive': 0.0,
   'hillside supper club': 0.0,
   'hillstone': 0.0,
   'hong kong clay pot restaurant': 0.0,
   'house of nanking': 0.0,
   'house of prime rib': 0.0,
   'hunan homes restaurant': 0.0,
   'incanto': 0.0,
   'isa': 0.0,
   'jannah': 0.0,
   'jasmine garden': 0.0,
   'jitlada thai cuisine': 0.0,
   'kappa japanese restaurant': 0.0,
   'kim thanh restaurant': 0.0,
   'kirin chinese restaurant': 0.0,
   'kiss seafood': 0.0,
   'kokkari estiatorio': 0.0,
   'la briciola': 0.0,
   'la ciccia': 0.0,
   'la folie': 0.0,
   'la mediterranee': 0.0,
   'la traviata': 0.0,
   'lahore karahi': 0.0,
   'lavash': 0.0,
   'le charm': 0.0,
   'le colonial': 0.0,
   'le soleil': 0.0,
   'lime tree southeast asian kitchen': 0.0,
   'little delhi': 0.0,
   'little nepal': 0.0,
   'luce': 0.0,
   'lucky creation restaurant': 0.0,
   'luella': 0.0,
   'lupa': 0.0,
   'm y china': 0.0,
   'maki restaurant': 0.0,
   'mangia tutti ristorante': 0.0,
   'manna': 0.0,
   'marlowe': 0.0,
   'marnee thai': 0.0,
   'maverick': 0.0,
   'mela tandoori kitchen': 0.0,
   'mescolanza': 0.0,
   'mezes': 0.0,
   'michael mina restaurant': 0.0,
   'millennium': 0.0,
   'minako organic japanese restaurant': 0.0,
   'minami restaurant': 0.0,
   'mission chinese food': 0.0,
   'mochica': 0.0,
   'modern thai': 0.0,
   'mona lisa restaurant': 0.0,
   'mozzeria': 0.0,
   'muguboka restaurant': 0.0,
   'my tofu house': 0.0,
   'nicaragua restaurant': 0.0,
   'nob hill cafe': 0.0,
   'nopa': 0.0,
   'old jerusalem restaurant': 0.0,
   'old skool cafe': 0.0,
   'one market restaurant': 0.0,
   'orexi': 0.0,
   'original us restaurant': 0.0,
   'osha thai': 0.0,
   'oyaji restaurant': 0.0,
   'ozumo': 0.0,
   'pad thai restaurant': 0.0,
   'panta rei restaurant': 0.0,
   'park tavern': 0.0,
   'pera': 0.0,
   'piperade': 0.0,
   'ploy 2': 0.0,
   'poc chuc': 0.0,
   'poesia': 0.0,
   'prospect': 0.0,
   'quince': 0.0,
   'radius san francisco': 0.0,
   'range': 0.0,
   'red door cafe': 0.0,
   'restaurant ducroix': 0.0,
   'ristorante bacco': 0.0,
   'ristorante ideale': 0.0,
   'ristorante milano': 0.0,
   'ristorante parma': 0.0,
   'rn74': 0.0,
   'rue lepic': 0.0,
   'saha': 0.0,
   'sai jai thai restaurant': 0.0,
   'salt house': 0.0,
   'san tung chinese restaurant': 0.0,
   'san wang restaurant': 0.0,
   'sanjalisco': 0.0,
   'sanraku': 0.0,
   'seasons': 0.0,
   'seoul garden': 0.0,
   'seven hills': 0.0,
   'shangri la vegetarian restaurant': 0.0,
   'singapore malaysian restaurant': 0.0,
   'skool': 0.0,
   'so': 0.0,
   'sotto mare': 0.0,
   'source': 0.0,
   'specchio ristorante': 0.0,
   'spruce': 0.0,
   'straits restaurant': 0.0,
   'stroganoff restaurant': 0.0,
   'sunflower potrero hill': 0.0,
   'sushi bistro': 0.0,
   'taiwan restaurant': 0.0,
   'tanuki restaurant': 0.0,
   'tataki': 0.0,
   'tekka japanese restaurant': 0.0,
   'thai cottage restaurant': 0.0,
   'thai house express': 0.0,
   'thai idea vegetarian': 0.0,
   'thai time restaurant': 0.0,
   'thanh long': 0.0,
   'the big 4 restaurant': 0.0,
   'the blue plate': 0.0,
   'the house': 0.0,
   'the richmond': 0.0,
   'the slanted door': 0.0,
   'the stinking rose': 0.0,
   'thep phanom thai restaurant': 0.0,
   'tommys joynt': 0.0,
   'toraya japanese restaurant': 0.0,
   'town hall': 0.0,
   'trattoria contadina': 0.0,
   'tu lan': 0.0,
   'tuba restaurant': 0.0,
   'u lee restaurant': 0.0,
   'udupi palace': 0.0,
   'venticello ristorante': 0.0,
   'vicoletto': 0.0,
   'yank sing': 0.0,
   'yummy yummy': 0.0,
   'z and y restaurant': 0.0,
   'zadin': 0.0,
   'zare at fly trap': 0.0,
   'zarzuela': 0.0,
   'zen yai thai restaurant': 0.0,
   'zuni cafe': 0.0,
   'zushi puzzle': 0.0},
  'near': {'**NONE**': 0.13300733496332517,
   'bayview hunters point': 0.0,
   'dontcare': 0.15859820700896493,
   'haight': 0.0,
   'japantown': 0.038712306438467806,
   'marina cow hollow': 0.0,
   'mission': 0.0,
   'nopa': 0.669682151589242,
   'north beach telegraph hill': 0.0,
   'soma': 0.0,
   'union square': 0.0},
  'price': {'**NONE**': 1.0,
   '10 dollar': 0.0,
   '10 euro': 0.0,
   '11 euro': 0.0,
   '15 euro': 0.0,
   '18 euro': 0.0,
   '20 euro': 0.0,
   '22 euro': 0.0,
   '25 euro': 0.0,
   '26 euro': 0.0,
   '29 euro': 0.0,
   '37 euro': 0.0,
   '6': 0.0,
   '7': 0.0,
   '9': 0.0,
   'between 0 and 15 euro': 0.0,
   'between 10 and 13 euro': 0.0,
   'between 10 and 15 euro': 0.0,
   'between 10 and 18 euro': 0.0,
   'between 10 and 20 euro': 0.0,
   'between 10 and 23 euro': 0.0,
   'between 10 and 30 euro': 0.0,
   'between 11 and 15 euro': 0.0,
   'between 11 and 18 euro': 0.0,
   'between 11 and 22 euro': 0.0,
   'between 11 and 25 euro': 0.0,
   'between 11 and 29 euro': 0.0,
   'between 11 and 35 euro': 0.0,
   'between 13 and 15 euro': 0.0,
   'between 13 and 18 euro': 0.0,
   'between 13 and 24 euro': 0.0,
   'between 15 and 18 euro': 0.0,
   'between 15 and 22 euro': 0.0,
   'between 15 and 26 euro': 0.0,
   'between 15 and 29 euro': 0.0,
   'between 15 and 33 euro': 0.0,
   'between 15 and 44 euro': 0.0,
   'between 15 and 58 euro': 0.0,
   'between 18 and 26 euro': 0.0,
   'between 18 and 29 euro': 0.0,
   'between 18 and 44 euro': 0.0,
   'between 18 and 55 euro': 0.0,
   'between 18 and 58 euro': 0.0,
   'between 18 and 73 euro': 0.0,
   'between 18 and 78 euro': 0.0,
   'between 2 and 15 euro': 0.0,
   'between 20 and 30 euro': 0.0,
   'between 21 and 23 euro': 0.0,
   'between 22 and 29 euro': 0.0,
   'between 22 and 30 dollar': 0.0,
   'between 22 and 37 euro': 0.0,
   'between 22 and 58 euro': 0.0,
   'between 22 and 73 euro': 0.0,
   'between 23 and 29': 0.0,
   'between 23 and 29 euro': 0.0,
   'between 23 and 37 euro': 0.0,
   'between 23 and 58': 0.0,
   'between 23 and 58 euro': 0.0,
   'between 26 and 33 euro': 0.0,
   'between 26 and 34 euro': 0.0,
   'between 26 and 37 euro': 0.0,
   'between 29 and 37 euro': 0.0,
   'between 29 and 44 euro': 0.0,
   'between 29 and 58 euro': 0.0,
   'between 29 and 73 euro': 0.0,
   'between 30 and 58': 0.0,
   'between 30 and 58 euro': 0.0,
   'between 31 and 50 euro': 0.0,
   'between 37 and 110 euro': 0.0,
   'between 37 and 44 euro': 0.0,
   'between 37 and 58 euro': 0.0,
   'between 4 and 22 euro': 0.0,
   'between 4 and 58 euro': 0.0,
   'between 5 an 30 euro': 0.0,
   'between 5 and 10 euro': 0.0,
   'between 5 and 11 euro': 0.0,
   'between 5 and 15 dollar': 0.0,
   'between 5 and 20 euro': 0.0,
   'between 5 and 25 euro': 0.0,
   'between 6 and 10 euro': 0.0,
   'between 6 and 11 euro': 0.0,
   'between 6 and 15 euro': 0.0,
   'between 6 and 29 euro': 0.0,
   'between 7 and 11 euro': 0.0,
   'between 7 and 13 euro': 0.0,
   'between 7 and 15 euro': 0.0,
   'between 7 and 37 euro': 0.0,
   'between 8 and 22 euro': 0.0,
   'between 9 and 13 dolllar': 0.0,
   'between 9 and 15 euro': 0.0,
   'between 9 and 58 euro': 0.0,
   'bteween 11 and 15 euro': 0.0,
   'bteween 15 and 22 euro': 0.0,
   'bteween 22 and 37': 0.0,
   'bteween 30 and 58 euro': 0.0,
   'bteween 51 and 73 euro': 0.0,
   'netween 20 and 30 euro': 0.0},
  'pricerange': {'**NONE**': 0.22571148184494605,
   'cheap': 0.0,
   'dontcare': 0.774288518155054,
   'expensive': 0.0,
   'moderate': 0.0},
  'requested': {'addr': 0.0,
   'allowedforkids': 0.0,
   'area': 0.0,
   'food': 0.0,
   'goodformeal': 0.0,
   'name': 0.0,
   'near': 0.0,
   'phone': 0.0,
   'postcode': 0.0,
   'price': 0.0,
   'pricerange': 0.0}},
 'features': {'inform_info': [False,
   False,
   False,
   True,
   True,
   False,
   False,
   False,
   True,
   True,
   False,
   True,
   False,
   False,
   False,
   False,
   True,
   False,
   False,
   False,
   False,
   True,
   False,
   False,
   False],
  'informedVenueSinceNone': [],
  'lastActionInformNone': False,
  'lastInformedVenue': '',
  'offerHappened': False},
 'userActs': [('inform(allowedforkids="1")', 0.90842356395668944),
  ('inform(allowedforkids="dontcare")', 0.0091759955955221153),
  ('inform(allowedforkids="0")', 0.0091759955955221153),
  ('inform(postcode)', 0.025509267755551478),
  ('inform(area="stonestown")', 0.024683428151954491),
  ('null()', 0.023031748944760511)]}

    b3 = {'beliefs': {'area': {'**NONE**': 0.12910550615265692,
   'centre': 0.8338099777773861,
   'dontcare': 0.0,
   'east': 0.03708451606995696,
   'north': 0.0,
   'south': 0.0,
   'west': 0.0},
  'discourseAct': {'ack': 0.0,
   'bye': 0.0,
   'hello': 0.0,
   'none': 1.0,
   'repeat': 0.0,
   'silence': 0.0,
   'thankyou': 0.0},
  'food': {'**NONE**': 0.020895546925810415,
   'afghan': 0.0,
   'african': 0.0,
   'afternoon tea': 0.0,
   'asian oriental': 0.0,
   'australasian': 0.0,
   'australian': 0.0,
   'austrian': 0.0,
   'barbeque': 0.0,
   'basque': 0.0,
   'belgian': 0.0,
   'bistro': 0.0,
   'brazilian': 0.0,
   'british': 0.0,
   'canapes': 0.0,
   'cantonese': 0.0,
   'caribbean': 0.0,
   'catalan': 0.0,
   'chinese': 0.0,
   'christmas': 0.0,
   'corsica': 0.0,
   'creative': 0.0,
   'crossover': 0.0,
   'cuban': 0.0,
   'danish': 0.0,
   'dontcare': 0.0,
   'eastern european': 0.0,
   'english': 0.0,
   'eritrean': 0.0,
   'european': 0.0,
   'french': 0.0,
   'fusion': 0.0,
   'gastropub': 0.0,
   'german': 0.0,
   'greek': 0.0,
   'halal': 0.0,
   'hungarian': 0.0,
   'indian': 0.0,
   'indonesian': 0.0,
   'international': 0.0,
   'irish': 0.0,
   'italian': 0.0,
   'jamaican': 0.0,
   'japanese': 0.0,
   'korean': 0.0,
   'kosher': 0.0,
   'latin american': 0.0,
   'lebanese': 0.0,
   'light bites': 0.0,
   'malaysian': 0.0,
   'mediterranean': 0.9791044530741896,
   'mexican': 0.0,
   'middle eastern': 0.0,
   'modern american': 0.0,
   'modern eclectic': 0.0,
   'modern european': 0.0,
   'modern global': 0.0,
   'molecular gastronomy': 0.0,
   'moroccan': 0.0,
   'new zealand': 0.0,
   'north african': 0.0,
   'north american': 0.0,
   'north indian': 0.0,
   'northern european': 0.0,
   'panasian': 0.0,
   'persian': 0.0,
   'polish': 0.0,
   'polynesian': 0.0,
   'portuguese': 0.0,
   'romanian': 0.0,
   'russian': 0.0,
   'scandinavian': 0.0,
   'scottish': 0.0,
   'seafood': 0.0,
   'singaporean': 0.0,
   'south african': 0.0,
   'south indian': 0.0,
   'spanish': 0.0,
   'sri lankan': 0.0,
   'steakhouse': 0.0,
   'swedish': 0.0,
   'swiss': 0.0,
   'thai': 0.0,
   'the americas': 0.0,
   'traditional': 0.0,
   'turkish': 0.0,
   'tuscan': 0.0,
   'unusual': 0.0,
   'vegetarian': 0.0,
   'venetian': 0.0,
   'vietnamese': 0.0,
   'welsh': 0.0,
   'world': 0.0},
  'method': {'byalternatives': 0.0,
   'byconstraints': 0.6359877465366015,
   'byname': 0.0,
   'finished': 0.0,
   'none': 0.0,
   'restart': 0.0},
  'name': {'**NONE**': 1.0,
   'ali baba': 0.0,
   'anatolia': 0.0,
   'ask': 0.0,
   'backstreet bistro': 0.0,
   'bangkok city': 0.0,
   'bedouin': 0.0,
   'bloomsbury restaurant': 0.0,
   'caffe uno': 0.0,
   'cambridge lodge restaurant': 0.0,
   'charlie chan': 0.0,
   'chiquito restaurant bar': 0.0,
   'city stop restaurant': 0.0,
   'clowns cafe': 0.0,
   'cocum': 0.0,
   'cote': 0.0,
   'cotto': 0.0,
   'curry garden': 0.0,
   'curry king': 0.0,
   'curry prince': 0.0,
   'curry queen': 0.0,
   'da vince pizzeria': 0.0,
   'da vinci pizzeria': 0.0,
   'darrys cookhouse and wine shop': 0.0,
   'de luca cucina and bar': 0.0,
   'dojo noodle bar': 0.0,
   'don pasquale pizzeria': 0.0,
   'efes restaurant': 0.0,
   'eraina': 0.0,
   'fitzbillies restaurant': 0.0,
   'frankie and bennys': 0.0,
   'galleria': 0.0,
   'golden house': 0.0,
   'golden wok': 0.0,
   'gourmet burger kitchen': 0.0,
   'graffiti': 0.0,
   'grafton hotel restaurant': 0.0,
   'hakka': 0.0,
   'hk fusion': 0.0,
   'hotel du vin and bistro': 0.0,
   'india house': 0.0,
   'j restaurant': 0.0,
   'jinling noodle bar': 0.0,
   'kohinoor': 0.0,
   'kymmoy': 0.0,
   'la margherita': 0.0,
   'la mimosa': 0.0,
   'la raza': 0.0,
   'la tasca': 0.0,
   'lan hong house': 0.0,
   'little seoul': 0.0,
   'loch fyne': 0.0,
   'mahal of cambridge': 0.0,
   'maharajah tandoori restaurant': 0.0,
   'meghna': 0.0,
   'meze bar restaurant': 0.0,
   'michaelhouse cafe': 0.0,
   'midsummer house restaurant': 0.0,
   'nandos': 0.0,
   'nandos city centre': 0.0,
   'panahar': 0.0,
   'peking restaurant': 0.0,
   'pipasha restaurant': 0.0,
   'pizza express': 0.0,
   'pizza express fen ditton': 0.0,
   'pizza hut': 0.0,
   'pizza hut cherry hinton': 0.0,
   'pizza hut city centre': 0.0,
   'pizza hut fen ditton': 0.0,
   'prezzo': 0.0,
   'rajmahal': 0.0,
   'restaurant alimentum': 0.0,
   'restaurant one seven': 0.0,
   'restaurant two two': 0.0,
   'rice boat': 0.0,
   'rice house': 0.0,
   'riverside brasserie': 0.0,
   'royal spice': 0.0,
   'royal standard': 0.0,
   'saffron brasserie': 0.0,
   'saigon city': 0.0,
   'saint johns chop house': 0.0,
   'sala thong': 0.0,
   'sesame restaurant and bar': 0.0,
   'shanghai family restaurant': 0.0,
   'shiraz restaurant': 0.0,
   'sitar tandoori': 0.0,
   'stazione restaurant and coffee bar': 0.0,
   'taj tandoori': 0.0,
   'tandoori palace': 0.0,
   'tang chinese': 0.0,
   'thanh binh': 0.0,
   'the cambridge chop house': 0.0,
   'the copper kettle': 0.0,
   'the cow pizza kitchen and bar': 0.0,
   'the gandhi': 0.0,
   'the gardenia': 0.0,
   'the golden curry': 0.0,
   'the good luck chinese food takeaway': 0.0,
   'the hotpot': 0.0,
   'the lucky star': 0.0,
   'the missing sock': 0.0,
   'the nirala': 0.0,
   'the oak bistro': 0.0,
   'the river bar steakhouse and grill': 0.0,
   'the slug and lettuce': 0.0,
   'the varsity restaurant': 0.0,
   'travellers rest': 0.0,
   'ugly duckling': 0.0,
   'venue': 0.0,
   'wagamama': 0.0,
   'yippee noodle bar': 0.0,
   'yu garden': 0.0,
   'zizzi cambridge': 0.0},
  'pricerange': {'**NONE**': 0.1340777132648503,
   'cheap': 0.0,
   'dontcare': 0.8659222867351497,
   'expensive': 0.0,
   'moderate': 0.0},
  'requested': {'addr': 0.0,
   'area': 0.0,
   'description': 0.0,
   'food': 0.0,
   'name': 0.0,
   'phone': 0.0,
   'postcode': 0.0,
   'pricerange': 0.0,
   'signature': 0.0}},
 'features': {'inform_info': [False,
   False,
   True,
   False,
   True,
   False,
   False,
   True,
   False,
   False,
   False,
   False,
   True,
   False,
   False,
   False,
   False,
   True,
   False,
   False,
   False,
   False,
   True,
   False,
   False],
  'informedVenueSinceNone': [],
  'lastActionInformNone': False,
  'lastInformedVenue': '',
  'offerHappened': False},
 'userActs': [('inform(food="mediterranean")', 0.84415346579983519),
  ('inform(area="east")', 0.037084516069956962),
  ('null()', 0.048530354363153554),
  ('reqmore()', 0.04541708634740408),
  ('confirm(phone)', 0.024814577419650211)]}

    return b1, b2, b3


def main():
    """
    unit test
    :return:
    """

    Settings.init('config/Tut-gp-Multidomain.cfg', 12345)
    Ontology.init_global_ontology()

    b1, b2, b3 = get_test_beliefs()
    '''state1 = DIP_state(b1, domainString='SFRestaurants')
    state2 = DIP_state(b2, domainString='SFRestaurants')
    state3 = DIP_state(b3, domainString='CamRestaurants')'''
    state1 = padded_state(b1, domainString='SFRestaurants')
    state2 = padded_state(b2, domainString='SFRestaurants')
    state3 = padded_state(b3, domainString='CamRestaurants')
    print(state1.get_beliefStateVec('area')[:state1.max_v])
    print(len(state2.get_beliefStateVec('near'))-state2.max_v)
    print(len(state3.get_beliefStateVec('pricerange'))-state3.max_v)
    #print len(state3.get_beliefStateVec('general'))
    s2 = state2.get_beliefStateVec('food')
    s3 = state3.get_beliefStateVec('food')
    a=1
    #print state3.get_beliefStateVec('general')[:state2.max_v]
    #print state2.max_v
    #print state3.max_v


if __name__ == '__main__':
    main()


from agents.basic_rl import BasicQLearner
from agents.abstract_agent import AbstractAgent
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.multioutput import MultiOutputRegressor
from util import NominalToBinary
from tasks.abstract_task import AbstractTask
from policies.basic_policies import EGreedyPolicy
import numpy as np
import json

class EgoAlloLearner(AbstractAgent):

    def __init__(self, memory_depth=1000):

        self.allo_system = BasicQLearner()
        self.allo_system.set_policy(EGreedyPolicy(epsilon=0.9))

        self.ego_system = EgocentricLearningSystem()
        self.memory = []
        self.memory_depth = memory_depth

        self.allocentric_sensory_association = AllocentricSensoryAssociativeMemory()

        self.total_updates = 0

        self.allow_transfer = True


    def select_action(self, state, allowed_actions):

        if self.total_updates > 2000 and self.allow_transfer:
            for act in allowed_actions:
                state_act = pd.DataFrame([dict(state['sensory'], **{'action':act})])
                try:
                    prediction = self.ego_system.predict(state_act)
                except:
                    continue
                predicted_new_state = state['allocentric'].coordinates + prediction['allocentric_change'] # *** this section should be made more generic. right now it's specific to gridworlds
                self.allo_system.update(state=tuple(state['allocentric'].coordinates),
                                        action=act,
                                        reward=prediction['reward'][0],
                                        new_state=tuple(predicted_new_state[0]),
                                        new_allowed_actions=allowed_actions) # ***** what to use for new_allowed_actions?

        return self.allo_system.select_action(state=tuple(state['allocentric'].coordinates),
                                              allowed_actions=allowed_actions)

    def update(self, state, action, reward, new_state, new_allowed_actions):

        # update the allocentric system
        self.allo_system.update(state=tuple(state['allocentric'].coordinates),
                                action=action,
                                reward=reward,
                                new_state=tuple(new_state['allocentric'].coordinates),
                                new_allowed_actions=new_allowed_actions)

        # add the experience to memory
        self.memory.insert(0, {'state':state, 'action':action, 'reward':reward, 'new_state':new_state})
        while len(self.memory) > self.memory_depth:
            self.memory.pop()

        # generate training set and train egocentric system *** this section should be made more generic. right now it's specific to gridworlds
        if len(self.memory) >= self.memory_depth and self.total_updates % 100 == 0:
            state_action_instances = [dict(transition['state']['sensory'], **{'action': transition['action']})
                                      for transition in self.memory]
            state_action_instances = pd.DataFrame(state_action_instances)

            rewards = [transition['reward'] for transition in self.memory]

            allocentric_changes = [np.array(self.get_allocentric_change(transition['state'], transition['new_state']))
                                   for transition in self.memory]

            self.ego_system.fit(state_action_instances, rewards, allocentric_changes)

        # update the associative memory
        self.allocentric_sensory_association.update(state['allocentric'], state['sensory'])

        self.total_updates += 1

    def get_allocentric_change(self, oldstate, newstate):
        new_coords = newstate['allocentric'].coordinates
        old_coords = oldstate['allocentric'].coordinates
        return [new_coords[0] - old_coords[0], new_coords[1] - old_coords[1]]


class AllocentricSensoryAssociativeMemory:

    def __init__(self):
        self.memory = dict()

    # update associative memory
    def update(self, allo_state, sensory):
        sensory_as_string = json.dumps(sensory)

        if allo_state not in self.memory:
            self.memory[allo_state] = dict()

        if sensory_as_string not in self.memory[allo_state]:
            self.memory[allo_state] = {sensory_as_string: 1}
        else:
            self.memory[allo_state][sensory_as_string] += 1

    # predict a sensory experience for a given allocentric state by sampling from the distribution of previous
    # experiences in that state
    def predict_sensory(self, allo_state):
        if allo_state not in self.memory:
            return None

        distribution = self.memory[allo_state]
        total_experiences = sum([x[1] for x in distribution.items()])
        p = np.random.random() * total_experiences
        for experience in distribution:
            if distribution[experience] >= p:
                return json.loads(experience)
            else:
                p -= distribution[experience]


class EgocentricLearningSystem:

    def __init__(self, reward_predictor=None, allocentric_change_predictor=None):
        if reward_predictor is None:
            self.reward_predictor = DecisionTreeRegressor()
        else:
            self.reward_predictor = reward_predictor

        if allocentric_change_predictor is None:
            self.allocentric_change_predictor = MultiOutputRegressor(DecisionTreeRegressor())
        else:
            self.allocentric_change_predictor = allocentric_change_predictor

    def fit(self, state_action_instances, rewards, allocentric_changes):
        state_action_instances.fillna(value='(empty)', inplace=True)

        self.binarizer = NominalToBinary()
        x = self.binarizer.fit_transform(state_action_instances)

        self.reward_predictor.fit(x, rewards)
        self.allocentric_change_predictor.fit(x, allocentric_changes)

    def predict(self, state_action_instances):
        state_action_instances.fillna(value='(empty)', inplace=True)

        x = self.binarizer.transform(state_action_instances)
        reward = self.reward_predictor.predict(x)
        allocentric_change = self.allocentric_change_predictor.predict(x)
        return {'reward': reward,
                'allocentric_change': np.round(allocentric_change)
                }

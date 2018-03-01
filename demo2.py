from agents.basic_rl import BasicQLearner
from agents.egoallo import EgoAlloLearner
from agents.basic_rl import BasicQLearner
from policies.basic_policies import EGreedyPolicy
from tasks.grid_world import BasicGridWorld, SensoryGridWorld
from util import Episode
from matplotlib import pyplot as plt
import numpy as np
import random

grid_world = BasicGridWorld()
grid_world.display()

agent = EgoAlloLearner()
agent.set_policy(EGreedyPolicy(epsilon=0.9))

episode = Episode(agent=agent, task=SensoryGridWorld(grid_world))
reward_history = episode.run(num_steps=1000)

# randomly sample some allocentric states and get the agent to predict what sensory data it will get there
random_states = set()
while len(random_states) < 5:
    random_states.add(random.choice(random.choice(grid_world.grid)))
for state in random_states:
    print('allocentric state: ' + str(state.coordinates) + ', predicted sensory = ' +
          str(agent.allocentric_sensory_association.predict_sensory(state)))

plt.figure()
# plt.plot(np.cumsum(reward_history))
plt.show()


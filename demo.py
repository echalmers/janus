from agents.basic_rl import BasicQLearner
from agents.egoallo import EgoAlloLearner
from agents.basic_rl import BasicQLearner
from policies.basic_policies import EGreedyPolicy
from tasks.grid_world import BasicGridWorld, SensoryGridWorld
from util import Episode
from matplotlib import pyplot as plt
import numpy as np

grid_world = BasicGridWorld()
grid_world.display()

agent1 = BasicQLearner()
agent1.set_policy(EGreedyPolicy(epsilon=0.9))

agent2 = EgoAlloLearner()
agent2.set_policy(EGreedyPolicy(epsilon=0.9))

episode = Episode(agent=agent1, task=grid_world)
reward_history1 = episode.run(num_steps=5000)

episode = Episode(agent=agent2, task=SensoryGridWorld(grid_world))
reward_history2 = episode.run(num_steps=5000)

plt.figure()
plt.plot(np.cumsum(reward_history1))
plt.plot(np.cumsum(reward_history2))
plt.show()


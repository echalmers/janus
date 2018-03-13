from tasks.abstract_task import AbstractTask
import numpy as np
from matplotlib import pyplot
from multiprocessing import Process, Pipe, Queue
import time

class BasicGridWorld(AbstractTask):

    actions = {'left': [0,-1],
               'right': [0,1],
               'up': [-1,0],
               'down': [1,0]}

    square_types = {'wall': {'frequency': 0.1,
                             'color': [0,0,255],
                             },
                    'open': {'frequency': 0.9,
                             'color': [255,255,255],
                             },
                    }

    goal_types = {'goal': {'color': [0,255,0]}}

    size = 10

    def create_link(self, grid_obj1, action, grid_obj2):
        new_grid_state = grid_obj1
        reward = 0

        if grid_obj2.type_name == 'open':
            new_grid_state = grid_obj2
        elif grid_obj2.type_name == 'goal':
            new_grid_state = self.grid[0][0]
            reward = 10
        elif grid_obj2.type_name == 'wall':
            reward = -1

        grid_obj1.links[action] = new_grid_state
        grid_obj1.rewards[action] = reward

    class GridSquare:

        def __init__(self, type_name, coordinates, grid_size):
            self.coordinates = coordinates
            self.type_name = type_name
            self.links = dict()
            self.rewards = dict()
            self.grid_size = grid_size

        def __hash__(self):
            return self.coordinates[0] + self.coordinates[1]*self.grid_size


    def __init__(self):

        # create a separate process to use for plotting
        self.display_q = Queue()
        self.display_proc = Process(target=show_image, args=[self.display_q,])
        self.display_proc.start()

        # generate a random grid
        self.grid = np.random.choice(list(self.square_types.keys()),
                                     size=[self.size,self.size],
                                     p=[self.square_types[type]['frequency'] for type in self.square_types.keys()]).tolist()
        # clear a path to the goal
        state = [0,0]
        while state != [self.size-1, self.size-1]:
            self.grid[state[0]][state[1]] = 'open'

            if state[0] == self.size-1:
                state[1] += 1
            elif state[1] == self.size-1:
                state[0] += 1
            elif np.random.random() < 0.5:
                state[1] += 1
            else:
                state[0] += 1

        # create grid state objects
        for i in range(0, len(self.grid)):
            for j in range(0, len(self.grid[i])):
                self.grid[i][j] = self.GridSquare(self.grid[i][j], [i,j], self.size)

        # add a goal state
        self.grid[self.size - 1][self.size - 1] = self.GridSquare('goal', [self.size - 1, self.size - 1], self.size)

        # create links between neighboring grid states
        for i in range(0, len(self.grid)):
            for j in range(0, len(self.grid[i])):
                for act in self.actions:
                    new_i = i+self.actions[act][0]
                    new_j = j+self.actions[act][1]
                    if new_i >= 0 and new_i < self.size and new_j >= 0 and new_j < self.size:
                        neighbor = self.grid[new_i][new_j]
                    else:
                        self.create_link(self.grid[i][j], act, self.grid[i][j])
                        continue
                    self.create_link(self.grid[i][j], act, neighbor)

        self._current_state = self.grid[0][0]

    def get_current_state(self):
        return self._current_state

    def execute_action(self, action):
        reward = self._current_state.rewards[action]
        self._current_state = self._current_state.links[action]
        return reward

    def get_allowed_actions(self, state):
        return list(self.actions.keys())

    def display(self):
        rgb = np.zeros((self.size, self.size, 3))
        for i in range(0, self.size):
            for j in range(0, self.size):
                square_type = self.grid[i][j].type_name
                if square_type in self.square_types:
                    rgb[i][j][:] = self.square_types[square_type]['color']
                elif square_type in self.goal_types:
                    rgb[i][j][:] = self.goal_types[square_type]['color']
        rgb[self._current_state.coordinates[0]][self._current_state.coordinates[1]][:] = 0

        self.display_q.put(rgb/255)


def show_image(q):
    while True:
        while q.qsize() == 0:
            pyplot.pause(0.1)
        image = None
        while q.qsize() > 0:
            image = q.get()
        pyplot.imshow(image)


class SensoryGridWorld:

    def __init__(self, basicGridWorld):
        self.gridworld = basicGridWorld

    def display(self):
        self.gridworld.display()

    def execute_action(self, action):
        return self.gridworld.execute_action(action)

    def get_allowed_actions(self, state):
        return self.gridworld.get_allowed_actions(state)

    def get_current_state(self):
        allo_state = self.gridworld._current_state
        ego_state = {}

        i = allo_state.coordinates[0]
        j = allo_state.coordinates[1]
        for act in self.gridworld.actions:
            new_i = i + self.gridworld.actions[act][0]
            new_j = j + self.gridworld.actions[act][1]
            if new_i >= 0 and new_i < self.gridworld.size and new_j >= 0 and new_j < self.gridworld.size:
                ego_state[act] = self.gridworld.grid[new_i][new_j].type_name

        return {'allocentric': allo_state, 'sensory': ego_state}


if __name__ == "__main__":
    basicGW = BasicGridWorld()
    gw = SensoryGridWorld(basicGW)
    basicGW.display()

    for i in range(0,100):

        state = gw.get_current_state()
        print(state['allocentric'].coordinates)
        print(state['sensory'])
        act = input('action: ')
        gw.execute_action(act)
        basicGW.display()




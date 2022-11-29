import random
import numpy as np

class random_selection:

    def __init__(self, n_agents):
        self.n_agents = n_agents
        self.agent_list = [i for i in range(self.n_agents)]
        '''self.weights = np.ones(self.n_agents)
        self.time_window = np.zeros(self.n_agents)
        self.count = self.n_agents*2'''
        #self.previous_pick = np.zeros(self.n_agents)

    def persistent_communication(self):
        id_of_agent_activated = random.sample(self.agent_list, k=random.randint(1, len(self.agent_list)))
        for i in id_of_agent_activated:
            self.agent_list.remove(i)
        if len(self.agent_list) == 0:
            self.agent_list = [i for i in range(self.n_agents)]
        return id_of_agent_activated

    def extraction(self):
        self.persistent_communication()
        iter = random.randint(1, self.n_agents)
        result = []
        done = False
        i = 0
        while not done:
            extraction = random.choices(range(self.n_agents), weights=self.weights)
            if i == 0 or (extraction[0] not in result):
                result += extraction
                self.time_window[extraction[0]] = 1
                i += 1
            if i == iter:
                done = True
        return result

    def persistent_communication_old(self):
        self.count -= 1
        for i in range(self.n_agents):
            if self.time_window[i] == 1:
                self.weights[i] = 1
            else:
                self.weights[i] += 1
        if self.count == 0:
            self.count = self.n_agents*2
            self.weights = np.ones(self.n_agents)
        return


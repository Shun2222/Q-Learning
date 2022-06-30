import numpy as np

class QLearning:
    def __init__(self, size):
        # ser parameter 
        self.gamma = .99
        self.alpha = .2
        self.epsilon = .05
        self.decayRate = 1.
        
        self.QTable = np.zeros(size)
        self.isTrain = True

        self.prevAction = None
        self.prevState = None
        self.action = None
        self.state = None

    def epsGreedyAction(self):

        if np.random.uniform() < self.epsilon && self.isTrain:
            act = np.random.random_integers(0, self.QTable[self.state].shape[0]-1)
        else:
            act = np.argmax(self.QTable[self.state])

        self.epsilon = self.epsilon * self.decayRate
        if self.epsilon < .01:
            self.epsilon = .01

        self.prevAction = self.action
        self.action = act

    def updateQTable(self, reward):




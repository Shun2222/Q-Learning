from model.agent import Agent
from model.prey import Prey
from model.environment import Env

import configparser
import json

class GameSystem:
    def __init__(self, agentNum, preyNum, size):
        config_ini = configparser.ConfigParser()
        config_ini.read("config.ini", encoding='utf-8')

        self.isFinished = False
        seflf.actionNum = json.loads(config.get("action","ACTION_NUM"))
        self.agents = np.array([])
        for i in range(agentNum):
            agents = np.append(agents, Agent())

        self.prey = np.array([])
        for i in range(preyNum):
            preys = np.append(prey, Prey())

        self.env = Env(size);

    def step(self):
        self.action = np.array([])
        for prey in self.preys:
            action = prey.action()
            self.env.move(prey.id, action)

        for agent in self.agents:
            action = agent.action()
            self.env.move(agent.id, action) # agent movement and give reward to agent 
            agent.learn()
            self.action = np.append(self.action, action)
            

    def isFinished(self):
        isAllCatched = True
        for prey in preys:
            if !prey.isCatched():
                isAllCatched = False

        isProblem = self.env.isProbrem()
        return (isProblem || isAllCatched)




             

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy


FILED_TYPE = {
    "N": 0,  #通常
    "W": 1,  #壁
    }

ACTIONS = {
    "UP": 0, 
    "DOWN": 1, 
    "LEFT": 2, 
    "RIGHT": 3,
    "STAY": 4
    }


class Prey:
    def __init__(self, pid):
        self.pid = pid
        self.is_captured = False

    def act(self):
        return np.random.randint(5)

    def reset(self):
        self.is_captured = False


class PredatorsPursuitGame:

    def __init__(self, agents):

        self.map = [[0,0,0,0,0], 
                    [0,0,0,0,0], 
                    [0,0,0,0,0], 
                    [0,0,0,0,0], 
                    [0,0,0,0,0]]

        self.agents = agents
        self.agents_pos = {}
        for agent in self.agents:
            pos = self._init_pos(self.agents_pos.values())
            self.agents_pos[agent.aid] = pos

        self.prey = Prey(0)
        self.preys = [self.prey]

        pos = self._init_pos(self.agents_pos.values())
        self.preys_pos = {0:pos}

    def step(self, actions):
        """
            全エージェントの行動の実行
            状態, 報酬、ゴールしたかを返却
        """

        # ハンターの移動
        for aid, action in enumerate(actions):
            x, y = copy.deepcopy(self.agents_pos[aid])
            to_x, to_y, is_collision = self.move_agent(x, y, action)
            self.agents_pos[aid] = (to_x, to_y)

        # preyの行動
        for pid, prey in enumerate(self.preys):
            to_x, to_y = self.move_prey(prey)
            self.preys_pos[prey.pid] = (to_x, to_y)

        # 終了判定(すべてのハンターがpreyに隣接しているか)
        is_capture = self._is_capture() 
        is_terminal = self._is_terminal()
        reward = self._compute_reward(is_capture, is_collision)

        obss = {}
        for agent in self.agents:
            obs = self.create_observation(agent)
            obss[agent.aid] = obs

        return obss, reward, is_terminal

    def _init_pos(self, poss=[]):
        """
            被らないposデータの生成 
        """

        x = np.random.randint(0, len(self.map[0]))
        y = np.random.randint(0, len(self.map))

        while (x, y) in poss:
            x = np.random.randint(0, len(self.map[0]))
            y = np.random.randint(0, len(self.map))

        return x, y

    def move(self, x, y, action):
        to_x = copy.deepcopy(x)
        to_y = copy.deepcopy(y)
        if action == ACTIONS["UP"]:
            to_y += -1
        elif action == ACTIONS["DOWN"]:
            to_y += 1
        elif action == ACTIONS["LEFT"]:
            to_x += -1
        elif action == ACTIONS["RIGHT"]:
            to_x += 1

        return to_x, to_y

    def move_agent(self, x, y, action):
        is_collision = False
        to_x, to_y = copy.deepcopy(x), copy.deepcopy(y)
        if self._is_possible_action(x, y, action):
            to_x, to_y = self.move(x, y, action)
        else:
            is_collision = True

        return to_x, to_y, is_collision

    def move_prey(self, prey):
        x, y = copy.deepcopy(self.preys_pos[prey.pid])
        action = prey.act()
        while self._is_possible_action(x, y, action) is False:
            action = prey.act()

        to_x, to_y = self.move(x, y, action)

        return to_x, to_y

    def in_map(self, agent_id, action):
        """ 
            実行可能な行動かどうかの判定
        """
        x, y = self.agents_pos[agent_id]
        to_x = copy.deepcopy(x)
        to_y = copy.deepcopy(y)
        if action == ACTIONS["STAY"]:
            return True
        else:
            to_x, to_y = self.move(to_x, to_y, action)

            if len(self.map) <= to_y or 0 > to_y:
                return False
            elif len(self.map[0]) <= to_x or 0 > to_x:
                return False

        return True

    def create_observation(self, agent):
        """
            観測情報の生成
        """
        obs = [self.agents_pos[agent.aid]]
        for agent2 in self.agents:
            if agent2 is not agent:
                obs.append(self.agents_pos[agent2.aid])

        for prey in self.preys:
            obs.append(self.preys_pos[prey.pid])

        return obs

    def _is_capture(self):
        """
            preyを捕まえたかの判定
        """
        is_capture = False
        # 各preyの隣接
        for prey in self.preys:
            if self._check_adjacent(prey):
                is_capture = True

        return is_capture

    def _check_adjacent(self, prey):
        """
            preyの隣接しているところにエージェントがいるかどうかの確認
        """
        to_x, to_y = self.preys_pos[prey.pid]
        nb_adjacent_agents = 0
        for action in np.arange(0, 4):
            to_x, to_y = self.move(to_x, to_y, action)

            if (to_x, to_y) in self.agents_pos.values():
                nb_adjacent_agents += 1

        if nb_adjacent_agents >= 2:
            prey.is_captured = True
            return True
        else:
            return False

    def _is_terminal(self):
        """
            すべてのpreyを捕まえたかどうかの確認
        """
        is_terminal = True

        # 各preyの隣接
        for prey in self.preys:
            if prey.is_captured is False:
                is_terminal = False

        return is_terminal

    def _is_wall(self, x, y):
        """
            x, yが壁かどうかの確認
        """
        if self.map[y][x] == FILED_TYPE["W"]:
            return True
        else:
            return False

    def _is_possible_action(self, x, y, action):
        """ 
            実行可能な行動かどうかの判定
        """
        if action == ACTIONS["STAY"]:
            return True
        else:
            to_x, to_y = self.move(x, y, action)

            if len(self.map) <= to_y or 0 > to_y:
                return False
            elif len(self.map[0]) <= to_x or 0 > to_x:
                return False
            elif self._is_wall(to_x, to_y) or self._is_agent_or_prey(to_x, to_y):
                return False

        return True

    def _is_agent_or_prey(self, x, y):
        """
            選択した状態にエージェントもしくはpreyがいるかの確認 
        """

        if (x, y) in self.agents_pos.values():
            return True
        elif (x, y) in self.preys_pos.values():
            return True

        return False

    def _compute_reward(self, is_capture, is_obstacle):
        if is_capture:
            return 100
        elif is_obstacle:
            return -1
        else:
            return 0

    def reset(self):
        self.agents_pos = {}
        for agent in self.agents:
            pos = self._init_pos(self.agents_pos.values())
            self.agents_pos[agent.aid] = pos

        pos = self._init_pos(self.agents_pos.values())
        self.preys_pos = {0: pos}
        self.prey.reset()

    def print_current_map(self):
        """
            デバック用
        """
        current_map = copy.deepcopy(self.map)
        for x, y in self.agents_pos.values():
            current_map[y][x] = "A"

        for x, y in self.preys_pos.values():
            current_map[y][x] = "P"

        current_map = np.array(current_map)
        print(current_map)

    def get_agents_ini_pos(self):
        return self.agents_ini_pos

    def get_agents_pos(self):
        return self.agents_pos


class EpsGreedyQPolicy:
    """
        ε-greedy
    """
    def __init__(self, epsilon=.05, decay_rate=1):
        self.epsilon = epsilon
        self.decay_rate = decay_rate

    def select_action(self, q_values):
        assert q_values.ndim == 1
        nb_actions = q_values.shape[0]

        if np.random.uniform() < self.epsilon:  # random行動
            action = np.random.random_integers(0, nb_actions-1)
        else:   # greedy 行動
            action = np.argmax(q_values)

        self.decay_eps_rate()

        return action

    def decay_eps_rate(self):
        self.epsilon = self.epsilon*self.decay_rate
        if self.epsilon < .01:
            self.epsilon = .01

    def select_greedy_action(self, q_values):
        assert q_values.ndim == 1
        nb_actions = q_values.shape[0]
        action = np.argmax(q_values)

        return action


class QLearningAgent:

    def __init__(self, aid=0, alpha=.2, policy=None, gamma=.99, actions=None, observation=None):
        self.aid = aid
        self.alpha = alpha
        self.gamma = gamma
        self.policy = policy
        self.reward_history = []
        self.actions = actions
        self.state = str(observation)
        self.ini_state = str(observation)
        self.previous_state = None
        self.previous_action_id = None
        self.q_values = self._init_q_values()
        self.traning = True

    def _init_q_values(self):
        """
           Q テーブルの初期化
        """
        q_values = {self.state: np.repeat(0., len(self.actions))}
        return q_values

    def init_state(self):
        """
            状態の初期化 
        """
        self.previous_state = copy.deepcopy(self.ini_state)
        self.state = copy.deepcopy(self.ini_state)
        return self.state

    def act(self, q_values=None):
        if self.traning:
            action_id = self.policy.select_action(self.q_values[self.state])
        else:
            action_id = self.policy.select_greedy_action(self.q_values[self.state])
        self.previous_action_id = action_id
        action = self.actions[action_id]
        return action

    def observe(self, next_state, reward=None):
        """
            次の状態と報酬の観測 
        """
        next_state = str(next_state)
        if next_state not in self.q_values: # 始めて訪れる状態であれば
            self.q_values[next_state] = np.repeat(0.0, len(self.actions))

        self.previous_state = copy.deepcopy(self.state)
        self.state = next_state

        if self.traning and reward is not None:
            self.reward_history.append(reward)
            self.learn(reward)

    def learn(self, reward):
        """
            Q値の更新 
        """
        q = self.q_values[self.previous_state][self.previous_action_id] # Q(s, a)
        max_q = max(self.q_values[self.state]) # max Q(s')
        # Q(s, a) = Q(s, a) + alpha*(r+gamma*maxQ(s')-Q(s, a))
        self.q_values[self.previous_state][self.previous_action_id] = q + (self.alpha * (reward + (self.gamma*max_q) - q))


if __name__ == '__main__':
    nb_episode = 10000

    actions = np.arange(5)
    agent1 = QLearningAgent(aid=0, policy=EpsGreedyQPolicy(epsilon=1., decay_rate=.99), actions=actions)
    agent2 = QLearningAgent(aid=1, policy=EpsGreedyQPolicy(epsilon=1., decay_rate=.99), actions=actions)
    agents = [agent1, agent2]

    game = PredatorsPursuitGame(agents)

    result = []
    for episode in range(nb_episode):
        is_capture = False
        agents_pos = game.get_agents_pos()

        for agent in agents:
            agent.observe(agents_pos[agent.aid])

        nb_step = 0
        while is_capture is False:
            actions = []
            for agent in agents:
                action = agent.act()
                while game.in_map(agent.aid, action) is False:
                    action = agent.act()
                actions.append(action)

            states, r, is_capture = game.step(actions)
            agent1.observe(states[0], reward=r)
            agent2.observe(states[1], reward=r)
            # game.print_current_map()  # 現状のマップの表示
            nb_step += 1
        result.append(nb_step)
        game.reset()

    result = pd.Series(result).rolling(50).mean().tolist()  # 捕らえるのにかかったステップ数の移動平均の計算

    plt.plot(np.arange(len(result)),result)
    plt.xlabel("Episodes")
    plt.ylabel("steps")
    plt.legend()
    plt.savefig("result.jpg")
    plt.show()


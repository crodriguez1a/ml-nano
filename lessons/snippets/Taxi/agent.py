import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))

    def q_learning(self, state, action, reward, next_state, done, epsilon, alpha=0.01, gamma=1.0):
        # pick next action A
        action = self.select_action(state)
        # update Q
        self.Q[state][action] = self.update_Q(self.Q[state][action], np.max(self.Q[next_state]),
                                              reward, alpha, gamma)

    def sarsa(self, state, action, reward, next_state, done, alpha=0.01, gamma=1.0, steps=2):
        # limit number of time steps per episode
        for t_step in np.arange(steps):
            # pick next action A'
            next_action = self.select_action(next_state)
            # update TD estimate of Q
            self.Q[state][action] = self.update_Q(self.Q[state][action],
                                                  self.Q[next_state][next_action],
                                                  reward, alpha, gamma,)

    def update_Q(self, Qsa, Qsa_next, reward, alpha, gamma):
        """
        updates the action-value function estimate
        using the most recent time step
        """
        return Qsa + (alpha * (reward + (gamma * Qsa_next) - Qsa))

    def epsilon_greedy_probs(self, Q_s, eps=0.001):
        """
        obtains the action probabilities corresponding
        to epsilon-greedy policy
        """
        epsilon = eps
        policy_s = np.ones(self.nA) * epsilon / self.nA
        policy_s[np.argmax(Q_s)] = 1 - epsilon + (epsilon / self.nA)

        return policy_s

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        # get epsilon-greedy action probabilities
        policy_s = self.epsilon_greedy_probs(self.Q[state])

        return np.random.choice(self.nA, p=policy_s)

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        self.sarsa(state, action, reward, next_state, done)
        # self.q_learning(state, action, reward, next_state, done, epsilon=0.001)
        # self.Q[state][action] += 1

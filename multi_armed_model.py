import numpy as np

class MultiArmedBandit:
    def __init__(self, true_means):
        # true_means: list or array of true reward means for each arm
        self.true_means = np.array(true_means)
        self.n_arms = len(true_means)

    def pull(self, arm):
        # Reward is sampled from a normal distribution around the arm's true mean
        return np.random.randn() + self.true_means[arm]


class EpsilonGreedyAgent:
    def __init__(self, n_arms, epsilon=0.1):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.counts = np.zeros(n_arms)      # how many times each arm was chosen
        self.values = np.zeros(n_arms)      # estimated value of each arm

    def select_arm(self):
        # Exploration vs exploitation
        if np.random.rand() < self.epsilon:
            # explore
            return np.random.randint(self.n_arms)
        else:
            # exploit
            return np.argmax(self.values)

    def update(self, arm, reward):
        # Incremental mean update
        self.counts[arm] += 1
        n = self.counts[arm]
        value = self.values[arm]
        self.values[arm] = value + (reward - value) / n


if __name__ == "__main__":
    # Define a 4-armed bandit
    true_means = [1.0, 1.5, 2.0, 0.5]  # hidden true rewards
    bandit = MultiArmedBandit(true_means)

    agent = EpsilonGreedyAgent(n_arms=4, epsilon=0.1)

    n_steps = 1000
    rewards = []

    for t in range(n_steps):
        arm = agent.select_arm()
        reward = bandit.pull(arm)
        agent.update(arm, reward)
        rewards.append(reward)

    print("Estimated values:", agent.values)
    print("True means:", true_means)
    print("Total reward:", np.sum(rewards))

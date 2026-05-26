import numpy as np
import gymnasium as gym
import logging

# -------------------------
# Logger
# -------------------------
logger = logging.getLogger("rl")
logger.setLevel(logging.INFO)


# -------------------------
# LINEAR AGENT
# -------------------------
class LinearAgent:
    def __init__(self):
        # 4 weights for CartPole observation
        self.parameters = np.random.randn(4)

    def next_action(self, observation):
        # linear decision: w · x
        return 0 if np.dot(self.parameters, observation) < 0 else 1


# -------------------------
# TRAINING ENV RUNNER
# -------------------------
class Harness:

    def run_episode(self, env, agent):
        observation, info = env.reset()
        total_reward = 0

        for _ in range(1000):

            action = agent.next_action(observation)

            observation, reward, terminated, truncated, info = env.step(action)

            total_reward += reward

            if terminated or truncated:
                break

        return total_reward


# -------------------------
# RANDOM SEARCH TRAINING
# -------------------------
def random_search():
    env = gym.make("CartPole-v1")  # change to render_mode="human" if you want visuals

    harness = Harness()

    best_reward = -1
    best_agent = None

    for i in range(100):   # number of trials

        agent = LinearAgent()
        reward = harness.run_episode(env, agent)

        print(f"Trial {i}: Reward = {reward}")

        if reward > best_reward:
            best_reward = reward
            best_agent = agent

    print("\n====================")
    print("BEST REWARD:", best_reward)
    print("====================")

    return best_agent


# -------------------------
# RUN
# -------------------------
if __name__ == "__main__":
    random_search()
import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="human")

while True:  # 🔁 runs forever (round & round)
    observation, info = env.reset()

    for step in range(1000):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            break   # restart episode

env.close()
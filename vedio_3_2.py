import gymnasium as gym

# Create MountainCar environment
env = gym.make("MountainCar-v0", render_mode="human")

while True:
    observation, info = env.reset()

    for _ in range(200):
        action = env.action_space.sample()  # random move (left/right/no push)

        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            break

env.close()
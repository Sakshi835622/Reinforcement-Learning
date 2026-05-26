import gymnasium as gym

# Create environment (modern version)
env = gym.make("CartPole-v1", render_mode="human")

obs, info = env.reset()

for step in range(20):

    env.render()

    action = env.action_space.sample()

    obs, reward, terminated, truncated, info = env.step(action)

    print(f"Step: {step}, Action: {action}, Reward: {reward}")

    if terminated or truncated:
        obs, info = env.reset()
        print("Episode finished, resetting environment.")

env.close()
import gymnasium as gym
import time

# Create environment
env = gym.make("CartPole-v1", render_mode="human")

# Reset environment
observation, info = env.reset()

for step in range(500):

    # Render frame
    env.render()

    # Take random action
    action = env.action_space.sample()

    # Step environment
    observation, reward, terminated, truncated, info = env.step(action)

    # Check if episode is done
    done = terminated or truncated

    # Unpack state
    x, x_dot, theta, theta_dot = observation

    # Print values
    print("\n----------------------")
    print(f"Step: {step}")
    print(f"Cart Position      : {x:.3f}")
    print(f"Cart Velocity      : {x_dot:.3f}")
    print(f"Pole Angle         : {theta:.3f}")
    print(f"Pole Angular Vel.  : {theta_dot:.3f}")
    print(f"Action Taken       : {action}")
    print(f"Reward             : {reward}")

    # Slow down simulation
    time.sleep(0.05)

    # If episode ends, restart or break
    if done:
        print("\nEpisode Finished")
        break

# Close environment
env.close()
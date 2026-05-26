import numpy as np

num_bandits = 4
num_contexts = 3
Q = np.zeros((num_contexts, num_bandits))
count = np.ones((num_contexts, num_bandits))

def get_context():
    return np.random.randint(0, num_contexts)

def get_reward(context, action):
    rewards = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0]
    ]
    return rewards[context][action]

episodes = 1000
epsilon = 0.1

for i in range(episodes):

    context = get_context()

    if np.random.rand() < epsilon:
        action = np.random.randint(num_bandits)
    else:
        action = np.argmax(Q[context])

    reward = get_reward(context, action)

    count[context][action] += 1

    Q[context][action] += (
        reward - Q[context][action]
    ) / count[context][action]

print("Q Table:\n", Q)

for c in range(num_contexts):
    print("Best action for context", c, "=", np.argmax(Q[c]))
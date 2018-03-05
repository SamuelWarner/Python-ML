from unityagents import UnityEnvironment
from mazex import MazeX
import numpy as np
import random

# Load environment
env = UnityEnvironment(file_name="..\Environments\BasicDriving\BasicDriving.exe", worker_id=0)

# Create network with 5 inputs, 2 hidden layers, and 3 outputs.
net = MazeX([5, 15, 9, 3], ['relu', 'relu', 'lin'], learning_constant=0.001)

# initial parameters
epsilon = 0.9              # Chance of choosing random action
epsilon_min = 0.1          # Minimum epsilon value after decaying
epsilon_decay = 0.99       # Rate of epsilon decay
gamma = 0.9                # Discounting of future rewards. Lower = short term reward focus.
episodes = 60              # Number of games to play
training = True            # If training is true render settings are reduced to speed training.
action_space = range(3)    # Available actions: left, right, or straight.

# Iterate through multiple games
for e in range(episodes):

    # reset state in the beginning of each game
    state = env.reset(train_mode=training)["Brain"].states

    # format state for network input
    state = np.array(state)

    score = 0        # Start score for this game at 0
    total_steps = 0  # Frame or "step" count to measure how long each game was played for

    # Start game loop and run for 2000 frames before terminating.
    for u in range(2000):

        # Add this frame to frame/step count
        total_steps += 1

        # Find our q-values(predicted reward values for each action) for this current game state
        qvals = net.forward(state)

        # pick best q-value action or a random one if epsilon is high enough
        if epsilon > random.random():
            action = random.choice(action_space)
        else:
            action = np.uint32(np.argmax(qvals)).item()

        # take action and get the environments feedback
        feedback = env.step(action)

        # organize environment feedback
        next_state = np.array(feedback["Brain"].states)
        reward = feedback["Brain"].rewards[0]
        done = feedback["Brain"].local_done[0]
        score += reward

        # Find new q-values for new state
        new_qvals = net.forward(next_state)

        # Find the max q value for our new state from the calculated values
        maxQ = np.amax(new_qvals)

        # Create the target q value the network should be aiming for to update our original prediction
        if not done:
            updated_q_value = reward + (gamma * maxQ)
        else:
            updated_q_value = reward

        # Create the training target for the network
        training_vector = qvals
        training_vector[0, action] = updated_q_value

        # Train the network on the new target
        net.train(state, training_vector)

        # Make next_state the new current state.
        state = next_state

        # If the environment returned an agent state of done then this game is over.
        if done:
            break

    # Display results of this game in console
    print(f"episode: {e}/{episodes}, total steps: {total_steps}, epsilon: {epsilon},  Score: {score}")

    # Decay epsilon for the next game to slowly reduce random behaviour
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

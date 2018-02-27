'''
Reinforcement learning code for CartPole game from openai. This code is only used to play the game with a previously
learned network to verify that the given network weights actually solve the CartPole problem.

Author Samuel Warner
'''
import gym
from mazex import MazeX
import numpy as np

if __name__ == "__main__":
    # initialize gym environment and the agent
    env = gym.make('CartPole-v0')

    # Create network with the same layout as the one that we learned with and load the learned weights
    net = MazeX([4, 64, 128, 256, 128, 64, 2], ['relu', 'relu', 'relu', 'relu', 'relu', 'lin'])
    net.load_weights("CartPoleLearnedWeights.npy")

    episodes = 100    # Number of games to play

    # Iterate through the games
    for e in range(episodes):

        # reset state in the beginning of each game
        state = env.reset()

        # format state for network input
        state = np.array([state])

        for score in range(200):
            # Render the game for visual feedback
            env.render()

            # find our q-values for this state
            qvals = net.forward(state)

            # pick best q-value action or a random one
            action = np.argmax(qvals)

            # take action and get new state
            next_state, reward, done, info = env.step(action)

            # convert state for network input
            next_state = np.array([next_state])

            # When game ends: Print result, break current game loop
            if done:
                # If we got a score under 200(0-199) the learning agent has failed.
                if score < 199:
                    print(f"FAILED with Score: {score+1}")
                    break
                else:
                    print(f"Score: {score+1}")
                    break

            # make next_state the new current state.
            state = next_state

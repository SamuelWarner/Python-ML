'''
Reinforcement learning code for CartPole game from openai. Uses training during game as well as replay memory to solve
the cartpole game. Replay is conducted on games with scores 0.5 standard deviations above the mean.

Author Samuel Warner
'''

import gym
from mazex import MazeX
import numpy as np
import random
import matplotlib.pyplot as plt
import time
import statistics as st

# Include to time learning
start = time.time()

if __name__ == "__main__":

    # initialize gym environment and the agent
    env = gym.make('CartPole-v0')

    net = MazeX([4, 64, 128, 256, 128, 64, env.action_space.n], ['relu', 'relu', 'relu', 'relu', 'relu', 'lin'],
                learning_constant=0.0001)

    # initial parameters
    epsilon = 1.0            # Chance of choosing random action
    epsilon_min = 0.20       # If epsilon is set to decay this is the lowest it will go.
    epsilon_decay = 0.999    # Rate of epsilon decay
    gamma = 0.9              # Discounting of future rewards. Lower = short term reward focus.
    episodes = 15000         # Number of games to play
    replay = []              # Replay memory
    buffer = 5000            # How many moves to store in memory
    batch_size = 300         # Batch size to select from memory for replay training
    replay_cursor = 0        # Position of next write to memory
    loss_reward = 1.0        # Reward for game over state.
    score_data = []          # List for storing game scores
    average_score = []       # Stores scores averaged over 14 games
    render = False           # Should the game be rendered

    all_scores = []          # List to store the score of every game
    temp_replay = []         # List to store game actions and observation for addition to memory after game ends

    # Iterate the game
    for e in range(episodes):
        # reset state in the beginning of each game
        state = env.reset()

        # format state for network input
        state = np.array([state])

        temp_replay = []

        for score in range(500):
            # Used to render the game for visual feedback
            if render:
                env.render()

            # find our q-values for this state
            qvals = net.forward(state)

            # pick best q-value action or a random one
            if epsilon > random.random():
                action = env.action_space.sample()
            else:
                action = np.argmax(qvals)

            # take action and get new state and reward
            next_state, reward, done, info = env.step(action)

            # convert state for network input
            next_state = np.array([next_state])

            # Find new q-values for new state
            new_qvals = net.forward(next_state)

            # Find max q value for our new state
            maxQ = np.amax(new_qvals)

            # create the target q value the network should be aiming for
            if not done:
                updated_q_value = reward + (gamma * maxQ)
            else:
                updated_q_value = loss_reward


            # create the training target for the network
            training_vector = qvals
            training_vector[0, action] = updated_q_value
            net.train(state, training_vector)

            temp_replay.append((state, action, reward, next_state))

            # When game ends: Save data for graph, print result, break current game loop
            if done:
                # store result
                all_scores.append(score)

                # after the first 21 games
                if len(all_scores) > 21:

                    # Determine mean and standard deviation of all scores
                    av = st.mean(all_scores[-20:])
                    std = st.stdev(all_scores[-20:])

                    # If the score of the most recent game was greater than a +0.5 standard deviation
                    if score >= int(av + std * 0.5):

                        #  Store latest move and observations in replay memory at correct position
                        for k in temp_replay:
                            if len(replay) < buffer:
                                replay.append(k)

                            # Overwrite old data if memory is full
                            else:
                                if (replay_cursor < (buffer - 1)):
                                    replay_cursor += 1
                                else:
                                    replay_cursor = 0

                                replay[replay_cursor] = k

                # store an average score every 15 games for the last 15 games
                if len(score_data) > 14:
                    average_score.append(sum(score_data)/len(score_data))
                    score_data = []
                    score_data.append(score)
                else:
                    score_data.append(score)

                # Display results of this game in console
                print(f"episode: {e}/{episodes}, score: {score}, epsilon: {epsilon}")

                # Break the current game loop to end game
                break

            # make next_state the new current state.
            state = next_state

        # If memory is full, replay some previous game actions to train network
        if len(replay) == buffer:
            # random sample a few actions to train on
            minibatch = random.sample(replay, batch_size)

            for memory in minibatch:
                # Get max_Q(S',a)
                old_state, action, reward, new_state = memory

                # find the max q for state S+1 with the net
                old_qvals = net.forward(old_state)
                newer_qvals = net.forward(new_state)
                maxq = np.amax(newer_qvals)

                # determine what the actual Q value should be for this action
                if reward == loss_reward:
                    update = reward + (gamma * maxq)
                else:
                    update = loss_reward

                # update the qvals for the given action
                old_qvals[0, action] = update

                # train on updated values
                net.train(old_state, old_qvals)

        # Decay epsilon over time to reduce random behaviour
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

    # add the final average score data point.
    average_score.append(sum(score_data)/len(score_data))

    # Graph and save the average scores to show learning over time
    plt.plot(average_score)
    plt.ylabel('Average Score')
    plt.xlabel("Number of Games (n/15)")
    plt.title('Score Average per 15 Games')
    plt.savefig('CartPoleGraph_StDev03.png')

    # Save the learned weights
    net.save_weights("CartPole_stdev")

    # Display runtime
    print(f"\n\nTotal training time {((time.time() - start)/60)} minutes.")


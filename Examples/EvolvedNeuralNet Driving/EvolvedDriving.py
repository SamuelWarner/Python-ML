from unityagents import UnityEnvironment
from mazee import MazeE
import numpy as np
import random
import copy

# Load environment
env = UnityEnvironment(file_name="..\Environments\EvolvedDriving\EvolvedNeuralNets.exe", worker_id=0)

# Setup start parameters
training = True       # Bool to determine if environment runs in training mode or inference mode.
generations = 3000    # Number of generations to evaluate
NeuralNets = []       # List of neural networks, one for each agent in a generation
scores = []           # List of scores achieved by each agent in a generation
game_length = 550     # Length in frame steps for each generation
population_size = 10  # Number of agents in a generation
actions = []          # Create empty actions list to store actions chosen by agents for each frame

# Iterate through the generations
for gen in range(generations):

    # reset the environment and store associated environment info
    data = env.reset(train_mode=training)

    # If this is the first generation create a set of nets with random weights to begin evolution.
    if gen == 0:
        for a in range(len(data["BotBrain"].agents)):
            NeuralNets.append(
                MazeE([env.brains["BotBrain"].state_space_size, 12, env.brains["BotBrain"].action_space_size],
                      ["sig", "sig"], weight_min=-4, weight_max=4))

    # Genetic algorithm to determine next generation given the performance of the last one
    else:
        # Find the best scoring neural net and store it while removing it from all other lists
        k = scores.index(max(scores))
        best = NeuralNets[k]
        NeuralNets.pop(k)
        scores.pop(k)

        # Find the second best scoring neural net and store a reference to it.
        k = scores.index(max(scores))
        second_best = NeuralNets[k]

        # Keep lasts games neural nets in a temporary list and clear the main neural net list so it can be repopulated.
        temp = copy.copy(NeuralNets)
        NeuralNets.clear()

        # Breed the best performing network with 5 other networks at random transfering 70% of it's genes to the child
        for i in range(5):
            NeuralNets.append(best.breed(random.choice(temp), .7))

        # Create one child from breeding the best and second best networks
        NeuralNets.append(best.breed(second_best, .5))

        # Mutate the recently bred neural networks to further introduce some slight variance in behavior
        for net in NeuralNets:
            net.mutate(0.05)

        # Fill out the remaining population with new genetic material in the form of a few new random weighted networks
        for i in range(population_size-len(NeuralNets)):
            NeuralNets.append(
                MazeE([env.brains["BotBrain"].state_space_size, 12, env.brains["BotBrain"].action_space_size],
                      ["sig", "sig"], weight_min=-4, weight_max=4))


    # Get the initial states of each agent
    states = data["BotBrain"].states

    scores.clear()  # Start the game with an empty score list
    frames = 0      # Initialize a frame counter at 0

    # Begin the main loop which collects agent states and takes actions each frame till the loop is broken
    while True:

        # Increment the frame count
        frames += 1

        # Count how many agents are available to receive actions
        agent_count = len(data["BotBrain"].agents)

        # If no agents are left, or all agents are done, the game is over and we break the loop.
        if agent_count == 0 or sum(data["BotBrain"].local_done) == agent_count:
            break

        # Clear the action choice list to receive new actions for this frame
        actions.clear()

        # Fill actions list with each neural nets chosen actions
        for i in range(agent_count):
            decision = NeuralNets[i].forward(np.array([data["BotBrain"].states[i]]))[0]
            actions.append((decision[0]*2)-1)  # steering input, originally 0 to 1 but modified to be -1 to 1
            actions.append(decision[1])  # Speed input

        # Send actions to environment and store returned environment info
        data = env.step(actions)

        # store each agents score after taking the chosen action
        scores = data["BotBrain"].rewards

        # If we have exceeded the desired game length we terminate this generation by breaking the loop
        if frames > game_length:
            break

    # Print some feedback for the user to determine how good the best agent is performing
    print(f"Generation {gen} best score: {max(scores)}")

    # Extend game length if certain scores are hit. Used to prevent bad behaviour such as driving in a circle.
    if max(scores) > 40 and game_length < 1500:
        game_length = 1500
    elif max(scores) > 120 and game_length < 4000:
        game_length = 4000

"""
Optional code to save the best weights achieved by running this script:

NeuralNets[scores.index(max(scores))].save_weights("SaveFileName") 
"""

# Close the game environment
env.close()

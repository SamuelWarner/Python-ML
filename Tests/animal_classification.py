"""
Animal classification example. Data taken from:
http://neuroph.sourceforge.net/tutorials/zoo/classification_of_animal_species_using_neural_network.html

#Inputs are the animals attributes:
1. hair   2. feathers   3. eggs   4. milk    5. airborne  6. aquatic  7. predator  8. toothed
9. backbone 10. breathes  11. venomous  12. fins   13. legs    14. tail  15. domestic  16. catsize

*legs normalized at max 8 legs.

The possible classification groups are:

1) aardvark, antelope, bear, boar, buffalo, calf, cavy, cheetah, deer, dolphin, elephant, fruitbat, giraffe, girl,
goat, gorilla, hamster, hare, leopard, lion, lynx, mink, mole, mongoose, opossum, oryx, platypus, polecat, pony,
porpoise, puma, pussycat, raccoon, reindeer, seal, sealion, squirrel, vampire, vole, wallaby, wolf
2) chicken, crow, dove, duck, flamingo, gull, hawk, kiwi, lark, ostrich, parakeet, penguin, pheasant, rhea, skimmer,
skua, sparrow, swan, vulture, wren
3) pitviper, seasnake, slowworm, tortoise, tuatara
4) bass, carp, catfish, chub, dogfish, haddock, herring, pike, piranha, seahorse, sole, stingray, tuna
5) frog, frog, newt, toad
6) flea, gnat, honeybee, housefly, ladybird, moth, termite, wasp
7) clam, crab, crayfish, lobster, octopus, scorpion, seawasp, slug, starfish, worm

Each classification group relates to a single output neuron
"""
from network import Maze
import random
import matplotlib.pyplot as plt
import time
import os

script_dir = os.path.dirname(__file__)
print(script_dir)
file_path = os.path.join(script_dir, "zoo_normalized_data.txt")
# prep data from txt file downloaded from web
with open(file_path, "r") as data_file:
    input_data = []
    output_data = []
    data = []
    temp = []
    for line in data_file:
        for item in line.split():
            temp.append(float(item))

        data.append([temp[:16], temp[16:]])
        temp = []

# Start time of run
start = time.time()

# create a network with 16 inputs, 10 hidden nodes, and 7 outputs.
net = Maze([16, 10, 7], ['relu', "lin"], lr=0.01)


error_data = []

# train on the data set imported above, run through all the data sets 100 times
print("training...")
for run in range(100):
    batch = random.sample(data, 70)
    for i in range(len(batch)):
        o, p, av = net.train([batch[i][0]], [batch[i][1]])

print("training complete\n")


# Display outcomes for some test cases after training is finished
print("Test random selection from data")
print("Target outcome: ", data[13][1])
print("Network outcome: ", net.forward([data[13][0]]))

print("\nTest random selection from data")
print("Target outcome: ", data[12][1])
print("Network outcome: ", net.forward([data[12][0]]))


print("\n\nScript ran for {} seconds.".format(time.time()-start))



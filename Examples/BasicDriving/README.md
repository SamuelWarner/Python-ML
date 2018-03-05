This environment was constructed using the Unity Game engine's machine learning tools. To use it you will need to download their
python api files from: https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Making-a-new-Unity-Environment.md

## Q-Learning with neural network

This example demos the use of the MazeX class in a Q-learning algorythm to steer a vehicle through a simple course without crashing.
For this problem I decided to not use a replay buffer as the given code solved the problem quickly, in as little as 33 episodes. 
Given that the inital weights were random mastery occured at varying episode counts, 33 being the quickest of 5 test runs. As can 
be seen in the "solved" video the vehicle navigates the course without incident. There is some oscillation in the steering control 
however the agent was rewarded only for surviving to the next frame and not for smooth steering input, thus the jerky driving 
style. The agent only recieved 5 distance sensor inputs and could choose to steer left, right, or striaght. The sensors were 
pointed as follows:

1. -45 degrees off forward direction
2. -22.5 degrees off forward direction
3.  Pointing directly forward
4.  +22.5 degrees off forward direction
5.  +45 degrees off forward direction

Distance was calculated in a 0 to 1 range from center of the vehicle to the max range of the sensor. 0 being nothing 
detected, and 0.5 being something detected halfway between the vehicle center and the max range of the sensor.

# Project 2 - Solving OpenAI Gym's lunar lander

## Module requirements
* python
* numpy
* pandas
* Box2D
* gym
* keras
* tensorflow

## How to run
From the command line run `python lunar-lander.py`. Scores will be periodically logged to the terminal. When the agent's simple moving average score for the previous 100 episodes crosses above 200, the successful model is saved to `model_success.h5` and the weights are saved to `weights_success.h5`

When the number of episodes reaches 3000 OR the agent's simple moving average score for the previous 100 episodes crosses above 200, the program terminates and saves the score, epsilon, and memory length for each episode to `scores_success.csv`.

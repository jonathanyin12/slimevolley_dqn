# Deep Q-Learning for Slime Volleyball


## Installation: 

## Basic Usage:
To train the deep q-network, run
```
python train_dqn.py
```
Training can take many hours before achieving decent intelligent gameplay.

To evaluate the agent, run
``` 
python test_model.py
```
This loads our best DQN agent and has it compete against the baseline agent from the Slime Volleyball Gym environment. Our agent is the yellow slime on the right.

By using the W, A, and D keys, or the Up, Left, and Right arrow keys, you can override the blue or yellow agents respectively. This allows for manual play against either agent.

## Results:
After training for 200 epochs, our model achieved an avg score of -3.03 against the baseline agent. Each game lasted on average 2760 time steps. Although the baseline agent is clearly better, we were impressed by the performance of our model nonetheless. It clearly plays intelligently and can beat the average human player.


## References:
The baseline agent is from [slimevolleygym](https://github.com/hardmaru/slimevolleygym), which is also the OpenAi Gym Environment we used to train our model. Our evaluation and test code is loosely modelled after existing test scripts in that repository.
# Deep Q-Learning for Slime Volleyball


https://user-images.githubusercontent.com/26175818/207496808-c5ef73e8-3663-4fe3-8c56-815f47134369.mov


## Installation: 
For installation, you will need to pip install the required packages
```
pip install -r requirements.txt
```

## Basic Usage:
To train the deep q-network, run
```
python train.py
```
Training can take many hours before achieving decent intelligent gameplay.

To evaluate the agent, run
``` 
python eval.py
```
This loads our best DQN agent and has it compete against the baseline agent from the Slime Volleyball Gym environment for 100 matches. It then generates match statistics. 

To visualize the agent's gameplay or manually play against it, run
``` 
python play.py
```
Our agent is the yellow slime on the right. By using the arrow keys, you can manually control the blue agent. 


## Results:
After training for 200 epochs, our model achieved an avg score of -3.03 against the baseline agent. Each game lasted on average 2760 time steps. Although the baseline agent is clearly better, we were impressed by the performance of our model nonetheless. It clearly plays intelligently and can beat the average human player.


## References:
The baseline agent is from [slimevolleygym](https://github.com/hardmaru/slimevolleygym), which is also the OpenAI Gym Environment we used to train our model.

"""
State mode (Optional Human vs Built-in AI)

FPS (no-render): 100000 steps /7.956 seconds. 12.5K/s.
"""

import math
import numpy as np
import gym
import slimevolleygym
import tensorflow as tf
from tensorflow import keras

np.set_printoptions(threshold=20, precision=3, suppress=True, linewidth=200)

def getAction(i):
    if i == 0:
        return [1, 0, 0]
    elif i == 1:
        return [1, 0, 1]
    elif i == 2:
        return [0, 1, 0]
    elif i == 3:
        return [0, 1, 1]
    elif i == 4:
        return [0, 0, 1]
    elif i == 5:
        return [0, 0, 0]


def policy(model, obs):
    current_state_tensor = tf.convert_to_tensor(obs)
    current_state_tensor = tf.expand_dims(current_state_tensor, 0)
    q_values = model(current_state_tensor, training=False)
    action_index = np.argmax(q_values)
    action = getAction(action_index)
    return action


if __name__ == "__main__":
    model = keras.models.load_model("model_epoch200.h5", compile=False)
    print(model.summary())
    base_policy = slimevolleygym.BaselinePolicy() # defaults to use RNN Baseline for player
    
    env = gym.make("SlimeVolley-v0")
    env.seed(np.random.randint(0, 10000))

   
    avg_score = 0
    avg_time = 0
   
    for i in range(100):
        done = False
        total_reward = 0
        t = 0
        score=0
        obs1 = env.reset()
        obs2 = obs1 # both sides always see the same initial observation.
        while not done:
            action2 = base_policy.predict(obs2)
            action1 = policy(model, obs1)
            obs1, reward, done, info = env.step(action1,action2) # extra argument
            obs2 = info['otherObs']
            score += reward
            t+=1
        
        avg_score += score
        avg_time+=t
        print(i)
    print("Avg score", avg_score/100, "Avg iterations", avg_time/100)

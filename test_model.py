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
from tensorflow.keras import layers, initializers

np.set_printoptions(threshold=20, precision=3, suppress=True, linewidth=200)


def custom_reward(obs):
    hit = dist = 0
    agent_x = obs[0]
    agent_y = obs[1]
    ball_x = obs[4]
    ball_y = obs[5]
    ball_x_vel = obs[6]
    ball_y_vel = obs[7]
    opp_x = obs[8]
    opp_y = obs[9]

    # reward for hitting the ball up
    if ball_x > 0 and ball_y_vel >=0:
        hit += 1/(abs(agent_x-ball_x) + abs(agent_y-ball_y))

    # reward for hitting ball forward
    if ball_x > 0 and ball_x_vel <=0:
        hit += 1/(abs(agent_x-ball_x) + abs(agent_y-ball_y))
    
    # # reward based on x dist
    if ball_x > 0:
        dist -=abs(agent_x-ball_x)
    if ball_x < 0:
        dist += abs(opp_x+ball_x)

    return hit+dist

def getAction(i):
    if i == 0:
        return [1,0,0]
    elif i == 1:
        return [1,0,1]
    elif i == 2:
        return [0,1,0]
    elif i == 3:
        return [0,1,1]
    elif i == 4:
        return [0,0,1]
    elif i == 5:
        return [0,0,0]

def policy(model, obs):
    current_state_tensor = tf.convert_to_tensor(obs)
    current_state_tensor = tf.expand_dims(current_state_tensor, 0)
    q_values = model(current_state_tensor, training=False)
    action_index = np.argmax(q_values)
    action = getAction(action_index)
    return action

if __name__=="__main__":
    

    from pyglet.window import key
    from time import sleep

    model = keras.models.load_model("model_epoch150_large.h5", compile=False)
    print(model.summary())
    base_policy = slimevolleygym.BaselinePolicy() # defaults to use RNN Baseline for player
    
    env = gym.make("SlimeVolley-v0")
    env.seed(np.random.randint(0, 10000))

    env.render()

   

    for i in range(1):
        obs1 = env.reset()
        obs2 = obs1 # both sides always see the same initial observation.

        done = False
        total_reward = 0
        while not done:

            action1 = policy(model, obs1)
            action2 = base_policy.predict(obs2)
            # print(custom_reward(obs1))
            obs1, reward, done, info = env.step(action1, action2) # extra argument
            obs2 = info['otherObs']

            total_reward += reward
            env.render()
            sleep(0.02) # 0.01

    env.close()
    print("cumulative score", total_reward)

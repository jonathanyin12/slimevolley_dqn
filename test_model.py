import numpy as np
import gym
import slimevolleygym
import tensorflow as tf
from tensorflow import keras

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
    if ball_x > 0 and ball_y_vel >= 0:
        hit += 1/(abs(agent_x-ball_x) + abs(agent_y-ball_y))

    # reward for hitting ball forward
    if ball_x > 0 and ball_x_vel <= 0:
        hit += 1/(abs(agent_x-ball_x) + abs(agent_y-ball_y))

    # # reward based on x dist
    if ball_x > 0:
        dist -= abs(agent_x-ball_x)
    if ball_x < 0:
        dist += abs(opp_x+ball_x)

    return hit+dist


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
    """
    Example of how to use Gym env, in single or multiplayer setting

    Humans can override controls:

    blue Agent:
    W - Jump
    A - Left
    D - Right

    Yellow Agent:
    Up Arrow, Left Arrow, Right Arrow
    """
    from pyglet.window import key
    from time import sleep

    manualAction = [0, 0, 0] # forward, backward, jump
    otherManualAction = [0, 0, 0]
    manualMode = False
    otherManualMode = False

    # taken from https://github.com/openai/gym/blob/master/gym/envs/box2d/car_racing.py
    def key_press(k, mod):
        global manualMode, manualAction, otherManualMode, otherManualAction
        if k == key.LEFT:  manualAction[0] = 1
        if k == key.RIGHT: manualAction[1] = 1
        if k == key.UP:    manualAction[2] = 1
        if (k == key.LEFT or k == key.RIGHT or k == key.UP): manualMode = True

        if k == key.D:     otherManualAction[0] = 1
        if k == key.A:     otherManualAction[1] = 1
        if k == key.W:     otherManualAction[2] = 1
        if (k == key.D or k == key.A or k == key.W): otherManualMode = True

    def key_release(k, mod):
        global manualMode, manualAction, otherManualMode, otherManualAction
        if k == key.LEFT:  manualAction[0] = 0
        if k == key.RIGHT: manualAction[1] = 0
        if k == key.UP:    manualAction[2] = 0
        if k == key.D:     otherManualAction[0] = 0
        if k == key.A:     otherManualAction[1] = 0
        if k == key.W:     otherManualAction[2] = 0


    model = keras.models.load_model("model_epoch30.h5", compile=False)
    print(model.summary())
    base_policy = slimevolleygym.BaselinePolicy()
    
    env = gym.make("SlimeVolley-v0")
    env.seed(np.random.randint(0, 10000))

    env.render()
    env.viewer.window.on_key_press = key_press
    env.viewer.window.on_key_release = key_release

   

    obs1 = env.reset()
    obs2 = obs1 # both sides always see the same initial observation.

    done = False
    total_reward = 0
    t = 0
    score=0
    while not done:
        if otherManualMode: # override with keyboard
            action2 = otherManualAction
        else:
            action2 = base_policy.predict(obs2)


        if manualMode:
            action1 = manualAction
        else:
            action1 = policy(model, obs1)


        obs1, reward, done, info = env.step(action1,action2) # extra argument
        obs2 = info['otherObs']

        # obs1, reward, done, info = env.step(action2) # extra argument
        # obs2=obs1
        score += reward
        total_reward += reward *10 + custom_reward(obs1)
        t+=1
        # print(total_reward/t)
        env.render()

    env.close()
    print(score, total_reward/t, t, total_reward)

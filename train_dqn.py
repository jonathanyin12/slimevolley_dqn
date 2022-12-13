import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, initializers

import random
from collections import deque
import gym
import slimevolleygym

gamma = 0.95          # decay rate of past observations
step_size = 1e-4       # step size

observe = 1000         # timesteps to observe before training
replay_memory = 50000      # number of previous transitions to remember
batch_size = 32           # size of each batch
num_actions = 6

def dq_model():
    state = layers.Input(shape=(12))
    hidden1 = layers.Dense(32, activation='relu', kernel_initializer=initializers.RandomNormal(stddev=0.1),
                           bias_initializer=initializers.Constant(0.0))(state)
    hidden2 = layers.Dense(32, activation='relu', kernel_initializer=initializers.RandomNormal(stddev=0.1),
                           bias_initializer=initializers.Constant(0.0))(hidden1)
    q_value = layers.Dense(6, activation='linear', kernel_initializer=initializers.RandomNormal(stddev=0.1),
                            bias_initializer=initializers.Constant(0.0))(hidden2)
    return keras.Model(inputs=state, outputs=q_value)

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

def train(model, batch, optimizer, loss_function):
    state_sample = np.array([d[0] for d in batch])
    action_sample = np.array([d[1] for d in batch])
    reward_sample = np.array([d[2] for d in batch])
    state_next_sample = np.array([d[3] for d in batch])
    terminal_sample = np.array([d[4] for d in batch])

    # compute the updated Q-values for the samples
    updated_q_value = []
    max_qs = np.max(model(state_next_sample, training = False), axis=1)
    for i in range(batch_size):
        if terminal_sample[i]:
            updated_q_value.append(reward_sample[i])
        else:
            updated_q_value.append(reward_sample[i] + gamma*max_qs[i])


    # train the model on the states and updated Q-values
    with tf.GradientTape() as tape:
        # compute the current Q-values for the samples
        current_q_value = []
        qvals =  model(state_sample, training=False)
        for i in range(batch_size):
            current_q_value.append(qvals[i][action_sample[i]])

        # compute the loss
        loss = loss_function(updated_q_value, current_q_value)

    # backpropagation
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss


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


def play_game(model, env, history_data, optimizer, loss_function, epsilon):

    obs = env.reset()
    done = False
    total_reward = 0
    score = 0
    t = 0
    loss = 0
    while not done:
        if epsilon > np.random.rand(1)[0]:
            action_index = np.random.choice(num_actions)
            # random action
        else:
            # compute the Q function
            current_state_tensor = tf.convert_to_tensor(obs)
            current_state_tensor = tf.expand_dims(current_state_tensor, 0)
            q_values = model(current_state_tensor, training=False)
            action_index = np.argmax(q_values)

        action = getAction(action_index)
        next_obs, reward, done, info = env.step(action)
       
        score += reward
        bonus = custom_reward(next_obs)
        reward = reward * 10 + bonus
        total_reward+=reward

        # store the observation
        history_data.append((obs, action_index, reward, next_obs, done))
        if len(history_data) > replay_memory:
            history_data.popleft() 

        if len(history_data) > observe:
            batch = random.sample(history_data, batch_size)
            loss += train(model, batch, optimizer, loss_function)
    
        t+=1
        obs = next_obs

    if len(history_data) > observe:
        print(total_reward, total_reward/t, np.array(q_values))
    return score, loss/t, t


def train_dqn(ckpt_path=None):
    try:
        # if you want to start from a checkpoint
        model = keras.models.load_model(ckpt_path, compile=False)
        print("Model loaded")
    except:
        model = dq_model()
        print("New model created")

    print(model.summary())

    epsilon = 0.2
    decay = 0.99
    min_epsilon = 0.001
    # specify the optimizer and loss function
    optimizer = keras.optimizers.Adam(learning_rate=step_size, clipnorm=1.0)
    loss_function = keras.losses.MeanSquaredError()
    history_data = deque()
    env = gym.make("SlimeVolley-v0")

    total_iterations = 0    

    for n in range(1000):
        score, mean_loss, iterations = play_game(model, env, history_data, optimizer, loss_function, epsilon)
        epsilon = max(min_epsilon, epsilon * decay)
        print(n, score, np.array(mean_loss), iterations)
        if n%10 == 0:
            print("Saving model_epoch{}.h5".format(n))
            model.save("model_epoch{}.h5".format(n), save_format='h5')


if __name__ == "__main__":
    train_dqn()

import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, initializers


class DQN:
    def __init__(self, checkpoint=None, gamma= 0.9, max_experiences= 100000, min_experiences= 10000, batch_size= 32, lr= 1e-4):
        self.model = self.load_model(checkpoint)
        print(self.model.summary())
        self.batch_size = batch_size
        self.gamma = gamma
        self.optimizer = keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0)
        self.loss_function = keras.losses.MeanSquaredError()
        self.history_data = deque()
        self.min_experiences = min_experiences
        self.max_experiences = max_experiences
        self.num_actions = 6

    def model(self):
        state = layers.Input(shape=(12))
        hidden1 = layers.Dense(32, activation='relu', kernel_initializer=initializers.RandomNormal(stddev=0.1),
                               bias_initializer=initializers.Constant(0.0))(state)
        hidden2 = layers.Dense(32, activation='relu', kernel_initializer=initializers.RandomNormal(stddev=0.1),
                               bias_initializer=initializers.Constant(0.0))(hidden1)
        q_value = layers.Dense(6, activation='linear', kernel_initializer=initializers.RandomNormal(stddev=0.1),
                               bias_initializer=initializers.Constant(0.0))(hidden2)
        return keras.Model(inputs=state, outputs=q_value)

    def load_model(self, checkpoint):
        try:
            model = keras.models.load_model(checkpoint, compile=False)
            print("Model loaded")
            return model
        except:
            model = self.model()
            print("New model created")
            return model



    def policy(self, obs, epsilon):
        # epsilon greedy
        q_values = None
        if epsilon > np.random.rand(1)[0]:
            # random action
            action_index = np.random.choice(self.num_actions)
        else:
            # compute the Q function
            current_state_tensor = tf.convert_to_tensor(obs)
            current_state_tensor = tf.expand_dims(current_state_tensor, 0)
            q_values = self.model(current_state_tensor, training=False)
            action_index = np.argmax(q_values)
        action = self.getAction(action_index)
        return action, action_index, q_values

    def getAction(self, i):
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

    def reward(self, obs):
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

    def train(self, batch):
        state_sample = np.array([d[0] for d in batch])
        action_sample = np.array([d[1] for d in batch])
        reward_sample = np.array([d[2] for d in batch])
        state_next_sample = np.array([d[3] for d in batch])
        terminal_sample = np.array([d[4] for d in batch])

        # compute the updated Q-values for the samples
        updated_q_value = []
        max_qs = np.max(self.model(state_next_sample, training=False), axis=1)
        for i in range(self.batch_size):
            if terminal_sample[i]:
                updated_q_value.append(reward_sample[i])
            else:
                updated_q_value.append(reward_sample[i] + self.gamma*max_qs[i])

        # train the model on the states and updated Q-values
        with tf.GradientTape() as tape:
            # compute the current Q-values for the samples
            current_q_value = []
            qvals = self.model(state_sample, training=False)
            for i in range(self.batch_size):
                current_q_value.append(qvals[i][action_sample[i]])

            # compute the loss
            loss = self.loss_function(updated_q_value, current_q_value)

        # backpropagation
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.model.trainable_variables))
        return loss

    def play_game(self, env, epsilon, eval = False):
        obs = env.reset()
        done = False
        total_reward = 0
        total_score = 0
        t = 0
        loss = 0
        while not done:
            action, action_index, q_values = self.policy(obs, epsilon)
            next_obs, score, done, info = env.step(action)

            total_score += score
            reward = score * 10 + self.reward(next_obs)
            total_reward += reward

            if not eval:
                # store the observation
                self.history_data.append(
                    (obs, action_index, reward, next_obs, done))
                if len(self.history_data) > self.max_experiences:
                    self.history_data.popleft()

                # train if done observing
                if len(self.history_data) > self.min_experiences:
                    batch = random.sample(self.history_data, self.batch_size)
                    loss += self.train(batch)

            t += 1
            obs = next_obs

        return np.array(loss/t), total_score, total_reward, t, np.array(q_values)

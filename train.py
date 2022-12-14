from slime_dqn import DQN
import gym
import slimevolleygym

def train_dqn(ckpt_path=None):
    gamma = 0.9
    lr = 1e-4
    observe = 10000
    replay_memory = 100000
    batch_size = 32
    agent = DQN(ckpt_path, gamma, replay_memory, observe, batch_size, lr)


    epsilon = 0.2
    decay = 0.99
    min_epsilon = 0.001
    env = gym.make("SlimeVolley-v0")

    for n in range(1000):
        print("EPOCH", n)
        mean_loss, score, total_reward, t, q_vals = agent.play_game(env, epsilon)
        epsilon = max(min_epsilon, epsilon * decay)
        print("Score:", score, "Loss:", mean_loss)
        print("Total reward:", total_reward, "Average reward:", total_reward/t)
        print("Q-vals:", q_vals)
        if n % 10 == 0:
            print("Saving model_epoch{}.h5".format(n))
            agent.model.save("model_epoch{}.h5".format(n), save_format='h5')
        print('')


if __name__ == "__main__":
    train_dqn()

import gym
import slimevolleygym
from slime_dqn import DQN

def eval_dqn(ckpt_path=None):
    agent = DQN(ckpt_path)
    env = gym.make("SlimeVolley-v0")
    avg_score = 0
    avg_time = 0
    avg_total_reward = 0
    games = 100
    for i in range(games):
        mean_loss, score, total_reward, t, q_vals = agent.play_game(env, 0, True)
        avg_score += score
        avg_total_reward += total_reward
        avg_time+=t
        print(i, " Score:", score, "Reward:", total_reward, "Length:", t)

    print("===================================================")
    print("Avg score", avg_score/games, "Avg reward", avg_total_reward,  "Avg game length", avg_time/games)

if __name__ == "__main__":
    eval_dqn("model_epoch60.h5")

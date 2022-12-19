import gym
import slimevolleygym
from slime_dqn import DQN

def eval_dqn(ckpt_path=None): #evaluates agent against baseline
    agent = DQN(ckpt_path)
    env = gym.make("SlimeVolley-v0")
    avg_score = 0
    avg_time = 0
    avg_total_reward = 0
    games = 100 #runs for 100 games
    for i in range(games):
        mean_loss, score, total_reward, t, q_vals = agent.play_game(env, 0, True)
        avg_score += score #calculates total score
        avg_total_reward += total_reward #calculates total reward
        avg_time+=t #calculates total time spent
        print(i, " Score:", score, "Reward:", total_reward, "Length:", t)

    print("===================================================")
    print("Avg score", avg_score/games, "Avg reward", avg_total_reward/games,  "Avg game length", avg_time/games) #prints average score, average reward, and average time of game. 

if __name__ == "__main__":
    eval_dqn("model.h5")

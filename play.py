import gym
import slimevolleygym
from pyglet.window import key
from time import sleep
from slime_dqn import DQN

def play(ckpt_path=None):
    global manualAction
    manualAction = [0, 0, 0] # values stored for forward, backward, jump actions, respectively. 
    global manualMode
    manualMode = False

    # taken from https://github.com/openai/gym/blob/master/gym/envs/box2d/car_racing.py
    def key_press(k, mod): #detects key press
        global manualMode, manualAction
        if k == key.RIGHT:  manualAction[0] = 1
        if k == key.LEFT: manualAction[1] = 1
        if k == key.UP:    manualAction[2] = 1
        if (k == key.LEFT or k == key.RIGHT or k == key.UP): manualMode = True

    def key_release(k, mod): #detects when keys are no longer pressed
        global manualMode, manualAction
        if k == key.RIGHT:  manualAction[0] = 0
        if k == key.LEFT: manualAction[1] = 0
        if k == key.UP:    manualAction[2] = 0

    agent = DQN(ckpt_path) #import our agent to play against
    
    env = gym.make("SlimeVolley-v0")

    env.render()
    env.viewer.window.on_key_press = key_press
    env.viewer.window.on_key_release = key_release

    obs = env.reset()
    done = False
    score=0
    while not done:
        if manualMode: # override with keyboard; allows for manual play
            action2 = manualAction
            action1, _, _ = agent.policy(obs, epsilon=0)
            obs, reward, done, info = env.step(action1, action2) # extra argument
        else: #otherwise play against baseline agent
            action1, _, _ = agent.policy(obs, epsilon=0)
            obs, reward, done, info = env.step(action1) # extra argument
            
        score += reward #calculates total score
        env.render()
        if manualMode:
            sleep(0.02)
    env.close()
    print(score)


if __name__ == "__main__":
    play("model.h5")

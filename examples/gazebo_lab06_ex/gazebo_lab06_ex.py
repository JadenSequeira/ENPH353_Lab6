#!/usr/bin/env python3

## @package examples
#  Main
#
#  The following script execute q-learning for robot in the ROS environment for line following


import gym
from gym import wrappers
import gym_gazebo
import time
import numpy
import random
import time
import qlearn
import liveplot
from matplotlib import pyplot as plt

## The render funciton renders the environment depending on the epsiode
#  It is an optional implementation in the q-learning model
def render():
    render_skip = 0  # Skip first X episodes.
    render_interval = 50  # Show render Every Y episodes.
    render_episodes = 10  # Show Z episodes every rendering.

    if (x % render_interval == 0) and (x != 0) and (x > render_skip):
        env.render()
    elif (((x-render_episodes) % render_interval == 0) and (x != 0) and
          (x > render_skip) and (render_episodes < x)):
        env.render(close=True)


## The main function executes the main q-learning algorithm through the use of the qlearn class
if __name__ == '__main__':

    # Setup environment with the reward system and liveplot
    env = gym.make('Gazebo_Lab06-v0')
    outdir = '/tmp/gazebo_gym_experiments'
    env = gym.wrappers.Monitor(env, outdir, force=True)
    plotter = liveplot.LivePlot(outdir)

    last_time_steps = numpy.ndarray(0)

    # Initialize the qlearning model. The alpha value os the learning rate, the gamma value is the consideration of future rewards,
    # and the epsilon is the exploration vs exploitation setting
    qlearn = qlearn.QLearn(actions=range(env.action_space.n),
                           alpha=0.5, gamma=0.5, epsilon=0.9)

    # Uncomment the below line when there is a Qvalue file that can be loaded

    #qlearn.loadQ("QValues_A+")

    # Initialize the epsilon value and its discounting each cycle (lower discount value the more exploitation in subsequent episodes)
    initial_epsilon = qlearn.epsilon
    epsilon_discount = 0.9986

    # Initialization of settings
    start_time = time.time()
    total_episodes = 10000
    highest_reward = 0

    # A new episode is started each time the camera loses sight of the track for more than 30 frames
    for x in range(total_episodes):
        done = False

        cumulated_reward = 0  # Should going forward give more reward then L/R?  - Depends on the track and tuning parameters of speed and turning. Also depends on the state space

        observation = env.reset()

        if qlearn.epsilon > 0.05:
            qlearn.epsilon *= epsilon_discount

       # render() #defined above, not env.render()

        state = ''.join(map(str, observation))

        # Main Q-learning execution
        i = -1
        while True:
            i += 1

            # Pick an action based on the current state
            action = qlearn.chooseAction(state)
            # Execute the action and get feedback
            observation, reward, done, info = env.step(action)
            cumulated_reward += reward

            # Saves the highest policy
            if highest_reward < cumulated_reward:
                highest_reward = cumulated_reward
                qlearn.saveQ("QValues_A+")

            nextState = ''.join(map(str, observation))

            # Update Q-Values
            qlearn.learn(state, action, reward, nextState)

            env._flush(force=True)

            if not(done):
                state = nextState
            else:
                last_time_steps = numpy.append(last_time_steps, [int(i + 1)])
                break

        print("===== Completed episode {}".format(x))

        # Plot every 5 episodes and save the Q-values
        if (x > 0) and (x % 5 == 0):
            qlearn.saveQ("QValues")
            plotter.plot(env)

        # Display the Q-learning settings after each episode, for the next episode
        m, s = divmod(int(time.time() - start_time), 60)
        h, m = divmod(m, 60)
        print ("Starting EP: " + str(x+1) +
               " - [alpha: " + str(round(qlearn.alpha, 2)) +
               " - gamma: " + str(round(qlearn.gamma, 2)) +
               " - epsilon: " + str(round(qlearn.epsilon, 2)) +
               "] - Reward: " + str(cumulated_reward) +
               "     Time: %d:%02d:%02d" % (h, m, s))

    # Github table content
    print ("\n|"+str(total_episodes)+"|"+str(qlearn.alpha)+"|" +
           str(qlearn.gamma)+"|"+str(initial_epsilon)+"*" +
           str(epsilon_discount)+"|"+str(highest_reward) + "| PICTURE |")

    l = last_time_steps.tolist()
    l.sort()

    # print("Parameters: a="+str)
    print("Overall score: {:0.2f}".format(last_time_steps.mean()))
    print("Best 100 score: {:0.2f}".
          format(reduce(lambda x, y: x + y, l[-100:]) / len(l[-100:])))

    env.close()

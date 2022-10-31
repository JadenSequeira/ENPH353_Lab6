#!/usr/bin/env python3

## @package examples
#  LivePlot Class
#
#  The following script enables live plotting of the rewards against episodes to display learning

import matplotlib
import matplotlib.pyplot as plt
import gym

rewards_key = 'episode_rewards'


## LivePlot class is used to create a live plot for rewards vs episodes
#
#  This plot displays the learning in realt after a set amount of episodes
class LivePlot(object):


    
    ## Liveplot renders a graph of either episode_rewards or episode_lengths
    #  @param outdir (outdir): Monitor output file location used to populate the graph
    #  @param data_key (Optional[str]): The key in the json to graph (episode_rewards or episode_lengths).
    #  @param line_color (Optional[dict]): Color of the plot.
    def __init__(self, outdir, data_key=rewards_key, line_color='blue'):
        self.outdir = outdir
        self.data_key = data_key
        self.line_color = line_color

        #styling options
        matplotlib.rcParams['toolbar'] = 'None'
        plt.style.use('ggplot')
        plt.xlabel("Episodes")
        plt.ylabel(data_key)
        fig = plt.gcf().canvas.set_window_title('simulation_graph')

    ## The plot function is used to plot the data depending on the type of graph
    #  @param env env is the environment containing the required data for graphing
    def plot(self, env):
        if self.data_key is rewards_key:
            data = gym.wrappers.Monitor.get_episode_rewards(env)
        else:
            data = gym.wrappers.Monitor.get_episode_lengths(env)

        plt.plot(data, color=self.line_color)

        # pause so matplotlib will display
        # may want to figure out matplotlib animation or use a different library in the future
        plt.pause(0.000001)

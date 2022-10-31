## @package examples
#  Qlearn algorithm Class
#
#  The following script implements the Qlearn class. The class is used
#  to choose an action and to also fill in the Qtable. Finally the class can be used
#  to store the Q table as a pickle file, or load a previous Q file

import random
import pickle
import csv


## QLearn Class provides functionality for Q Reinforcement Learning
#
#  The class enables chosing an action based on epsilon, filling in the Q values,
#  saving the Q values, and loading the Q values 
class QLearn:

    ## Initialization function for Qlearn object
    #  @param actions the actions that can be taken
    #  @param epsilon the exploration-exploitation control value for choosing the next action
    #  @param alpha the learning rate
    #  @param gamma the the discounting value of future rewards
    def __init__(self, actions, epsilon, alpha, gamma):
        self.q = {}
        self.epsilon = epsilon  # exploration constant
        self.alpha = alpha      # discount constant
        self.gamma = gamma      # discount factor
        self.actions = actions


    
    ## loadQ function loads the Q-values from a pickle file
    #  @param filename the name of the file to load the Q-values from
    def loadQ(self, filename):

        with open(filename, "rb") as f:
            self.q = pickle.load(f)

        print("Loaded file: {}".format(filename+".pickle"))


    ## Save the Q state-action values in a pickle file.
    #  @param filename the name of the file to save the Q-values to
    def saveQ(self, filename):

        with open(filename, "wb") as f:
            pickle.dump(self.q, f)
        
        # Saving to a CSV file for visual inspection, note that the filename should be modified
        # to ensure we are not overwriting pickle file to a CSV file. This can be done by also changing
        # the functions signature by adding another parameter


        #with open(filename, 'w') as f:
        #    writer = csv.writer(f)
        #    writer.writerow(self.q)

        f.close()

        print("Wrote to file: {}".format(filename+".pickle"))


    ## getQ function returns the state, action Q value or 0.0 if the value is missing
    #  @param state the state
    #  @param action the action
    def getQ(self, state, action):
        return self.q.get((state, action), 0.0)


    ## chooseAction implements exploration vs exploitation based on the epsilon setting. Therefore,
    #  it returns a random action epsilon % of the time or the action associated with the largest Q value in (1-epsilon)% of the time.
    #  Accounts for the case where two actions have the same q-values
    #  @param state the state in which the robot is in
    #  @param return_q (default = False) can be used to return the q value associated with the action chosen
    def chooseAction(self, state, return_q=False):
        q = [self.getQ(state, a) for a in self.actions]
        maxQ = max(q)

        if random.random() < self.epsilon:

            # Alternate algorithm for chosing an action
            #rand_val = random.random()
            #num_actions = len(self.actions)
            #action = sum([1 if rand_val > (float(i) / num_actinos) else 0 for i in range(num_actions)]) - 1
            #print("Num actions: {} | Random value: {} | Random action: {}".format(num_actinos, rand_val, action))
            #return action

            # Faster leraning through the below four lines as they randomize based while centering on each Q-value.
            # This ensures that actions choices are not too distant/volatile.
            minQ = min(q)
            mag = max(abs(minQ), abs(maxQ))
            q = [q[i] + random.random() * mag - .5*mag for i in range(len(self.actions))]
            maxQ = max(q)
        
        ## Accounts for the possibility of having two of the same max Q values
        count = q.count(maxQ)
        if count > 1:
            best = [i for i in range(len(self.actions)) if q[i] == maxQ]
            i = random.choice(best)
        else:
            i = q.index(maxQ)

        action = self.actions[i]
        if return_q:
            return action, q
        return action


    ## learn updates the Q(state,value) dictionary using the bellman update equation
    #  Q(s1, a1) += alpha * [reward(s1,a1) + gamma* max(Q(s2)) - Q(s1,a1)]
    #  Find max(Q) for state2
    #  Update Q for (state1, action1) (use discount factor gamma for future rewards)
    #  @param state1 the current state
    #  @param action1 the action being taken
    #  @param reward the reward for taking the action
    #  @param state2 the subsequent state
    def learn(self, state1, action1, reward, state2):
        maxqnew = max([self.getQ(state2, a) for a in self.actions])
        self.learnQ( state1, action1, reward, reward + self.gamma*maxqnew)


    ## learnQ updates Q-value based on provided parameters
    #  Address edge case if the [state, action] is not in our dictionary?
    #  @param state1 the current state
    #  @param action1 the action being taken
    #  @param reward the reward for taking the action
    #  @param value the future rewards multiplied by the discount factor
    def learnQ(self, state1, action1, reward, value):
        oldv = self.q.get((state1, action1), None)
        if oldv is None:
            self.q[(state1,action1)] = reward
        else:
            self.q[(state1,action1)] = oldv + self.alpha * (value - oldv)

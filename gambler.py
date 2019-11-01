from enums import *
import random

class Gambler:

    def __init__(self,learning_rate=0.1, discount=0.95,exploration_rate=1.0,iterations=1000):
        self.q_table=[[0,0,0,0,0],[0,0,0,0,0]]
        self.learning_rate=learning_rate
        self.discount=discount
        self.exploration_rate=exploration_rate
        self.iterations=iterations
        self.exploration_delta= 1.0 /self.iterations

    def get_next_action(self,state):
        if random.random() > self.exploration_rate:
            return self.greedyaction(state)
        else:
            return self.random_action(state)

    def greedyaction(self,state):
        if(self.q_table[FORWARD][state] > self.q_table[BACKWARD][state]):
            return FORWARD
        elif (self.q_table[FORWARD][state] < self.q_table[BACKWARD][state]):
            return BACKWARD
        else:
            return FORWARD if random.random()<0.5 else BACKWARD
    

    def random_action(self,state):
        return FORWARD if random.random()<0.5 else BACKWARD

    def update(self,old_state,new_state,action,reward):
        old_value=self.q_table[action][old_state]

        future_action = self.greedyaction(new_state)

        future_value=self.q_table[future_action][new_state]

        new_value= old_value + self.learning_rate * (reward + self.discount * future_value -old_value)
        self.q_table[action][old_state]=new_value

        if self.exploration_rate > 0:
            self.exploration_delta -= self.exploration_delta
        
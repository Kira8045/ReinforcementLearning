from enums import *
import random

class Accountant:
    def __init__(self):
        self.q_table=[[0]*5]*2

    def get_next_action(self,state):
        
        if(self.q_table[FORWARD][state] > self.q_table[BACKWARD][state]):
            return FORWARD
        elif (self.q_table[FORWARD][state] < self.q_table[BACKWARD][state]):
            return BACKWARD
        else:
            return FORWARD if random.random()<0.5 else BACKWARD
    
    def update(self,old_state,new_state,action,reward):
        self.q_table[action][old_state] +=reward

    
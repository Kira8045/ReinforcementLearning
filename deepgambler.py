from enums import *
import random
import tensorflow as tf
import numpy as np

class DeepGambler:
    def __init__(self,learning_rate,discount=0.95,exploration_rate=1.0,iterations=1000):
        self.learning_rate= learning_rate
        self.discount= discount
        self.exploration_rate=exploration_rate
        self.exploration_delta= 1/iterations
        self.input_count=5
        self.output=2
        self.iterations=iterations

        self.session = tf.Session()

        self.define_model()

        self.session.run(self.initializer)

    def define_model(self):
        self.model_input = tf.placeholder(dtype=tf.float32, shape=[None,self.input_count])
        fc1=tf.layers.dense(self.model_input,16,activation= tf.sigmoid , kernel_initializer = tf.constant_initializer(np.zeros((self.input_count,16))))
        fc2=tf.layers.dense(fc1,16,activation=tf.sigmoid, kernel_initializer= tf.constant_initializer(np.zeros((16,self.output))))

        self.model_output = tf.layers.dense(fc2, self.output)

        self.target_output = tf.placeholder(shape=[None, self.output],dtype=tf.float32)

        loss= tf.losses.mean_squared_error(self.target_output, self.model_output)

        self.optimizer=tf.train.GradientDescentOptimizer(learning_rate= self.learning_rate).minimize(loss)

        self.initializer= tf.global_variables_initializer()


    def get_Q(self,state):
        return self.session.run(self.model_output,feed_dict= {self.model_input: self.to_one_hot(state)})[0]

    def to_one_hot(self,state):
        one_hot=np.zeros((1,5))
        one_hot[0][state]=1
        return one_hot
    
    def get_next_action(self,state):
        if random.random() > self.exploration_rate:
            return self.greedyaction(state)
        else:
            return self.random_action(state)

    def greedyaction(self,state):
        return np.argmax(self. get_Q(state))

    def random_action(self,state):
        return FORWARD if random.random()<0.5 else BACKWARD

    def train(self,old_state,new_state,action,reward):

        old_state_qvalues = self.get_Q(old_state)

        new_state_qvalues = self.get_Q(new_state)

        old_state_qvalues[action]= reward + self.discount * np.amax(new_state_qvalues)

        train_input = self.to_one_hot(old_state)

        train_output = [old_state_qvalues]

        training_data ={self.model_input: train_input, self.target_output: train_output} 

        self.session.run(self.optimizer, feed_dict = training_data)

    def update(self,old_state,new_state,action,reward):
        self.train(old_state,new_state, action,reward)
        if self.exploration_rate >0:
            self.exploration_rate -= self.exploration_delta
    
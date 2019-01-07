#coding:utf-8
# -----------------------------
"""
2018-5-26
selective DQN
Branch1 	choose canditate samples ,which means choosed samples be joined in choose set
three full layers & relu
"""
# -----------------------------

import tensorflow as tf 
import numpy as np 
import random
import time
import os
from collections import deque 
from six.moves import xrange
import makedata
layer1_num = 1024
layer2_num = 512
layer3_num = 256
layer4_num = 32
class DQN_Net():	
	def __init__(self,sess,args):
		#init replay memory
		self.replayMemory = deque()
		self.sess = sess
		self.state_size = args.state_size
		self.CANDIDATE_NUM = args.candidate_num
		self.ACTION_NUM = args.candidate_num
		self.stateInput = tf.placeholder(tf.float32, [None,self.state_size])
		#self.SELECT_NUM = args.select_num
		self.yInput = tf.placeholder(tf.float32, [None])
		self.actionInput = tf.placeholder(tf.int32, [None,self.ACTION_NUM], name='a')
		self.timeStep = 1
		self.train_flag = False
		self.Dataset_name_source = args.Dataset_name_source
		self.Dataset_name_target = args.Dataset_name_target

		self.createQNetwork()
		#t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
		#e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')

		#with tf.variable_scope('soft_replacement'):
		#	self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]
			#self.target_replace_op_choose = [tf.assign(t, e) for t, e in zip(t_params_choose, e_params_choose)]
		self.epsilon = args.INITIAL_EPSILON
		self.Q_optim = tf.train.AdamOptimizer(args.lr_Q, beta1=args.beta2) \
					.minimize(self.loss)

		
	def createQNetwork(self):
			# ------------------ build evaluate_net ------------------
		with tf.variable_scope('eval_net'):
			layer1 = self.fc_layer(self.stateInput, layer1_num, "layer1")
			relu1 = tf.nn.relu(layer1)
			layer2 = self.fc_layer(relu1, layer2_num, "layer2")
			relu2 = tf.nn.relu(layer2)
			layer3 = self.fc_layer(relu2, layer3_num, "layer3")
			relu3 = tf.nn.relu(layer3) 
			self.QValue = self.fc_layer(relu3, self.ACTION_NUM, "layer4")


			# ------------------ build target_net ------------------
		'''with tf.variable_scope('target_net'):
			target_layer1 = self.fc_layer(self.stateInput, layer1_num, "target_layer1")
			target_relu1 = tf.nn.relu(target_layer1)
			target_layer2 = self.fc_layer(target_relu1, layer2_num, "target_layer2")
			target_relu2 = tf.nn.relu(target_layer2)
			target_layer3 = self.fc_layer(target_relu2, layer3_num, "target_layer3")
			target_relu3 = tf.nn.relu(target_layer3) 
			self.q_next = self.fc_layer(target_relu3, self.ACTION_NUM, "target_layer4")'''

			# ------------------ calu target_net ------------------

		Q_action = tf.reduce_sum(tf.multiply(self.QValue, tf.cast(self.actionInput,tf.float32)),reduction_indices = 1)
		self.loss = tf.reduce_mean(tf.square(self.yInput-Q_action))
		self.Q_loss_sum = tf.summary.scalar("Q_loss", self.loss)

		self.saver = tf.train.Saver()

	def getAction(self,args,selected_num):
		#print("selected_num,curentState.shape:",selected_num,self.currentState.shape)
		Flag = False
		print("self.state_size:",self.state_size)
		QValue_action = self.QValue.eval(session = self.sess,feed_dict= {
									self.stateInput: self.currentState.reshape(1,self.state_size)})
		action = np.zeros(self.ACTION_NUM)
		if np.random.random() <= self.epsilon:
			action_index = random.randrange(self.CANDIDATE_NUM-selected_num)
			action[action_index] = 1
			Flag = True
			#print("random take action:", action_index)
		else:
			action_index = np.argmax(QValue_action[0,0:(self.CANDIDATE_NUM-selected_num)])
			action[action_index] = 1
			#print("take action:",action_index)

		if self.epsilon > args.FINAL_EPSILON and self.timeStep > args.OBSERVE:
			self.epsilon -= (args.INITIAL_EPSILON - args.FINAL_EPSILON)/args.EXPLORE
		return action,action_index,Flag

	def setPerception(self,args,action,reward,next_state,selected_num_nextstate,terminal,i,writer):
		self.replayMemory.append((self.currentState,action,reward,next_state,selected_num_nextstate,terminal))

		state = ""
		if self.timeStep <= args.OBSERVE:
			state = "observe"
		elif self.timeStep > args.OBSERVE and self.timeStep <= args.OBSERVE + args.EXPLORE:
			self.train_flag = True
			state = "explore"
		else:
			state = "train"
		if len(self.replayMemory) > args.REPLAY_MEMORY:
			self.replayMemory.popleft()
		if self.timeStep > args.OBSERVE:
			# Train the network
			'''if self.timeStep % args.replace_target_iter == 1:
				print('target network update')
				self.copyTargetQNetwork()'''
				#w_target = tf.get_default_graph().get_tensor_by_name("target_net/target_layer1W:0")
				#w_eval = tf.get_default_graph().get_tensor_by_name("eval_net/layer1W:0")
				#print("w_eval:",self.sess.run(w_eval))
				#print("w_target:",self.sess.run(w_target))
			print("DQN TIMESTEP", self.timeStep, "/ STATE", state, "/ EPSILON", self.epsilon)
			if terminal == 1:
				self.trainQNetwork(args,i,writer)

		#if self.timeStep % args.save_iter_dqn == 1:
	    #	print('dqn model save')
		#	self.save(args.checkpoint_dir_dqn, self.timeStep)
# print info
		self.currentState = next_state
		self.timeStep += 1


	def trainQNetwork(self,args,i,writer_dqn):
		start_time = time.time()
		self.writer = writer_dqn
		#self.writer = tf.summary.FileWriter("./logs-dqn", self.sess.graph)
		# Step 1: obtain random minibatch from replay memory
		minibatch = random.sample(self.replayMemory,args.sample_dqn)
		state_batch = [data[0] for data in minibatch]
		action_batch = [data[1] for data in minibatch]
		reward_batch = [data[2] for data in minibatch]
		nextState_batch = [data[3] for data in minibatch]
		selected_num_nextstate = [data[4] for data in minibatch]
		#print(len(nextState_batch))
		# Step 2: calculate y 
		y_batch = []
		QValue_batch = self.QValue.eval(session = self.sess,feed_dict={self.stateInput:np.array(nextState_batch).reshape(args.sample_dqn,-1)})
		for j in xrange(0,args.sample_dqn):
			terminal = minibatch[j][5]
			if terminal:
				y_batch.append(reward_batch[j])
			else:
				y_batch.append(reward_batch[j] + args.GAMMA * np.argmax(QValue_batch[j][0:self.CANDIDATE_NUM-selected_num_nextstate[j]]))
		
		_, summary_str = self.sess.run([self.Q_optim, self.Q_loss_sum],feed_dict={self.yInput : y_batch,
							self.actionInput : action_batch,self.stateInput : np.array(state_batch).reshape(args.sample_dqn,-1)})
		self.writer.add_summary(summary_str, self.timeStep)
		Q_loss = self.loss.eval(session = self.sess, feed_dict={self.yInput : y_batch,self.actionInput : action_batch,self.stateInput : np.array(state_batch).reshape(args.sample_dqn,-1)})

		#print('Q_loss.shape:',Q_loss.shape)
		#for j in xrange(0,Q_loss.shape[0]):
		print("Episode-dqn: [%2d] time: %4.4f, Q_loss: %.8f" % (self.timeStep, time.time() - start_time, Q_loss))
		
		# save network every 100000 iteration

	#def copyTargetQNetwork(self):
	#	self.sess.run(self.target_replace_op)


	def decay_epsilon(self):
		if self.epsilon > 0.02:
			self.epsilon = self.epsilon - 0.02
			print('epsilon=',self.epsilon)

	def setState(self,state):
		#p = np.array([[p]])
		state = state.reshape(1,-1)
		self.currentState = state

	def setNextState(self,state):
		self.currentState = state

	def xavier_init(self,size):
		in_dim = size[0]
		xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
		return tf.random_normal(shape=size, stddev=xavier_stddev)

	def fc_layer(self, bottom, n_weight, name):
		shape = bottom.get_shape().as_list()
		dim = 1
		for d in shape[1:]:
			dim *= d
		bottom = tf.reshape(bottom, [-1, dim])
		assert len(bottom.get_shape()) == 2
		n_prev_weight = bottom.get_shape()[1]
		initer = tf.contrib.layers.xavier_initializer()
		W = tf.get_variable(name+'W', dtype=tf.float32, shape=[n_prev_weight, n_weight], initializer=initer)
		b = tf.get_variable(name+'b', dtype=tf.float32, initializer=tf.constant(0.01, shape=[n_weight], dtype=tf.float32))
		print("W",W)
		fc = tf.nn.bias_add(tf.matmul(bottom, W), b)
		return fc	

	def save(self,checkpoint_dir,step):
		model_name = "DQN.model"
		model_dir = "%s_%s_%s" % (self.Dataset_name_source,self.Dataset_name_target, self.state_size)
		checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
		
		if not os.path.exists(checkpoint_dir):
			os.makedirs(checkpoint_dir)
			
		self.saver.save(self.sess,
						os.path.join(checkpoint_dir, model_name),
						global_step=step)
	
	def load(self, checkpoint_dir):
		print("Reading checkpoint...")
		
		model_dir = "%s_%s_%s" % (self.Dataset_name_source,self.Dataset_name_target,self.state_size)
		checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
		
		ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
		if ckpt and ckpt.model_checkpoint_path:
			ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
			self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
			return True
		else:
			return False
#coding:utf-8
"""
Resnet 输出维度为128，subset图片特征拼接起来，共128*10
目标域128*100
Generator 
Discriminator
condition :y = tf.placeholder(tf.float32, shape=[None, y_dim])噪声
"""
import tensorflow as tf
import math
import numpy as np
import tensorflow.contrib.slim as slim
import os
from six.moves import xrange
def maxPoolLayer(x, kHeight, kWidth, strideX, strideY, name, padding = "SAME"):
	"""max-pooling"""
	return tf.nn.max_pool(x, ksize = [1, kHeight, kWidth, 1],
                          strides = [1, strideX, strideY, 1], padding = padding, name = name)

def dropout(x, keepPro, name = None):
	"""dropout"""
	return tf.nn.dropout(x, keepPro, name)

def LRN(x, R, alpha, beta, name = None, bias = 1.0):
	"""LRN"""
	return tf.nn.local_response_normalization(x, depth_radius = R, alpha = alpha,
                                              beta = beta, bias = bias, name = name)

def fcLayer(x, inputD, outputD, reluFlag, name,reuse):
	"""fully-connect"""
	with tf.variable_scope(name,reuse=reuse) as scope:
		w = tf.get_variable("w", shape = [inputD, outputD], dtype = "float")
		b = tf.get_variable("b", [outputD], dtype = "float")
		out = tf.nn.xw_plus_b(x, w, b, name = scope.name)
		if reluFlag:
			return tf.nn.relu(out)
		else:
			return out

def convLayer(x, kHeight, kWidth, strideX, strideY,
				featureNum, name, reuse,padding = "SAME", groups = 1):
	"""convolution"""
	channel = int(x.get_shape()[-1])
	conv = lambda a, b: tf.nn.conv2d(a, b, strides = [1, strideY, strideX, 1], padding = padding)
	with tf.variable_scope(name,reuse=reuse) as scope:
		w = tf.get_variable("w", shape = [kHeight, kWidth, channel/groups, featureNum])
		b = tf.get_variable("b", shape = [featureNum])

		xNew = tf.split(value = x, num_or_size_splits = groups, axis = 3)
		wNew = tf.split(value = w, num_or_size_splits = groups, axis = 3)

		featureMap = [conv(t1, t2) for t1, t2 in zip(xNew, wNew)]
		mergeFeatureMap = tf.concat(axis = 3, values = featureMap)
		# print mergeFeatureMap.shape
		out = tf.nn.bias_add(mergeFeatureMap, b)
		#print ("--------------------")
		l=mergeFeatureMap.get_shape().as_list()
		l[0]=-1
	return tf.nn.relu(tf.reshape(out, l), name = scope.name)
def LeakyRelu(x, leak=0.2, name="LeakyRelu"):
     with tf.variable_scope(name):
         f1 = 0.5 * (1 + leak)
         f2 = 0.5 * (1 - leak)
         return f1 * x + f2 * tf.abs(x)
class GAN_Net(object):
	def __init__(self,sess,args,h1_dim=1024,h2_dim=1024,h3_dim=32,h4_dim=256,h5_dim=128,h6_dim=64):
		self.source_images = tf.placeholder(tf.float32, [None, 227,227,3])
		self.target_images = tf.placeholder(tf.float32, [None, 227,227,3])
		self.KEEPPRO = args.keepPro
		self.CLASSNUM = 1000
		self.SKIP = args.skip
		self.MODELPATH = args.alexnet_model_path

		self.feature_dim = args.feature_dim
		self.h1_dim = h1_dim
		self.h2_dim = h2_dim
		self.category_num = args.category_num
		self.batch_size = args.batch_size
		#self.X_feature_S = tf.placeholder(tf.float32, [None, self.feature_dim])
		self.y_label_S = tf.placeholder(tf.int32,[None])
		self.label_tg = tf.placeholder(tf.int32,[None,args.category_num+1])
		self.label_td = tf.placeholder(tf.int32,[None,args.category_num+1])
		self.label_sg = tf.placeholder(tf.int32,[None,args.category_num+1])
		self.label_sd = tf.placeholder(tf.int32,[None,args.category_num+1])
		self.lambda_T = args.LambdaT
		self.BetaGD = args.BetaGD
		
		self.Dataset_name_source = args.Dataset_name_source
		self.Dataset_name_target = args.Dataset_name_target
		self.sess = sess
		self.build_model()
	def build_model(self):
		""" classier Net model """
		self.C_W1 = tf.Variable(self.xavier_init([self.feature_dim, self.category_num]))
		self.C_b1 = tf.Variable(tf.zeros(shape=[self.category_num]))
		self.theta_C = [self.C_W1,self.C_b1]

		self.D_W1 = tf.Variable(self.xavier_init([self.feature_dim, self.h1_dim]))
		self.D_b1 = tf.Variable(tf.zeros(shape=[self.h1_dim]))		
		self.D_W2 = tf.Variable(self.xavier_init([self.h1_dim, self.h2_dim]))
		self.D_b2 = tf.Variable(tf.zeros(shape=[self.h2_dim]))
		self.D_W3 = tf.Variable(self.xavier_init([self.h2_dim, self.category_num+1]))
		self.D_b3 = tf.Variable(tf.zeros(shape=[self.category_num+1]))
		self.theta_D = [self.D_W1, self.D_W2, self.D_W3,self.D_b1, self.D_b2, self.D_b3]
		""" extract feature model """
		self.X_feature_S = self.feature_extractor(self.source_images,reuse=None)
		self.X_feature_T = self.feature_extractor(self.target_images,reuse=True)

		self.D_S, self.D_logit_S = self.discriminator(self.X_feature_S)		
		self.D_T, self.D_logit_T = self.discriminator(self.X_feature_T)
		
		self.class_pred_S = self.classifier(self.X_feature_S)
		self.class_pred_T = self.classifier(self.X_feature_T)

		self.D_S_sum = tf.summary.histogram("D_S", self.D_S)
		self.D_T_sum = tf.summary.histogram("D_T", self.D_T)

		self.C_loss_S = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_label_S,logits=self.class_pred_S))
		
		self.class_pred_T_softmax = slim.softmax(self.class_pred_T)
		self.class_pred_S_softmax = slim.softmax(self.class_pred_S)
		self.C_T_softmax_sum_h = tf.summary.histogram("class_pred_T_softmax", self.class_pred_T_softmax)
		#self.C_T_softmax_sum_d = tf.summary.scalar("class_pred_T_softmax", self.class_pred_T_softmax)

		self.C_loss_T = -self.lambda_T *tf.reduce_mean(tf.reduce_sum(self.class_pred_T_softmax * tf.log(self.class_pred_T_softmax), axis=1))
		self.C_loss_S_sum = tf.summary.scalar("C_loss_S", self.C_loss_S)
		self.C_loss_T_sum = tf.summary.scalar("C_loss_T", self.C_loss_T)

		self.D_loss_S = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.D_logit_S, labels=self.label_sd))
		self.D_loss_T = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.D_logit_T, labels=self.label_td))		
		self.D_loss = self.BetaGD * (self.D_loss_S + self.D_loss_T)
		#self.D_loss = self.BetaGD * self.D_loss_T
		self.D_loss_S_sum = tf.summary.scalar("D_loss_S", self.D_loss_S)
		self.D_loss_T_sum = tf.summary.scalar("D_loss_T", self.D_loss_T)
		self.D_loss_sum = tf.summary.scalar("D_loss", self.D_loss)

		self.G_loss_S = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.D_logit_S, labels=self.label_sg))
		self.G_loss_T = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.D_logit_T, labels=self.label_tg))
		self.G_loss = self.BetaGD*(self.G_loss_T + self.G_loss_S)
		#self.G_loss = self.BetaGD*(self.G_loss_T + self.G_loss_S)
		self.G_loss_S_sum = tf.summary.scalar("G_loss_S", self.G_loss_S)
		self.G_loss_T_sum = tf.summary.scalar("G_loss_T", self.G_loss_T)
		self.G_loss_sum = tf.summary.scalar("G_loss", self.G_loss)
		self.saver = tf.train.Saver(max_to_keep=1)
		
	def xavier_init(self,size):
		in_dim = size[0]
		xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
		return tf.random_normal(shape=size, stddev=xavier_stddev)

	def feature_extractor(self,image,reuse):
		conv1 = convLayer(image, 11, 11, 4, 4, 96, "conv1",reuse, "VALID")
		lrn1 = LRN(conv1, 2, 2e-05, 0.75, "norm1") 
		pool1 = maxPoolLayer(lrn1, 3, 3, 2, 2, "pool1", "VALID")
		conv2 = convLayer(pool1, 5, 5, 1, 1, 256, "conv2", reuse,groups = 2)
		lrn2 = LRN(conv2, 2, 2e-05, 0.75, "lrn2")
		pool2 = maxPoolLayer(lrn2, 3, 3, 2, 2, "pool2", "VALID")

		conv3 = convLayer(pool2, 3, 3, 1, 1, 384, "conv3",reuse)

		conv4 = convLayer(conv3, 3, 3, 1, 1, 384, "conv4",reuse, groups = 2)

		conv5 = convLayer(conv4, 3, 3, 1, 1, 256, "conv5", reuse,groups = 2)
		pool5 = maxPoolLayer(conv5, 3, 3, 2, 2, "pool5","VALID")

		fcIn = tf.reshape(pool5, [-1, 256 * 6 * 6])
		fc1 = fcLayer(fcIn, 256 * 6 * 6, 4096, True, "fc6",reuse)
		dropout1 = dropout(fc1, self.KEEPPRO)

		fc2 = fcLayer(dropout1, 4096, 4096, True, "fc7",reuse)

		dropout2 = dropout(fc2, self.KEEPPRO)
		fc3 = fcLayer(dropout2, 4096, 256, True, "fc8",reuse)
		print(tf.shape(fc2)[0])
		return fc3

	def classifier(self,feature):
		C_h1 = tf.matmul(feature, self.C_W1) + self.C_b1
		#C_h2 = tf.nn.relu(tf.matmul(C_h1, self.C_W2) + self.C_b2)
		#C_h3 = tf.matmul(C_h2, self.C_W3) + self.C_b3
		return C_h1
		
	def discriminator(self,feature):
		D_h1 = LeakyRelu(tf.matmul(feature, self.D_W1) + self.D_b1)
		D_h2 = LeakyRelu(tf.matmul(D_h1, self.D_W2) + self.D_b2)
		#D_h3 = tf.nn.leaky_relu(tf.matmul(D_h2, self.D_W3) + self.D_b3)
		D_h3 = tf.matmul(D_h2, self.D_W3) + self.D_b3
		return slim.softmax(D_h3), D_h3
	def loadModel(self, sess):
		"""load model"""
		wDict = np.load(self.MODELPATH, encoding = "bytes").item()
		#for layers in model
		for name in wDict:
			if name not in self.SKIP:
				with tf.variable_scope(name, reuse = True):
					for p in wDict[name]:
						if len(p.shape) == 1:
							#bias
							sess.run(tf.get_variable('b', trainable = False).assign(p))
						else:
							#weights
							sess.run(tf.get_variable('w', trainable = False).assign(p))
		
	def save(self,checkpoint_dir,step):
		model_name = "GAN.model"
		model_dir = "%s_%s_%s_D_%s_%s" % (self.Dataset_name_source,self.Dataset_name_target, self.feature_dim,self.h1_dim,self.h2_dim)
		checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
		
		if not os.path.exists(checkpoint_dir):
			os.makedirs(checkpoint_dir)
			
		self.saver.save(self.sess,
						os.path.join(checkpoint_dir, model_name),
						global_step=step)

	def load(self, checkpoint_dir):
		print("Reading checkpoint...")
		
		model_dir = "%s_%s_%s_D_%s_%s" % (self.Dataset_name_source,self.Dataset_name_target,self.feature_dim,self.h1_dim,self.h2_dim)
		checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
		
		ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
		if ckpt and ckpt.model_checkpoint_path:
			ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
			self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
			return True
		else:
			return False		

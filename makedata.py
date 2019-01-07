#coding:utf-8
'''
从数据集中随机挑选出seed，candidate，choose
并生成image_table
'''
import numpy as np
import os
import random
from PIL import Image
import shutil
from glob import glob
from six.moves import xrange
import cv2

class_name=[]
feature_path=''
#feature_dimensionality = 4096
picture = '.jpg'
imgMean = np.array([104,117,123], np.float) 

def get_data(args):
	list_source=[]
	list_target=[]
	label_s = []
	label_t = []
	target_dir = []
	#f = open("data_class_A31-W10.txt",'w')
	source_path = os.path.join(args.total_feature_path,args.Dataset_name_source)
	for source_root,source_dirs,source_files in os.walk(source_path):
		for dir in source_dirs:
			source_classes = source_dirs
			class_dir_s = os.path.join(source_root,dir)
			class_label_s = source_dirs.index(dir)
			#print("class_dir_s:",class_dir_s,";class_label_s:",class_label_s)
			for file_s in os.listdir(class_dir_s):
				file_path_s = os.path.join(class_dir_s + os.sep, file_s)
				list_source.append(file_path_s)
				label_s.append(class_label_s)
				#temp_s = file_path_s + ' ' + str(calss_label_s) + '\n'
				#f.write(temp_s)
		
	print("source_classes",source_classes)

	target_path = os.path.join(args.total_feature_path,args.Dataset_name_target)
	for target_root,target_dirs,target_files in os.walk(target_path):
		for dir in target_dirs:
			#print("dir:",dir)
			class_dir_t = os.path.join(target_root,dir)
			class_label_t = source_classes.index(dir)
			#print("class_label_t:",class_label_t)
			target_dir.append(class_label_t)
			#print("class_dir_t:",class_dir_t,";calss_label_t:",calss_label_t)
			for file_t in os.listdir(class_dir_t):
				file_path_t = os.path.join(class_dir_t + os.sep, file_t)
				list_target.append(file_path_t)
				label_t.append(class_label_t)
				#temp_t = file_path_t + ' ' + str(calss_label_t) + '\n'
				#f.write(temp_t)

	#f.close()
	print("target_classes:",target_dir)
	data_source = list(zip(list_source,label_s))
	data_target = list(zip(list_target,label_t))
	return data_source,data_target,target_dir,source_classes

def get_select_data(args,class_remove):
	list_source=[]
	list_target=[]
	label_s = []
	source_classes = []
	#f = open("data_class_A31-W10.txt",'w')
	source_path = os.path.join(args.total_feature_path,args.Dataset_name_source)
	for source_root,source_dirs,source_files in os.walk(source_path):
		for dir in source_dirs:
			#source_classes = source_dirs
			class_dir_s = os.path.join(source_root,dir)
			class_label_s = source_dirs.index(dir)
			if class_label_s in class_remove:
				#print("class_remove:",class_remove)
				print("class_label_s:",class_label_s)
				print("dir:",dir)
				continue
			source_classes.append(dir)
			#print("class_dir_s:",class_dir_s,";calss_label_s:",calss_label_s)
			for file_s in os.listdir(class_dir_s):
				file_path_s = os.path.join(class_dir_s + os.sep, file_s)
				list_source.append(file_path_s)
				label_s.append(class_label_s)
				#temp_s = file_path_s + ' ' + str(calss_label_s) + '\n'
				#f.write(temp_s)		
	print("source_classes",source_classes)
	data_source = list(zip(list_source,label_s))
	return data_source


def one_hot_label(label_batch,domain_code,FlagD,args):

	num = len(label_batch)
	label = np.zeros((num,args.category_num+1))
	if FlagD:
		#the label for D
		if domain_code == 0:
			#the label for source
			for i in xrange(num):
				label[i,label_batch[i]]=1
		else:
			label[:,args.category_num]=1
	else:
		if domain_code == 0:
			label[:,args.category_num] = 1
		else:
			for i in xrange(num):
				label[i,label_batch[i]]=1
	#print("domain_code:",domain_code,"FlagD:",FlagD,"label:",label,)
	return label

def process_imgs(path_list):
	batch_img = np.zeros([len(path_list),227,227,3])
	for i in xrange(len(path_list)):
		#print(path_list[i])
		img = cv2.resize(cv2.imread(path_list[i]),(227,227)) - imgMean
		if len(img.shape)==2:
			batch_img[i,:,:,0] = img
			batch_img[i,:,:,1] = img
			batch_img[i,:,:,2] = img
		else:
			batch_img[i,:,:,:] = img
	#print(batch_img.shape)
	return batch_img
def process_img(path):
	batch_img = np.zeros([1,227,227,3])
	#print(path)
	img = cv2.resize(cv2.imread(path),(227,227)) - imgMean
	if len(img.shape)==2:
		batch_img[0,:,:,0] = img
		batch_img[0,:,:,1] = img
		batch_img[0,:,:,2] = img
	else:
		batch_img[0,:,:,:] = img
	#print(batch_img.shape)
	return batch_img




def load_data(args):
	#source_dataset_order=random.randrange(0,len(Dataset_name))
	class_random_order=random.randrange(0,len(args.Class_name))
	#print('class label:',class_random_order)
	data_source = glob('{}/{}-alexnet_feature/{}/*.npy'.format(args.total_feature_path,args.Dataset_name_source,args.Class_name[class_random_order]))
	data_target = glob('{}/{}-alexnet_feature-seed/{}/*.npy'.format(args.total_feature_path,args.Dataset_name_target,args.Class_name[class_random_order]))
	#print('len(data_source):',len(data_source))
	#print('len(data_target):',len(data_target))
	batch_source = random.sample(data_source,args.batch_size)
	batch_target= random.sample(data_target,args.batch_size)
	return batch_source,batch_target,class_random_order

def get_nextstate(args,state,selected_num):
    temp = np.zeros([1,args.feature_dim])
    #print(state.shape)
    #print(temp.shape)
    for i in xrange(selected_num):
        state=np.vstack((state,temp))
    #print(state)
    #p = np.array([[p]])
    #next_state = np.hstack((state.reshape(1,-1),p))
    #print("NextState",next_state.shape)
    next_state = state.reshape(1,-1)
    return next_state


def read_feature(path_list,table):
	class_num = len(path_list)
	sample_feature = np.zeros((table.shape[0],class_num*feature_dimensionality))
	for i in range(class_num):
		for j in range(table.shape[0]):
				temp = table[j][i].astype(int)
				#print(temp,path_list[i][temp])
				sample_feature[j,feature_dimensionality*(i):feature_dimensionality*(i+1)] = np.load(path_list[i][temp])
	return sample_feature

def read_feature_matrix(path_list,table):
	class_num = len(path_list)
	sample_feature = np.zeros((table.shape[0]*class_num,feature_dimensionality))
	
	for i in range(class_num):
		for j in range(table.shape[0]):
				temp = table[j][i].astype(int)
				sample_feature[(i-1)*table.shape[0]+j,:]= np.load(path_list[i][temp])
	return sample_feature

def get_feature_label_class(path,dimensionality,class_name):
	label_list = []
	feature_list = []
	label = 1;
	for i in range(len(class_name)):
		class_dir = os.path.join(path + os.sep,class_name[i])
		for file in os.listdir(class_dir):
			file_path = os.path.join(class_dir + os.sep,file)
			feature_list = feature_list + np.load(file_path).tolist()
			label_list = label_list + [label]
		label = label + 1
	feature = np.array(feature_list).reshape(-1,dimensionality)
	label = np.array(label_list)
	print (feature.shape,label.shape)
	return feature,label

def get_feature_label(path,dimensionality):
	label_list = []
	feature_list = []
	label = 1;
	for root,dirs,files in os.walk(path):
		for dir in dirs:
			class_dir = os.path.join(root,dir)
			print(class_dir)
			for file in os.listdir(class_dir):
				file_path = os.path.join(class_dir + os.sep,file)
				feature_list = feature_list + np.load(file_path).tolist()
				label_list = label_list + [label]
			label = label + 1
	feature = np.array(feature_list).reshape(-1,dimensionality)
	label = np.array(label_list)
	print(feature.shape,label.shape)
	return feature,label

def save_image(path_list,table,update_data_iter,feature_path_c,image_path_c,image_save_path,feature_save_path,class_name):
	class_num = len(path_list)
	image_save_path = image_save_path + '_' + str(update_data_iter)
	feature_save_path = feature_save_path + '_' + str(update_data_iter)
	if not os.path.exists(image_save_path):
		os.mkdir(image_save_path)
	for i in range(class_num):
		save_path = image_save_path + os.sep + class_name[i]
		save_path_feature = feature_save_path + os.sep + class_name[i]
		if os.path.exists(save_path):
			shutil.rmtree(save_path)
		if os.path.exists(save_path_feature):
			shutil.rmtree(save_path_feature)
		if not os.path.exists(save_path):
			os.mkdir(save_path)		
		if not os.path.exists(save_path_feature):
			os.makedirs(save_path_feature)
		for j in range(table.shape[0]):
			a = table[j][i].astype(int)
			feature = np.load(path_list[i][a])
			np.save(save_path_feature + os.sep + os.path.split(path_list[i][a])[1],feature)
			temp = path_list[i][a].replace(feature_path_c,image_path_c)
			temp = temp.replace('.npy',picture)
			#print temp
			im = Image.open(temp)
			im.save(save_path + os.sep + os.path.split(temp)[1])

def save_feature(path_list,table,update_data_iter,feature_save_path,class_name):
	class_num = len(path_list)
	#image_save_path = image_save_path + '_' + str(update_data_iter)
	feature_save_path = feature_save_path + '_' + str(update_data_iter)
	#if not os.path.exists(image_save_path):
		#os.mkdir(image_save_path)
	for i in range(class_num):
		#save_path = image_save_path + os.sep + class_name[i]
		save_path_feature = feature_save_path + os.sep + class_name[i]
		#if not os.path.exists(save_path):
			#os.mkdir(save_path)
		if os.path.exists(save_path_feature):
			shutil.rmtree(save_path_feature)			
		if not os.path.exists(save_path_feature):
			os.makedirs(save_path_feature)	
		for j in range(table.shape[0]):
			a = table[j][i].astype(int)
			feature = np.load(path_list[i][a])
			np.save(save_path_feature + os.sep + os.path.split(path_list[i][a])[1],feature)
			#temp = path_list[i][a].replace(feature_path_c,image_path_c)
			#temp = temp.replace('.npy',picture)
			#print temp
			#im = Image.open(temp)
			#im.save(save_path + os.sep + os.path.split(temp)[1])

def save_feature_generate(path_list,table,update_data_iter,feature_save_path,class_name,feature_generate):
	class_num = len(path_list)
	feature_save_path = feature_save_path + '_' + str(update_data_iter)
	for i in range(class_num):
		save_path_feature = feature_save_path + os.sep + class_name[i]
		#if not os.path.exists(save_path):
			#os.mkdir(save_path)]
		if os.path.exists(save_path_feature):
			shutil.rmtree(save_path_feature)	
		if not os.path.exists(save_path_feature):
			os.makedirs(save_path_feature)	
		for j in range(table.shape[0]):
			a = table[j][i].astype(int)
			feature = feature_generate[j]
			np.save(save_path_feature + os.sep + os.path.split(path_list[i][a])[1],feature)	
	



def get_seed(datapath):
	feature = np.load(datapath)

def get_candidate_all(path_list,candidate_num,class_name=[]):
	candidate_path_list = []
	total_num=len(path_list[0])
	start_array=np.arange(candidate_num)
	candidate=np.random.permutation(candidate_num)+(total_num-candidate_num)
	
	for j in range(len(class_name)):
		class_candidate_path=[]
		for i in range(len(candidate)):
			class_candidate_path.append(path_list[j][i])
		candidate_path_list.append(class_candidate_path)
	return candidate_path_list

def get_candidate_set(candidate_path_list,this_set_num,candidate_set_num,class_name=[]):
	candidate_set_path_list=[]
	for i in range(len(class_name)):
		candidate_set_path_list.append(candidate_path_list[i][candidate_set_num*(this_set_num-1):candidate_set_num*this_set_num])
	candidate_set_feature = read_feature(candidate_set_path_list)
	return candidate_set_path_list,candidate_set_feature
	
#给定
def get_choose(path_list,choose_num,class_name=[]):
	choose_path_list = []
	for i in range(len(class_name)):
		choose_path_list.append(path_list[i][0:choose_num])
	choose_feature = read_feature(choose_path_list)
	return choose_path_list,choose_feature
	

#给定数据库特征路径，返回list
def get_list(class_name=[],feature_path=''):  
	path_list=[];
	for i in range(len(class_name)):
		class_list=[];
		for file in os.listdir(feature_path + os.sep + class_name[i]):
			file_path = os.path.join(feature_path + os.sep + class_name[i]+os.sep, file) 
			class_list.append(file_path);
		path_list.append(class_list);		
	return 	path_list
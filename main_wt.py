import argparse
import os
import numpy as np 
import scipy.misc
import tensorflow as tf 
import random
import time
import makedata
from skimage import transform
from six.moves import xrange
from GAN import GAN_Net
from DQN import DQN_Net
from glob import glob
from scipy.misc import imread
from collections import deque 

parser = argparse.ArgumentParser(description='')

parser.add_argument('--Dataset_name_source', dest='Dataset_name_source', default='C256')
parser.add_argument('--Dataset_name_target', dest='Dataset_name_target', default='W10')
#parser.add_argument('--Class_name', dest='Class_name', default=\
#['motorbike','monitor','horse','dog','car','bottle','boat','bird','bike','airplane','people','bus'])
#parser.add_argument('--Class_name', dest='Class_name', default=['projector','mug','mouse','monitor','laptop_computer','keyboard','headphones','calculator','bike','back_pack'])
parser.add_argument('--category_num', dest='category_num', type=int, default=256)
parser.add_argument('--total_feature_path', dest='total_feature_path', default=\
    '/media/mcislab3d/Elements/chenjin/CVPR18/data')
parser.add_argument('--epoch', dest='epoch', type=int, default=5000, help='# of epoch')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=16, help='# batchsize of GAN')
parser.add_argument('--sample_dqn', dest='sample_dqn', type=int, default=32, help='# batchsize of RL')
parser.add_argument('--state_size', dest='state_size', default=256*128, help='scale state to this size')
parser.add_argument('--candidate_num', dest='candidate_num', type=int, default=128, help='candidate number')
#parser.add_argument('--select_num', dest='select_num', type=int, default=4, help='select number')
parser.add_argument('--feature_dim', dest='feature_dim', type=int, default=256)
#parser.add_argument('--save_iter_gan', dest='save_iter_gan', type=int, default=50,help='save model')
parser.add_argument('--test_iter', dest='test_iter', type=int, default=10)
parser.add_argument('--test_iter_s', dest='test_iter_s', type=int, default=500)

parser.add_argument('--lr', dest='lr', type=float, default=1e-4, help='initial learning rate for GDC')
parser.add_argument('--beta1', dest='beta1', type=float, default=0.5, help='momentum term of adam')
parser.add_argument('--beta2', dest='beta2', type=float, default=0.9, help='momentum term of adam')
parser.add_argument('--BetaGD', dest='BetaGD', type=float, default=1.0, help='initial learning rate for adam')
parser.add_argument('--LambdaT', dest='LambdaT', type=float, default=0, help='initial learning rate for adam')
parser.add_argument('--lr_Q', dest='lr_Q', type=float, default=1e-4, help='initial learning rate for adam')
parser.add_argument('--OBSERVE', dest='OBSERVE', type=int, default=50, help='# OBSERVE for the agent')
parser.add_argument('--EXPLORE', dest='EXPLORE', type=int, default=2000, help='# EXPLORE for the agent')
parser.add_argument('--INITIAL_EPSILON', dest='INITIAL_EPSILON', type=float, default=1.0, help='# EXPLORE for the agent')
parser.add_argument('--FINAL_EPSILON', dest='FINAL_EPSILON', type=float, default=0, help='# EXPLORE for the agent')
#parser.add_argument('--train_for_GAN', dest='train_for_GAN', type=int, default=0, help='# train for GAN')
#parser.add_argument('--add_entropy', dest='add_entropy', type=int, default=500, help='# train for GAN')
#parser.add_argument('--save_iter_dqn', dest='save_iter_dqn', type=int, default=50, help='# train for GAN')
parser.add_argument('--replace_target_iter', dest='replace_target_iter', type=int, default=20, help='# train for GAN')
parser.add_argument('--REPLAY_MEMORY', dest='REPLAY_MEMORY', type=int, default=200, help='# EXPLORE for the agent')
parser.add_argument('--REPLAY_MEMORY_GAN', dest='REPLAY_MEMORY_GAN', type=int, default=20, help='# EXPLORE for GAN')
parser.add_argument('--GAMMA', dest='GAMMA', type=float, default=0.9, help='momentum term of adam')
parser.add_argument('--checkpoint_dir_gan', dest='checkpoint_dir_gan', default='./checkpoint_gan', help='models are saved here')
parser.add_argument('--checkpoint_dir_dqn', dest='checkpoint_dir_dqn', default='./checkpoint_dqn', help='models are saved here')
parser.add_argument('--checkpoint_gan_pretrain', dest='checkpoint_gan_pretrain', default='./checkpoint-finetune-bottleneck-c5', help='models are saved here')
parser.add_argument('--alexnet_model_path', dest='alexnet_model_path', default='/media/mcislab3d/Elements/chenjin/TNNLS/bvlc_alexnet.npy')
parser.add_argument('--keepPro', dest='keepPro',type=int, default=1.0)
parser.add_argument('--skip', dest='skip', default=['fc8'])
parser.add_argument('--train_layers', dest='train_layers', default=['fc8'])
parser.add_argument('--r_threshold', type= float, dest='r_threshold', default=0.3)
parser.add_argument('--r', type=int,dest='r', default=1)
args = parser.parse_args()
print(args)


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES']='1' 
gpu_options = tf.GPUOptions(allow_growth=True)
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
sess_DQN = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
sess_GAN = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
#sess_Alexnet = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

def main():
    if not os.path.exists(args.checkpoint_dir_gan):
        os.makedirs(args.checkpoint_dir_gan)
    if not os.path.exists(args.checkpoint_dir_dqn):
        os.makedirs(args.checkpoint_dir_dqn)        
    train(args) 

def train(args):
    #np.set_printoptions(threshold='nan')
    print("D_score_w>t: r=+1------------------------------------------------------------------------------------")
    np.set_printoptions(threshold=np.inf)  
    filename = 'accuracy_wt_test_'+args.Dataset_name_source+args.Dataset_name_target+'.txt'
    f = open(filename,"a+")
    max_acc = 0
    GANetwork = GAN_Net(sess_GAN,args)
    DQNetwork = DQN_Net(sess_DQN,args)
    select_set = deque(maxlen=args.REPLAY_MEMORY_GAN)
    select_set_update = deque(maxlen=args.REPLAY_MEMORY_GAN)
    var_list_alexnet = [v for v in tf.trainable_variables() if v.name.split('/')[0] in args.train_layers]

    D_optim = tf.train.AdamOptimizer(args.lr, beta1=args.beta1).minimize(GANetwork.D_loss, var_list=GANetwork.theta_D)
    G_optim = tf.train.AdamOptimizer(args.lr, beta1=args.beta1).minimize(GANetwork.G_loss, var_list=var_list_alexnet)
    C_optim_S = tf.train.AdamOptimizer(args.lr, beta1=args.beta1)\
                .minimize(GANetwork.C_loss_S, var_list=[GANetwork.theta_C,var_list_alexnet])
    #C_optim_T = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
    #            .minimize(GANetwork.C_loss_T, var_list=[GANetwork.theta_C,var_list_alexnet])

    sess_DQN.run(tf.global_variables_initializer())
    sess_GAN.run(tf.global_variables_initializer())
    GANetwork.load(args.checkpoint_gan_pretrain)

    #GANetwork.loadModel(sess_GAN)
    D_sum = tf.summary.merge([GANetwork.D_loss_S_sum,GANetwork.D_loss_T_sum,GANetwork.D_loss_sum,GANetwork.D_S_sum,GANetwork.D_T_sum])
    G_sum = tf.summary.merge([GANetwork.G_loss_sum, GANetwork.G_loss_S_sum,GANetwork.G_loss_T_sum])
    C_sum_S = GANetwork.C_loss_S_sum
    #C_sum_T = GANetwork.C_loss_T_sum
    Q_sum = DQNetwork.Q_loss_sum
    writer = tf.summary.FileWriter("./logs-GAN-test4",sess_GAN.graph)
    writer_dqn = tf.summary.FileWriter("./logs-DQN-test4",sess_DQN.graph)
    data_source,data_target,target_classes,source_classes  = makedata.get_data(args)
    correct_num = 0
    data_test_lists,test_labels = zip(*data_target)
    target_images_total = makedata.process_imgs(data_test_lists)      
    test_image = np.zeros([1,227,227,3])
    class_weight = np.zeros((1,256))
    #maxtir = np.zeros((256,256)).astype(int)
    #maxtir_final = np.zeros((10,256)).astype(int)
    for test_it in xrange(len(data_test_lists)):
        #print("data_test_lists[test_it]:",data_test_lists[test_it])
        test_image[0,:,:,:]= target_images_total[test_it]
        class_pred_T_softmax  = sess_GAN.run([GANetwork.class_pred_T_softmax],\
            feed_dict={GANetwork.target_images: test_image})
        #print("class_pred_T_softmax:",class_pred_T_softmax)
        #test = test+class_pred_T_softmax
        label_pred = np.argmax(class_pred_T_softmax[0],axis=1)
        #maxtir[test_labels[test_it]][label_pred] = maxtir[test_labels[test_it]][label_pred] + 1
        class_weight = class_weight + class_pred_T_softmax[0]
        if label_pred[0] == test_labels[test_it]:
            #print("correct-----------------------------------------------------------")
            correct_num = correct_num + 1
    #print("correct_num:",correct_num)
    #print("len(data_test_lists):",len(data_test_lists))
    target_accuracy = correct_num/float(len(data_test_lists))   

    class_weight = class_weight/float(len(data_test_lists))
    class_weight = class_weight/np.max(class_weight) 
    class_weight = class_weight.tolist()[0]
    #print("class_weight:",class_weight)
    #print("len(class_weight):",len(class_weight))
    #print("sum(class_weight):",sum(class_weight))   
   # print("target_accuracy:",target_accuracy)
    #data_source_select  = makedata.get_select_data(args,class_remove)
    for i in xrange(args.epoch):        
        #random.shuffle(data_source)
        D_score = np.zeros([args.candidate_num])
        data_batch_source = random.sample(data_source,args.candidate_num)
        batch_source_list,batch_label = zip(*data_batch_source)
        data_batch_target = random.sample(data_target,args.candidate_num)
        batch_target_list,batch_target_label = zip(*data_batch_target)

        batch_source_images = makedata.process_imgs(list(batch_source_list))
        batch_target_images = makedata.process_imgs(list(batch_target_list))

        state,D_sorce_c = sess_GAN.run([GANetwork.X_feature_S,GANetwork.D_S],\
            feed_dict={GANetwork.source_images:batch_source_images})
        D_score = D_sorce_c[:,args.category_num].copy()
        D_score_w = []
        baseline = 0

        for k in xrange(args.candidate_num):
            #print("batch_label[k]:",batch_label[k])
            #print("class_weight[batch_label[k]]:",class_weight[batch_label[k]])
            aaa = D_score[k]*class_weight[batch_label[k]]
            baseline = baseline + aaa
            D_score_w.append(aaa)

        baseline = baseline/float(args.candidate_num)
        print("baseline:",baseline)

        for j in xrange(len(D_score_w)):
            if batch_label[j] in target_classes:
                print("InClasses,D_score_w:",D_score_w[j],"source_label:", batch_label[j])
            else:
                print("OutClasses,D_score_w:",D_score_w[j],"source_label:", batch_label[j])
        DQNetwork.setState(state)

        it = 0 
        while batch_source_images.shape[0] > 0:
            terminal = 0
            action,action_index,Flag= DQNetwork.getAction(args,it)
            #if batch_label[action_index] in target_classes:
            #    reward = args.r 
            #else:
            #    reward = -args.r
            #if class_weight[batch_label[action_index]]>args.r_threshold:
            #    reward = args.r
            #else:
            if D_score_w[action_index] > args.r_threshold:
                reward = args.r                              
            else:
                reward = -args.r
            print("target_classes:",target_classes)      
            print("action random",Flag,"source_label:",batch_label[action_index],\
                "D_score_w:",D_score_w[action_index],"reward:",reward)
            if DQNetwork.timeStep < args.OBSERVE:
                print("OBSERVE-------------------------------------------------------------------")
                select_set.append((batch_source_images[action_index],batch_label[action_index]))
            else:
                if reward > 0:
                    select_set_update.append((batch_source_images[action_index],batch_label[action_index])) 
            D_score_w = np.delete(D_score_w,action_index,axis=0)
            batch_source_images = np.delete(batch_source_images,action_index,axis = 0)
            batch_label = np.delete(batch_label,action_index,axis = 0)
            state = np.delete(state,action_index,axis=0)

            it = it+1
            next_state = makedata.get_nextstate(args,state,it)

            if reward < 0:
                terminal = 1
            DQNetwork.setPerception(args,action,reward,next_state,it,terminal,i,writer_dqn)

            if reward < 0:
                break

        if len(select_set) > args.batch_size and DQNetwork.timeStep < args.OBSERVE:
            print("update discriminator---------------------------------------------------------")
            minibatch = random.sample(select_set,args.batch_size)
            source_batch = [data[0] for data in minibatch]
            source_batch_label = [data[1] for data in minibatch]
            print("source_batch_label:",source_batch_label)
            target_data = random.sample(data_target,args.batch_size)
            target_batch,target_batch_label = zip(*target_data)
            batch_target_images = makedata.process_imgs(list(target_batch))

            D_T_softmax = sess_GAN.run([GANetwork.class_pred_T_softmax],\
                feed_dict={GANetwork.target_images: batch_target_images})

            D_T_softmax = D_T_softmax[0]
            #print("D_T_softmax.shape",D_T_softmax.shape)
            label_t = np.argmax(D_T_softmax,axis = 1)

            source_label_d = makedata.one_hot_label(source_batch_label,0,True,args)
            target_label_d = makedata.one_hot_label(label_t,1,True,args)
            source_label_g = makedata.one_hot_label(source_batch_label,0,False,args)
            target_label_g = makedata.one_hot_label(label_t,1,False,args)
            #print("target_label_d:",target_label_d)

            _, summary_str,errD = sess_GAN.run([D_optim, D_sum,GANetwork.D_loss],\
            feed_dict={GANetwork.source_images: source_batch,\
                       GANetwork.target_images: batch_target_images,\
                       GANetwork.label_sd:source_label_d, \
                       GANetwork.label_td:target_label_d})
            writer.add_summary(summary_str, i)
            print("Epoch: [%2d] d_loss: %.8f "\
                % (i, errD))  
        if len(select_set_update) > args.batch_size:
            print("adding adversarial loss---------------------------------------------------------")
            minibatch = random.sample(select_set_update,args.batch_size)
            source_batch = [data[0] for data in minibatch]
            source_batch_label = [data[1] for data in minibatch]
            print("source_batch_label:",source_batch_label)
            target_data = random.sample(data_target,args.batch_size)
            target_batch,target_batch_label = zip(*target_data)
            batch_target_images = makedata.process_imgs(list(target_batch))

            D_T_softmax = sess_GAN.run([GANetwork.class_pred_T_softmax],\
                feed_dict={GANetwork.target_images: batch_target_images})

            D_T_softmax = D_T_softmax[0]
            #print("D_T_softmax.shape",D_T_softmax.shape)
            label_t = np.argmax(D_T_softmax,axis = 1)

            source_label_d = makedata.one_hot_label(source_batch_label,0,True,args)
            target_label_d = makedata.one_hot_label(label_t,1,True,args)
            source_label_g = makedata.one_hot_label(source_batch_label,0,False,args)
            target_label_g = makedata.one_hot_label(label_t,1,False,args)
            #print("target_label_d:",target_label_d)

            _, summary_str,errD = sess_GAN.run([D_optim, D_sum,GANetwork.D_loss],\
            feed_dict={GANetwork.source_images: source_batch,\
                       GANetwork.target_images: batch_target_images,\
                       GANetwork.label_sd:source_label_d, \
                       GANetwork.label_td:target_label_d})
            writer.add_summary(summary_str, i)

    # Update G network
            _, summary_str,errG = sess_GAN.run([G_optim, G_sum,GANetwork.G_loss],\
            feed_dict={GANetwork.source_images: source_batch,\
                       GANetwork.target_images: batch_target_images,\
                       GANetwork.label_sg:source_label_g, \
                       GANetwork.label_tg:target_label_g})
            writer.add_summary(summary_str, i)

            _, summary_str,errC_S = sess_GAN.run([C_optim_S, C_sum_S,GANetwork.C_loss_S],\
            feed_dict={GANetwork.source_images: source_batch,\
                       GANetwork.y_label_S: source_batch_label})                
            writer.add_summary(summary_str, i)

            #if i > args.add_entropy:
            #    _, summary_str,errC_T = sess_GAN.run([C_optim_T, C_sum_T,GANetwork.C_loss_T],\
            #        feed_dict={GANetwork.target_images: batch_target_images})        
            #    writer.add_summary(summary_str, i)
            #    print("Epoch: [%2d] d_loss: %.8f, g_loss: %.8f, C_loss_S: %.8f,C_loss_T: %.8f"\
            #    % (i, errD, errG,errC_S,errC_T)) 
            #else:   
            print("Epoch: [%2d] d_loss: %.8f, g_loss: %.8f, C_loss_S: %.8f"\
                % (i, errD,errG,errC_S))                          
        if np.mod(i,args.test_iter) == 0:
            correct_num = 0
            #data_test_lists,test_labels = zip(*data_target)            
            for test_it in xrange(len(data_test_lists)):
                test_image[0,:,:,:] = target_images_total[test_it]
                class_pred_T_softmax = sess_GAN.run([GANetwork.class_pred_T_softmax],\
                    feed_dict={GANetwork.target_images: test_image})
                label_pred = np.argmax(class_pred_T_softmax[0],axis=1)
                if label_pred == test_labels[test_it]:
                    correct_num = correct_num + 1
            target_accuracy = correct_num/float(len(target_images_total))       
            print("Epoch: [%2d] target accuracy: %.8f " % (i,target_accuracy))
            if target_accuracy > max_acc:
                max_acc = target_accuracy
                print("update max_acc")
                #GANetwork.save(args.checkpoint_dir_gan,i)  
                #DQNetwork.save(args.checkpoint_dir_dqn, DQNetwork.timeStep)  

        if np.mod(i,args.test_iter_s) == 0 and i > 0:
            correct_num_s = 0
            source_list,source_label = zip(*data_source) 
            for test_it_s in xrange(len(source_list)):
                test_image[0,:,:,:] = makedata.process_img(source_list[test_it_s]) 
                class_pred_S_softmax = sess_GAN.run([GANetwork.class_pred_S_softmax],\
                    feed_dict={GANetwork.source_images: test_image})                  
                label_pred_s = np.argmax(class_pred_S_softmax[0],axis=1)
                if label_pred_s == source_label[test_it_s]:
                    correct_num_s = correct_num_s + 1                
            source_accuracy = correct_num_s/float(len(source_list))     
            print("Epoch: [%2d] source accuracy: %.8f " % (i,source_accuracy))  
    temp = str(float('%.2f'%(max_acc*100)))+'\n'
    f.write(temp)    
    f.close()    
if __name__ == '__main__':
    main()


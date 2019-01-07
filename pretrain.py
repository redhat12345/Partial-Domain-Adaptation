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
parser.add_argument('--Dataset_name_target', dest='Dataset_name_target', default='D10')
#parser.add_argument('--Class_name', dest='Class_name', default=\
#['motorbike','monitor','horse','dog','car','bottle','boat','bird','bike','airplane','people','bus'])
#parser.add_argument('--Class_name', dest='Class_name', default=['projector','mug','mouse','monitor','laptop_computer','keyboard','headphones','calculator','bike','back_pack'])
parser.add_argument('--category_num', dest='category_num', type=int, default=256)
parser.add_argument('--total_feature_path', dest='total_feature_path', default=\
    '/media/mcislab3d/Elements/chenjin/CVPR18/data')
#parser.add_argument('--total_feature_path', dest='total_feature_path', default=\
#    '/home/mcislab/chenjin/data')
parser.add_argument('--epoch', dest='epoch', type=int, default=2000, help='# of epoch')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=128, help='# batchsize of GAN')
parser.add_argument('--feature_dim', dest='feature_dim', type=int, default=256)
parser.add_argument('--save_iter_gan', dest='save_iter_gan', type=int, default=50,help='save model')
parser.add_argument('--test_iter', dest='test_iter', type=int, default=2)
parser.add_argument('--test_iter_s', dest='test_iter_s', type=int, default=500)

parser.add_argument('--lr', dest='lr', type=float, default=1e-4, help='initial learning rate for GDC')
parser.add_argument('--beta1', dest='beta1', type=float, default=0.9, help='momentum term of adam')
parser.add_argument('--BetaGD', dest='BetaGD', type=float, default=1.0, help='initial learning rate for adam')
parser.add_argument('--LambdaT', dest='LambdaT', type=float, default=0, help='initial learning rate for adam')
parser.add_argument('--checkpoint_dir_gan_pretrain', dest='checkpoint_dir_gan_pretrain', default='./checkpoint-finetune-bottleneck-c5', help='models are saved here')
parser.add_argument('--phase', dest='phase', default='train', help='train, test')
parser.add_argument('--alexnet_model_path', dest='alexnet_model_path', default='/media/mcislab3d/Elements/chenjin/TNNLS/bvlc_alexnet.npy')
parser.add_argument('--keepPro', dest='keepPro',type=int, default=1)
parser.add_argument('--skip', dest='skip', default=['fc8'])
parser.add_argument('--feature_layers', dest='feature_layers', default=['conv5','fc6','fc7'])
parser.add_argument('--bottleneck_layers', dest='bottleneck_layers', default=['fc8'])
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
    if not os.path.exists(args.checkpoint_dir_gan_pretrain):
        os.makedirs(args.checkpoint_dir_gan_pretrain)
#    if not os.path.exists(args.checkpoint_dir_dqn):
#        os.makedirs(args.checkpoint_dir_dqn)        
    protrain(args) 

def protrain(args):
    max_acc = 0
    filename = 'accuracy_bottleneckfinetuneall_'+args.Dataset_name_source+args.Dataset_name_target+'.txt'
    f = open(filename,"a+")
    GANetwork = GAN_Net(sess_GAN,args)
    var_list_feature_layers = [v for v in tf.trainable_variables() if v.name.split('/')[0] in args.feature_layers]
    var_list_bottleneck_layers = [v for v in tf.trainable_variables() if v.name.split('/')[0] in args.bottleneck_layers]
    C_optim_SF = tf.train.AdamOptimizer(args.lr, beta1=args.beta1)\
                .minimize(GANetwork.C_loss_S, var_list=[var_list_feature_layers])
    C_optim_SB = tf.train.AdamOptimizer(10*args.lr, beta1=args.beta1)\
                .minimize(GANetwork.C_loss_S, var_list=[GANetwork.theta_C,var_list_bottleneck_layers])
    #C_optim_T = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
    #            .minimize(GANetwork.C_loss_T, var_list=[GANetwork.theta_C,var_list_alexnet])

    sess_GAN.run(tf.global_variables_initializer())
    GANetwork.loadModel(sess_GAN)
    #D_sum = tf.summary.merge([GANetwork.D_loss_S_sum,GANetwork.D_loss_T_sum,GANetwork.D_loss_sum,GANetwork.D_S_sum,GANetwork.D_T_sum])
    #G_sum = tf.summary.merge([GANetwork.G_loss_sum, GANetwork.G_loss_S_sum,GANetwork.G_loss_T_sum])
    C_sum_S = GANetwork.C_loss_S_sum
    #C_sum_T = GANetwork.C_loss_T_sum
    #Q_sum = DQNetwork.Q_loss_sum
    #writer = tf.summary.FileWriter("./logs",sess_GAN.graph)
    #writer_dqn = tf.summary.FileWriter("./logs-DQN",sess_DQN.graph)
    data_source,data_target,_,_  = makedata.get_data(args)
    #target_list,target_label = zip(*data_target)
    #target_images_total = makedata.process_imgs(target_list)    
    #source_list, source_label = zip(*data_source)
    #source_images_total = makedata.process_imgs(source_list)
    print(len(data_target))

    for i in xrange(args.epoch):        
        #random.shuffle(data_source)
        data_batch_source = random.sample(data_source,args.batch_size)
        batch_source_list,batch_label = zip(*data_batch_source)
        #data_batch_target = random.sample(data_target,args.candidate_num)
        #batch_target_list,batch_target_label = zip(*data_batch_target)
        batch_label = list(batch_label)
        print(len(batch_label))

        batch_source_images = makedata.process_imgs(list(batch_source_list))
        #batch_target_images = makedata.process_imgs(list(batch_target_list))

        _, _,errC_S = sess_GAN.run([C_optim_SF,C_optim_SB,GANetwork.C_loss_S],\
        feed_dict={GANetwork.source_images: batch_source_images,\
                   GANetwork.y_label_S: batch_label})                
        #writer.add_summary(summary_str, i)
        print("Epoch: [%2d] C_loss_S: %.8f" % (i,errC_S))                            

#print('______________________________________________________')
        #test_image = np.zeros([1,227,227,3])
        if np.mod(i,args.test_iter) == 0 and i>0:
            correct_num = 0
            data_test_lists,test_labels = zip(*data_target)
            
            for test_it in xrange(len(data_test_lists)):
                test_image = makedata.process_img(data_test_lists[test_it])
                class_pred_T_softmax = sess_GAN.run([GANetwork.class_pred_T_softmax],\
                    feed_dict={GANetwork.target_images: test_image})
                label_pred = np.argmax(class_pred_T_softmax[0],axis=1)
                if label_pred == test_labels[test_it]:
                    correct_num = correct_num + 1
            target_accuracy = correct_num/float(len(data_test_lists))       
            print("Epoch: [%2d] target accuracy: %.8f " % (i,target_accuracy))
            if target_accuracy > max_acc:
                max_acc = target_accuracy
                print("saving model")
                GANetwork.save(args.checkpoint_dir_gan_pretrain,i)  
                #DQNetwork.save(args.checkpoint_dir_dqn, DQNetwork.timeStep)  

        if np.mod(i,args.test_iter_s) == 0 and i > 0:
            correct_num_s = 0
            source_list, source_label = zip(*data_source)
            for test_it_s in xrange(len(source_list)):                
                #print("source_list[test_it_s]:",source_list[test_it_s])
                test_image_s = makedata.process_img(source_list[test_it_s])
                class_pred_S_softmax = sess_GAN.run([GANetwork.class_pred_S_softmax],\
                    feed_dict={GANetwork.source_images: test_image_s})                  
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


from __future__ import print_function
from numpy.random import seed
seed(1)
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import os
from PIL import Image
import io
from sklearn.model_selection import StratifiedShuffleSplit
from vgg16module import VGG16

from keras.models import Model, model_from_json, model_from_yaml, Sequential
from keras.layers import Input, Convolution2D, MaxPooling2D, LSTM, Reshape, Merge, TimeDistributed, Flatten, Activation, Dense, Dropout, merge, AveragePooling2D, ZeroPadding2D, Lambda
from keras.optimizers import Adam, SGD
from keras.layers.normalization import BatchNormalization 
from keras import backend as K
K.set_image_dim_ordering('th')
from keras.utils import np_utils
from sklearn.metrics import confusion_matrix, accuracy_score
from skimage.io import imsave
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from keras.utils.np_utils import to_categorical
import json
from scipy.ndimage import minimum, maximum, imread
import math
import numpy.ma as ma
import matplotlib.cm as cm
import h5py
import random
from collections import OrderedDict
import scipy.io as sio
import cv2
import glob
import gc
from scipy.stats import mode
from collections import Counter
from sklearn import svm
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import KFold
from keras.layers.advanced_activations import ELU

def plot_training_info(case, metrics, save, history):
    # summarize history for accuracy
    plt.ioff()
    if 'accuracy' in metrics:     
        fig = plt.figure()
        plt.plot(history['acc'])
        plt.plot(history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        if save == True:
            plt.savefig(case + 'accuracy.png')
            plt.gcf().clear()
        else:
            plt.show()
        plt.close(fig)

    # summarize history for loss
    if 'loss' in metrics:
        fig = plt.figure()
        plt.plot(history['loss'])
        plt.plot(history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        #plt.ylim(1e-3, 1e-2)
        plt.yscale("log")
        plt.legend(['train', 'val'], loc='upper left')
        if save == True:
            plt.savefig(case + 'loss.png')
            plt.gcf().clear()
        else:
            plt.show()
        plt.close(fig)
        
def step_decay(epoch):
	initial_lrate = 0.005
	drop = 0.5
	epochs_drop = 10.0
	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
	return lrate
 
def generator(folder1,folder2):
    for x,y in zip(folder1,folder2):
        yield x,y
          
def saveFeatures(param, max_label, batch_size, phase, save_features, feature_extractor, classifier, features_file, labels_file, train_split, test_split):
        #data_folder = '/ssd_drive/data/'
        data_folder = '/ssd_drive/MultiCam_OF2/'
        mean_file = '/ssd_drive/flow_mean.mat'
        L = 10 
        
        class0 = 'Falls'
        class1 = 'NotFalls'
        
        # TRANSFORMACIONES
        flip = False
        crops = False
        rotate = False
        translate = False
      
        #i, j = 0, 0
        # substract mean
        d = sio.loadmat(mean_file)
        flow_mean = d['image_mean']
        num_features = 4096
        
        # ===============================================
        print('Starting the loading')
        
        mult = 1
        if flip:
            mult = 2
        if flip and crops:
            mult = 10
        if flip and crops and rotate and translate:
            mult = 39
        
        #size = 0
        #folders, classes = [], []
        #fall_videos = [f for f in os.listdir(data_folder + class0) if os.path.isdir(os.path.join(data_folder + class0, f))]
        #fall_videos.sort()
        #for fall_video in fall_videos:
        #    x_images = glob.glob(data_folder + class0 + '/' + fall_video + '/flow_x*.jpg')
        #    if int(len(x_images)) >= 10:
        #        folders.append(data_folder + class0 + '/' + fall_video)
        #        classes.append(0)

        #not_fall_videos = [f for f in os.listdir(data_folder + class1) if os.path.isdir(os.path.join(data_folder + class1, f))]
        #not_fall_videos.sort()
        #for not_fall_video in not_fall_videos:
        #    if int(len(x_images)) >= 10:
        #        x_images = glob.glob(data_folder + class1 + '/' + not_fall_video + '/flow_x*.jpg')
        #        folders.append(data_folder + class1 + '/' + not_fall_video)
        #        classes.append(1)

        h5features = h5py.File(features_file,'w')
        h5labels = h5py.File(labels_file,'w')
        
        
        # Get all folders and classes
        i = 0
        idx = 0   
       
        #nb_total_stacks = 0
        #for folder in folders:
        #    x_images = glob.glob(folder + '/flow_x*.jpg')
        #    nb_total_stacks += int(len(x_images))-L+1
        #size_test *= mult 
        
        #dataset_features_train = h5features.create_dataset('train', shape=(size_train, num_features), dtype='float64')
        #dataset_features_test = h5features.create_dataset('test', shape=(size_test, num_features), dtype='float64')
        #dataset_labels_train = h5labels.create_dataset('train', shape=(size_train, 1), dtype='float64')  
        #dataset_labels_test = h5labels.create_dataset('test', shape=(size_test, 1), dtype='float64')  
        #print(size_train, size_test)
        
        #a = np.zeros((20,10))
        #b = range(20)
        #nb_stacks=20-10+1
        #for i in range(len(b)):
        #     for s in list(reversed(range(min(10,i+1)))):
        #         print(i,s)
        #         a[i-s,s] = b[i]
        #ind_fold = 0
        
        
        fall_videos = np.zeros((24,2), dtype=np.int)
        i = 0
        while i < 3:
            fall_videos[i,:] = [i*7, i*7+7]
            i += 1
        fall_videos[i,:] = [i*7, i*7+14]
        i += 1
        while i < 23:
            fall_videos[i,:] = [i*7, i*7+7]
            i += 1
        fall_videos[i,:] = [i*7, i*7]
        
        not_fall_videos = np.zeros((24,2), dtype=np.int)
        i = 0
        while i < 23:
            not_fall_videos[i,:] = [i*7, i*7+14]
            i += 1
        not_fall_videos[i,:] = [i*7, i*7+7]
        
        stages = []
        for i in [24] + range(1,24):
            stages.append('chute{:02}'.format(i))
        black = np.zeros((224,224))
        idx_falls, idx_nofalls = 0, 0
        for stage, nb_stage in zip(stages, range(len(stages))):
            print(nb_stage, stage)
            h5features.create_group(stage)
            h5labels.create_group(stage)
            
            #h5features[stage].create_group('augmented')
            h5features[stage].create_group('not_augmented')
            #h5labels[stage].create_group('augmented')
            h5labels[stage].create_group('not_augmented')
            
            cameras = glob.glob(data_folder + stage + '/cam*')
            cameras.sort()
            for camera, nb_camera in zip(cameras, range(1, len(cameras)+1)):
                print('Cam {}'.format(nb_camera))
                #h5features[stage]['augmented'].create_group('cam{}'.format(nb_camera))
                h5features[stage]['not_augmented'].create_group('cam{}'.format(nb_camera))
                #h5labels[stage]['augmented'].create_group('cam{}'.format(nb_camera))
                h5labels[stage]['not_augmented'].create_group('cam{}'.format(nb_camera))
                #not_falls = [f for f in os.listdir(data_folder + stage + '/cam{}/NotFalls/'.format(nb_camera)) if os.path.isdir(os.path.join(data_folder + stage + '/cam{}/NotFalls/'.format(nb_camera), f))]
                not_falls = glob.glob(camera + '/NotFalls/notfall*'.format(nb_camera))
                not_falls.sort()
                h5features.close()
                h5labels.close()
                h5features = h5py.File(features_file,'a')
                h5labels = h5py.File(labels_file,'a')
                for not_fall in not_falls:
                    print(not_fall)
                    label = 1
                    x_images = glob.glob(not_fall + '/flow_x*.jpg')
                    x_images.sort()
                    y_images = glob.glob(not_fall + '/flow_x*.jpg')
                    y_images.sort()
                    nb_stacks = int(len(x_images))-L+1
                    
                    #features_aug_notfall = h5features[stage]['augmented']['cam{}'.format(nb_camera)].create_dataset('notfall{:04}'.format(idx_nofalls), shape=(nb_stacks*2, num_features), dtype='float64')
                    features_notfall = h5features[stage]['not_augmented']['cam{}'.format(nb_camera)].create_dataset('notfall{:04}'.format(idx_nofalls), shape=(nb_stacks, num_features), dtype='float64')
                    #labels_aug_notfall = h5labels[stage]['augmented']['cam{}'.format(nb_camera)].create_dataset('notfall{:04}'.format(idx_nofalls), shape=(nb_stacks*2, 1), dtype='float64')
                    labels_notfall = h5labels[stage]['not_augmented']['cam{}'.format(nb_camera)].create_dataset('notfall{:04}'.format(idx_nofalls), shape=(nb_stacks, 1), dtype='float64')
                    idx_nofalls += 1
                    
                    if stage == 'chute24' or stage == 'chute23':
                        flow = np.zeros(shape=(224,224,2*L,nb_stacks), dtype=np.float64)               
                        gen = generator(x_images,y_images)
                        for i in range(len(x_images)):
                            flow_x_file, flow_y_file = gen.next()
                            img_x = cv2.imread(flow_x_file, cv2.IMREAD_GRAYSCALE)
                            img_y = cv2.imread(flow_y_file, cv2.IMREAD_GRAYSCALE)
                            for s in list(reversed(range(min(10,i+1)))):
                                if i-s < nb_stacks:
                                    flow[:,:,2*s,  i-s] = img_x
                                    flow[:,:,2*s+1,i-s] = img_y
                            del img_x,img_y
                            gc.collect()
                        flow = flow - np.tile(flow_mean[...,np.newaxis], (1, 1, 1, flow.shape[3]))
                        flow = np.transpose(flow, (3, 2, 0, 1)) 
                        predictions = np.zeros((flow.shape[0], num_features), dtype=np.float64)
                        truth = np.zeros((flow.shape[0], 1), dtype=np.float64)
                        for i in range(flow.shape[0]):
                            prediction = feature_extractor.predict(np.expand_dims(flow[i, ...],0))
                            predictions[i, ...] = prediction
                            truth[i] = label
                        features_notfall[:,:] = predictions
                        labels_notfall[:,:] = truth
                        del predictions, truth, flow, features_notfall, labels_notfall
                        gc.collect()
                        
                        #flow_aug = np.zeros(shape=(224,224,2*L,nb_stacks*2), dtype=np.float64)
                        #gen = generator(x_images,y_images)
                        #for i in range(len(x_images)):
                        #    flow_x_file, flow_y_file = gen.next()
                        #    img_x = cv2.imread(flow_x_file, cv2.IMREAD_GRAYSCALE)
                        #    img_y = cv2.imread(flow_y_file, cv2.IMREAD_GRAYSCALE)
                        #    flip_x = 255 - img_x[:, ::-1]
                        #    flip_y = img_y[:, ::-1]
                        #    for s in list(reversed(range(min(10,i+1)))):
                        #        if i-s < nb_stacks:
                        #            flow_aug[:,:,2*s,  i-s] = img_x
                        #            flow_aug[:,:,2*s+1,i-s] = img_y
                        #            flow_aug[:,:,2*s,  i-s+nb_stacks] = flip_x
                        #            flow_aug[:,:,2*s+1,i-s+nb_stacks] = flip_y
                        #    del img_x,img_y,flip_x,flip_y
                        #    gc.collect()
                        #flow_aug = flow_aug - np.tile(flow_mean[...,np.newaxis], (1, 1, 1, flow_aug.shape[3]))
                        #flow_aug = np.transpose(flow_aug, (3, 2, 0, 1))
                        #predictions = np.zeros((flow_aug.shape[0], num_features), dtype=np.float64)
                        #truth = np.zeros((flow_aug.shape[0], 1), dtype=np.float64)
                        #for i in range(flow_aug.shape[0]):
                        #    prediction = feature_extractor.predict(np.expand_dims(flow_aug[i, ...],0))
                        #    predictions[i, ...] = prediction
                        #    truth[i] = label
                        #features_aug_notfall[:,:] = predictions
                        #labels_aug_notfall[:,:] = truth
                        #del predictions, truth, flow_aug, features_aug_notfall, labels_aug_notfall,
                        #gc.collect()
                    # NOT CHUTE24 ==================
                    else:
                        flow = np.zeros(shape=(224,224,2*L,nb_stacks), dtype=np.float64)
                        #flow_aug = np.zeros(shape=(224,224,2*L,nb_stacks*2), dtype=np.float64)
                        gen = generator(x_images,y_images)
                        for i in range(len(x_images)):
                            flow_x_file, flow_y_file = gen.next()
                            img_x = cv2.imread(flow_x_file, cv2.IMREAD_GRAYSCALE)
                            img_y = cv2.imread(flow_y_file, cv2.IMREAD_GRAYSCALE)
                            flip_x = 255 - img_x[:, ::-1]
                            flip_y = img_y[:, ::-1]
                            for s in list(reversed(range(min(10,i+1)))):
                                if i-s < nb_stacks:
                                    flow[:,:,2*s,  i-s] = img_x
                                    flow[:,:,2*s+1,i-s] = img_y
                                    
                                    #flow_aug[:,:,2*s,  i-s] = img_x
                                    #flow_aug[:,:,2*s+1,i-s] = img_y
                                    #flow_aug[:,:,2*s,  i-s+nb_stacks] = flip_x
                                    #flow_aug[:,:,2*s+1,i-s+nb_stacks] = flip_y
                            del img_x,img_y,flip_x,flip_y
                            gc.collect()
                        flow = flow - np.tile(flow_mean[...,np.newaxis], (1, 1, 1, flow.shape[3]))
                        flow = np.transpose(flow, (3, 2, 0, 1)) 
                        predictions = np.zeros((flow.shape[0], num_features), dtype=np.float64)
                        truth = np.zeros((flow.shape[0], 1), dtype=np.float64)
                        for i in range(flow.shape[0]):
                            prediction = feature_extractor.predict(np.expand_dims(flow[i, ...],0))
                            predictions[i, ...] = prediction
                            truth[i] = label
                        features_notfall[:,:] = predictions
                        labels_notfall[:,:] = truth
                        del predictions, truth, flow, features_notfall, labels_notfall
                        gc.collect()
                        
                        #flow_aug = flow_aug - np.tile(flow_mean[...,np.newaxis], (1, 1, 1, flow_aug.shape[3]))
                        #flow_aug = np.transpose(flow_aug, (3, 2, 0, 1))
                        #predictions = np.zeros((flow_aug.shape[0], num_features), dtype=np.float64)
                        #truth = np.zeros((flow_aug.shape[0], 1), dtype=np.float64)
                        #for i in range(flow_aug.shape[0]):
                        #    prediction = feature_extractor.predict(np.expand_dims(flow_aug[i, ...],0))
                        #    predictions[i, ...] = prediction
                        #    truth[i] = label
                        #features_aug_notfall[:,:] = predictions
                        #labels_aug_notfall[:,:] = truth
                        #del predictions, truth, flow_aug, features_aug_notfall, labels_aug_notfall,
                        #gc.collect()
                    del x_images, y_images, nb_stacks
                    gc.collect()
                
                if stage == 'chute24':
                    idx += 2
                    continue
                falls = glob.glob(camera + '/Falls/fall*'.format(nb_camera))
                falls.sort()
                h5features.close()
                h5labels.close()
                h5features = h5py.File(features_file,'a')
                h5labels = h5py.File(labels_file,'a')
                for fall in falls:     
                    print(fall)
                    label = 0
                    x_images = glob.glob(fall + '/flow_x*.jpg')
                    x_images.sort()
                    y_images = glob.glob(fall + '/flow_y*.jpg')
                    y_images.sort()
                    nb_stacks = int(len(x_images))-L+1
                    
                    #features_aug_fall = h5features[stage]['augmented']['cam{}'.format(nb_camera)].create_dataset('fall{:04}'.format(idx_falls), shape=(nb_stacks*2, num_features), dtype='float64')
                    features_fall = h5features[stage]['not_augmented']['cam{}'.format(nb_camera)].create_dataset('fall{:04}'.format(idx_falls), shape=(nb_stacks, num_features), dtype='float64')
                    #labels_aug_fall = h5labels[stage]['augmented']['cam{}'.format(nb_camera)].create_dataset('fall{:04}'.format(idx_falls), shape=(nb_stacks*2, 1), dtype='float64')
                    labels_fall = h5labels[stage]['not_augmented']['cam{}'.format(nb_camera)].create_dataset('fall{:04}'.format(idx_falls), shape=(nb_stacks, 1), dtype='float64')
                    idx_falls += 1
                    flow = np.zeros(shape=(224,224,2*L,nb_stacks), dtype=np.float64)
                    #flow_aug = np.zeros(shape=(224,224,2*L,nb_stacks*2), dtype=np.float64)
                    
                    gen = generator(x_images,y_images)
                    for i in range(len(x_images)):
                        flow_x_file, flow_y_file = gen.next()
                        img_x = cv2.imread(flow_x_file, cv2.IMREAD_GRAYSCALE)
                        img_y = cv2.imread(flow_y_file, cv2.IMREAD_GRAYSCALE)
                        flip_x = 255 - img_x[:, ::-1]
                        flip_y = img_y[:, ::-1]
                        for s in list(reversed(range(min(10,i+1)))):
                            if i-s < nb_stacks:
                                flow[:,:,2*s,  i-s] = img_x
                                flow[:,:,2*s+1,i-s] = img_y
                                
                                #flow_aug[:,:,2*s,  i-s] = img_x
                                #flow_aug[:,:,2*s+1,i-s] = img_y
                                #flow_aug[:,:,2*s,  i-s+nb_stacks] = flip_x
                                #flow_aug[:,:,2*s+1,i-s+nb_stacks] = flip_y
                        del img_x,img_y,flip_x,flip_y
                        gc.collect()
                    flow = flow - np.tile(flow_mean[...,np.newaxis], (1, 1, 1, flow.shape[3]))
                    flow = np.transpose(flow, (3, 2, 0, 1)) 
                    predictions = np.zeros((flow.shape[0], num_features), dtype=np.float64)
                    truth = np.zeros((flow.shape[0], 1), dtype=np.float64)
                    for i in range(flow.shape[0]):
                        prediction = feature_extractor.predict(np.expand_dims(flow[i, ...],0))
                        predictions[i, ...] = prediction
                        truth[i] = label
                    features_fall[:,:] = predictions
                    labels_fall[:,:] = truth
                    del predictions, truth, flow, features_fall, labels_fall
                    gc.collect()
                        
                    #flow_aug = flow_aug - np.tile(flow_mean[...,np.newaxis], (1, 1, 1, flow_aug.shape[3]))
                    #flow_aug = np.transpose(flow_aug, (3, 2, 0, 1))
                    #predictions = np.zeros((flow_aug.shape[0], num_features), dtype=np.float64)
                    #truth = np.zeros((flow_aug.shape[0], 1), dtype=np.float64)
                    #for i in range(flow_aug.shape[0]):
                    #    prediction = feature_extractor.predict(np.expand_dims(flow_aug[i, ...],0))
                    #    predictions[i, ...] = prediction
                    #    truth[i] = label
                    #features_aug_fall[:,:] = predictions
                    #labels_aug_fall[:,:] = truth
                    
                    #del predictions, truth, features_fall, labels_aug_fall, labels_fall, flow_aug, features_aug_fall, 
                    #gc.collect()
                   
        h5features.close()
        h5labels.close()
        sys.exit()         
        for folder, label in zip(folders, classes):
            print(folder)
            h5features.create_group(folder)
            h5labels.create_group(folder)
            #os.makedirs('/home/anunez/imagenes/' + folder)
            x_images = glob.glob(folder + '/flow_x*.jpg')
            x_images.sort()
            y_images = glob.glob(folder + '/flow_y*.jpg')
            y_images.sort()
            nb_stacks = int(len(x_images))-(2*L)+1
        
            flow = np.zeros(shape=(224,224,2*L,nb_stacks), dtype=np.float64)
            flow_aug = np.zeros(shape=(224,224,2*L,nb_stacks*mult), dtype=np.float64)
            
            gen = generator(x_images,y_images)
            for i in range(len(x_images)):
                flow_x_file, flow_y_file = gen.next()
                img_x = cv2.imread(flow_x_file, cv2.IMREAD_GRAYSCALE)
                img_y = cv2.imread(flow_y_file, cv2.IMREAD_GRAYSCALE)
                flip_x = 255 - img_x[:, ::-1]
                flip_y = img_y[:, ::-1]
                for s in list(reversed(range(min(10,i+1)))):
                    flow[:,:,2*s,  i-s] = img_x
                    flow[:,:,2*s+1,i-s] = img_y
                    
                    flow_aug[:,:,2*s,  i-s] = img_x
                    flow_aug[:,:,2*s+1,i-s] = img_y
                    flow_aug[:,:,2*s,  i-s+nb_stacks] = flip_x
                    flow_aug[:,:,2*s+1,i-s+nb_stacks] = flip_y
                del img_x,img_y,flip_x,flip_y
                gc.collect()
            
       
            flow = flow - np.tile(flow_mean[...,np.newaxis], (1, 1, 1, flow.shape[3]))
            flow = np.transpose(flow, (3, 2, 0, 1)) 
            predictions = np.zeros((flow.shape[0], num_features), dtype=np.float64)
            truth = np.zeros((flow.shape[0], 1), dtype=np.float64)
            #print('flow shape {}'.format(flow.shape[0]))
            #print(flow.shape)
            for i in range(flow.shape[0]):
                prediction = feature_extractor.predict(np.expand_dims(flow[i, ...],0))
                predictions[i, ...] = prediction
                truth[i] = label
            dataset_features_train[cont_train:cont_train+flow.shape[0], :] = predictions
            dataset_labels_train[cont_train:cont_train+flow.shape[0]] = truth
            cont_train += flow.shape[0]
                
            flow_aug = flow_aug - np.tile(flow_mean[...,np.newaxis], (1, 1, 1, flow_aug.shape[3]))
            flow_aug = np.transpose(flow_aug, (3, 2, 0, 1))
            predictions = np.zeros((flow.shape[0], num_features), dtype=np.float64)
            truth = np.zeros((flow_aug.shape[0], 1), dtype=np.float64)
            for i in range(flow_aug.shape[0]):
                prediction = feature_extractor.predict(np.expand_dims(flow_aug[i, ...],0))
                predictions[i, ...] = prediction
                truth[i] = label
            dataset_features_train[cont_train:cont_train+flow.shape[0], :] = predictions
            dataset_labels_train[cont_train:cont_train+flow.shape[0]] = truth
            cont_train += flow.shape[0]
            #print(cont, flow.shape[0])
            #features[cont:cont+flow.shape[0], :] = predictions
            #all_labels[cont:cont+flow.shape[0], :] = truth
            #cont += flow.shape[0]
            #dataset_features2[...] = predictions2
            #dataset_labels2[...] = truth
            del flow, predictions, truth
            gc.collect()
            
            #is_testing = False
        #if p == 0:
        #    np.save(features_file + '2_train.npy', features)
        #    np.save(labels_file + '2_train.npy', all_labels)
        #else:
        #    np.save(features_file + '2_test.npy', features)
        #    np.save(labels_file + '2_test.npy', all_labels)
        h5features.close()
        h5labels.close()
            

def main(learning_rate, batch_size, dropout, batch_norm, weight_0, weight_1, nb_neurons, exp, model_file, weights_file): 
    best_model = 'best_weights/best_weights_{}.hdf5'.format(exp)
    print(exp)
    num_features = 4096
    
    with open(parameter_file) as data_file:    
        param = json.load(data_file)
        
    # VGG16 =====================================================
    model = Sequential()
    
    model.add(ZeroPadding2D((1, 1), input_shape=(param['input_channels'], param['input_width'], param['input_height'])))
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    
    model.add(Flatten())
    model.add(Dense(4096, name='fc6', init='glorot_uniform'))
   
    extracted_features = Input(shape=(4096,), dtype='float32', name='input')
    #x = Dense(4096, activation='relu', name='fc1')(extracted_features)
    #if batch_norm:
    #    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(extracted_features)
    #    x = ELU(alpha=1.0)(x)
    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(extracted_features)
    x = Activation('relu')(x)
    x = Dropout(0.9)(x)
    x = Dense(nb_neurons, name='fc2', init='glorot_uniform')(x)
    #if batch_norm:
    #    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)
    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)
    x = Activation('relu')(x)
    x = Dropout(0.8)(x)
    x = Dense(1, name='predictions', init='glorot_uniform')(x)
    x = Activation('sigmoid')(x)
    classifier = Model(input=extracted_features, output=x, name='classifier')

    layerskeras = ['block1_conv1', 'block1_conv2', 'block2_conv1', 'block2_conv2', 'block3_conv1', 'block3_conv2', 'block3_conv3', 'block4_conv1', 'block4_conv2', 'block4_conv3', 'block5_conv1', 'block5_conv2', 'block5_conv3', 'fc1', 'fc2', 'predictions']
    layerscaffe = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'conv3_3', 'conv4_1', 'conv4_2', 'conv4_3', 'conv5_1', 'conv5_2', 'conv5_3', 'fc6', 'fc7', 'fc8']
    i = 0
    h5 = h5py.File('/home/anunez/project/caffedata.h5')
    
    layer_dict = dict([(layer.name, layer) for layer in model.layers])

    for layer in layerscaffe[:-3]:
        w2, b2 = h5['data'][layer]['0'], h5['data'][layer]['1']
        w2 = np.transpose(np.asarray(w2), (0,1,2,3))
        w2 = w2[:, :, ::-1, ::-1]
        b2 = np.asarray(b2)
        #model.get_layer(layerskeras[i]).W.set_value(w2)
        #model.get_layer(layerskeras[i]).b.set_value(b2)
        layer_dict[layer].W.set_value(w2)
        layer_dict[layer].b.set_value(b2)
        i += 1
        
    layer = layerscaffe[-3]
    w2, b2 = h5['data'][layer]['0'], h5['data'][layer]['1']
    w2 = np.transpose(np.asarray(w2), (1,0))
    b2 = np.asarray(b2)
    #model.get_layer(layerskeras[i]).W.set_value(w2)
    #model.get_layer(layerskeras[i]).b.set_value(b2)
    layer_dict[layer].W.set_value(w2)
    layer_dict[layer].b.set_value(b2)
    i += 1
    
    copy_dense_weights = False
    if copy_dense_weights:
        print('Copiando pesos de capas densas')
        #for layer in layerscaffe[-2:]:
        layer = layerscaffe[-2]
        w2, b2 = h5['data'][layer]['0'], h5['data'][layer]['1']      
        w2 = np.transpose(w2,(1,0))
        b2 = np.asarray(b2)
        print(layerskeras[i])
        classifier.get_layer('fc2').W.set_value(w2)
        classifier.get_layer('fc2').b.set_value(b2)
        i += 1
    for layer in classifier.layers:
        layer.trainable = True
    w,b = classifier.get_layer('fc2').get_weights()
    #print(np.allclose(w, w))

    #classifier.load_weights(best_model)
    #plot_model(model, to_file='model.png', show_shapes=False, show_layer_names=True)
    #plot_model(classifier, to_file='classifier.png', show_shapes=False, show_layer_names=True)
    adam = Adam(lr=learning_rate, beta_1=param['beta_1'], beta_2=param['beta_2'], epsilon=param['adam_eps'], decay=param['decay'])
    #sgd = SGD(lr=learning_rate, momentum=0.0, decay=param['decay'], nesterov=False)
    #if True == 'adam':
    model.compile(optimizer=adam, loss='categorical_crossentropy',  metrics=param['metrics'])
    #elif optimizer == 'sgd':
    #    model.compile(optimizer=sgd, loss='categorical_crossentropy',  metrics=param['metrics'])
    c = ModelCheckpoint(filepath=best_model, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto')
    #e = EarlyStopping(monitor='loss', min_delta=0, patience=0, verbose=0, mode='auto')
    #r = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
    #l = LearningRateScheduler(step_decay)
    #pb = printbatch()
    #callbacks = [c]

    amount_of_data, parts = [],[]
    
    #validationGenerator = getDataChinese(param, param['classes'], param['batch_size'], training_parts, amount_of_data, training_parts, validation_parts, test_parts)
    
    if not os.path.isdir('train_history_results'):
        os.mkdir('train_history_results')
    #model.load_weights(best_model)
        
    # =============================================================================================================
    # FEATURE EXTRACTION
    # =============================================================================================================
    #features_file = '/home/anunez/project/features/features_multicam_final.h5'
    #labels_file = '/home/anunez/project/labels/labels_multicam_final.h5'
    features_file = '/home/anunez/project/features/features_multicam_final2.h5'
    labels_file = '/home/anunez/project/labels/labels_multicam_final2.h5'
    features_file2 = '/home/anunez/project/features/features_urfall_final3.h5'
    labels_file2 = '/home/anunez/project/labels/labels_urfall_final3.h5'
    features_file3 = '/home/anunez/project/features/features_fdd_final3.h5'
    labels_file3 = '/home/anunez/project/labels/labels_fdd_final3.h5'
    #features_file = '/home/anunez/project/features/features_multicam.h5'
    #labels_file = '/home/anunez/project/labels/labels_multicam.h5'
    save_features = False
    if save_features:
        print('Saving features')
        #saveFeatures(param, param['classes'], param['batch_size'], 0, amount_of_data, parts, save_features, model, model2, classifier, features_file + '1.h5', labels_file + '1.h5', train_splits[0], test_splits[0])
        saveFeatures(param, param['classes'], param['batch_size'], 0, amount_of_data, parts, save_features, model, classifier, features_file, labels_file, [], [])
        #saveFeatures(param, param['classes'], param['batch_size'], 0, amount_of_data, parts, save_features, model, model2, classifier, features_file + '3.h5', labels_file + '3.h5', train_splits[2], test_splits[2])
        #saveFeatures(param, param['classes'], param['batch_size'], 1, amount_of_data, parts, save_features, model, model2, classifier, features_file + 'validation.h5', labels_file + 'validation.h5', features_file2 + 'validation.h5', labels_file2 + 'validation.h5')
        #saveFeatures(param, param['classes'], param['batch_size'], 2, amount_of_data, parts, save_features, model, model2, classifier, features_file + 'testing.h5', labels_file + 'testing.h5', features_file2 + 'testing.h5', labels_file2 + 'testing.h5')   
        print('Feature extraction finished')
    #sys.exit()
    # =============================================================================================================
    # TRAINING
    # =============================================================================================================
    #if optimizer == 'adam':
    adam = Adam(lr=learning_rate, beta_1=param['beta_1'], beta_2=param['beta_2'], epsilon=param['adam_eps'], decay=param['decay'])
    
    #elif True or optimizer == 'sgd':
    #    classifier.compile(optimizer=sgd, loss='categorical_crossentropy',  metrics=param['metrics'])
    do_training = True
    do_training_with_features = True
    do_training_normal = not do_training_with_features
    
    compute_metrics = False
    compute_roc_curve = False
    threshold = 0.5
    e = EarlyStopping(monitor='val_loss', min_delta=0, patience=100, verbose=0, mode='auto')

    if do_training:
        if do_training_with_features:
            h5features = h5py.File(features_file, 'r')
            h5labels = h5py.File(labels_file, 'r')
            h5features2 = h5py.File(features_file2, 'r')
            h5labels2 = h5py.File(labels_file2, 'r')
            h5features3 = h5py.File(features_file3, 'r')
            h5labels3 = h5py.File(labels_file3, 'r')
            stages = []
            for i in range(1,25):
                stages.append('chute{:02}'.format(i))
            use_aug = False
            aug = 'not_augmented'
            if use_aug:
                aug =  'augmented'
            cams_x = []
            cams_y = []
            for stage, nb_stage in zip(stages, range(len(stages))):   
                for cam, nb_cam in zip(h5features[stage][aug].keys(), range(8)):
                    temp_x = []
                    temp_y = []
                    for key in h5features[stage][aug][cam].keys():
                        #print(h5features[stage][aug][cam][key].shape)
                        temp_x.append(np.asarray(h5features[stage][aug][cam][key]))
                        temp_y.append(np.asarray(h5labels[stage][aug][cam][key]))
                    #temp_x = np.asarray(temp_x)
                    #temp_y = np.asarray(temp_y)
                    temp_x = np.concatenate(temp_x,axis=0)
                    temp_y = np.concatenate(temp_y,axis=0)
                    if nb_stage == 0:
                        cams_x.append(temp_x)
                        cams_y.append(temp_y)
                    else:
                        cams_x[nb_cam] = np.concatenate([cams_x[nb_cam], temp_x], axis=0)
                        cams_y[nb_cam] = np.concatenate([cams_y[nb_cam], temp_y], axis=0)
                    
            sensitivities_general = []
            specificities_general = []
            sensitivities_urfall = []
            specificities_urfall = []
            sensitivities_multicam = []
            specificities_multicam = []
            sensitivities_fdd = []
            specificities_fdd = []
            
            X = np.asarray(np.concatenate(cams_x,axis=0))
            _y = np.asarray(np.concatenate(cams_y,axis=0))
            X2 = np.asarray(h5features2['features']) #fdd
            _y2 = np.asarray(h5labels2['labels'])
            X3 = np.asarray(h5features3['train']) #fdd
            _y3 = np.asarray(h5labels3['train'])
            
            size_0 = np.asarray(np.where(_y2==0)[0]).shape[0]
            size_1 = np.asarray(np.where(_y2==1)[0]).shape[0]
            
            all0_1 = np.asarray(np.where(_y==0)[0])
            all1_1 = np.asarray(np.where(_y==1)[0])
            all0_2 = np.asarray(np.where(_y2==0)[0])
            all1_2 = np.asarray(np.where(_y2==1)[0])
            all0_3 = np.asarray(np.where(_y3==0)[0])
            all1_3 = np.asarray(np.where(_y3==1)[0])
            print(all0_1.shape[0], all1_1.shape[0])
            print(all0_2.shape[0], all1_2.shape[0])
            print(all0_3.shape[0], all1_3.shape[0])
            all0_1 = np.random.choice(all0_1, size_0, replace=False)
            all1_1 = np.random.choice(all1_1, size_0, replace=False)
            all0_2 = np.random.choice(all0_2, size_0, replace=False)
            all1_2 = np.random.choice(all1_2, size_0, replace=False)
            all0_3 = np.random.choice(all0_3, size_0, replace=False)
            all1_3 = np.random.choice(all1_3, size_0, replace=False)
            
            
      
            
            slice_size = size_0/5
            # LEAVE-ONE-OUT
            for fold in range(5):
                #print('='*30)
                #print('LEAVE-ONE-OUT STEP {}/8'.format(cam))
                #print('='*30)
                #test_x = cams_x[cam]
                #test_y = cams_y[cam]
                #train_x = cams_x[0:cam] + cams_x[cam+1:]
                #train_y = cams_y[0:cam] + cams_y[cam+1:]
                
                
                #sss = StratifiedShuffleSplit(n_splits=2, test_size=0.2, random_state=777)
                #sss.get_n_splits(train_x, train_y)
                #train_index, test_index = [], []
                #for a, b in sss.split(train_x, train_y):
                #    train_index = a
                #    test_index = b                
                #train_index = np.asarray(train_index)
                #test_index = np.asarray(test_index)
                #train_x, test_x = train_x[train_index], train_x[test_index]
                #train_y, test_y = train_y[train_index], train_y[test_index]
                #
                print(all0_1.shape[0], fold*slice_size, (fold+1)*slice_size)
                print(all0_1[0:fold*slice_size].shape, all0_1[(fold+1)*slice_size:].shape)
                temp = np.concatenate((
                                    np.hstack((
                                                    all0_1[0:fold*slice_size],all0_1[(fold+1)*slice_size:])),
                                    np.hstack((
                                                    all1_1[0:fold*slice_size],all1_1[(fold+1)*slice_size:]))))
                X1_train = X[temp]
                _y1_train = _y[temp]
                temp = np.concatenate((
                                    np.hstack((
                                                    all0_2[0:fold*slice_size],all0_2[(fold+1)*slice_size:])),
                                    np.hstack((
                                                    all1_2[0:fold*slice_size],all1_2[(fold+1)*slice_size:]))))
                X2_train = X2[temp]
                _y2_train = _y2[temp]
                temp = np.concatenate((
                                    np.hstack((
                                                    all0_3[0:fold*slice_size],all0_3[(fold+1)*slice_size:])),
                                    np.hstack((
                                                    all1_3[0:fold*slice_size],all1_3[(fold+1)*slice_size:]))))
                X3_train = X3[temp]
                _y3_train = _y3[temp]
                # TEST
                temp = np.concatenate((
                                    np.hstack((
                                                    all0_1[fold*slice_size:(fold+1)*slice_size])),
                                    np.hstack((
                                                    all1_1[fold*slice_size:(fold+1)*slice_size]))))
                X1_test = X[temp]
                _y1_test = _y[temp]
                temp = np.concatenate((
                                    np.hstack((
                                                    all0_2[fold*slice_size:(fold+1)*slice_size])),
                                    np.hstack((
                                                    all1_2[fold*slice_size:(fold+1)*slice_size]))))
                X2_test = X2[temp]
                _y2_test = _y2[temp]
                temp = np.concatenate((
                                    np.hstack((
                                                    all0_3[fold*slice_size:(fold+1)*slice_size])),
                                    np.hstack((
                                                    all1_3[fold*slice_size:(fold+1)*slice_size]))))
                X3_test = X3[temp]
                _y3_test = _y3[temp]
                
                #print(_y.shape, _y2.shape, _y3.shape)
    
                recortar = False
                if recortar:
                    all1_1 = np.random.choice(all1_1, size_0, replace=False)
                    allin = np.concatenate((all0_1.flatten(),all1_1.flatten()))
                    allin.sort()
                    X = X[allin,...]
                    _y = _y[allin]
                    
                    all1_2 = np.random.choice(all1_2, size_0, replace=False)
                    allin = np.concatenate((all0_2.flatten(),all1_2.flatten()))
                    allin.sort()
                    X2 = X2[allin,...]
                    _y2 = _y2[allin]
                    
                    all1_3 = np.random.choice(all1_3, size_0, replace=False)
                    allin = np.concatenate((all0_3.flatten(),all1_3.flatten()))
                    allin.sort()
                    X3 = X3[allin,...]
                    _y3 = _y3[allin]
                    
                    
                #sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=777)
                #sss.get_n_splits(X, _y)
                #train_index, test_index = [], []
                #for a, b in sss.split(X, _y):
                #    train_index = a
                #    test_index = b                
                #train_index = np.asarray(train_index)
                #test_index = np.asarray(test_index)
                #X1_train, X1_test = X[train_index], X[test_index]
                #_y1_train, _y1_test = _y[train_index], _y[test_index]
                #
                #sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=777)
                #sss.get_n_splits(X2, _y2)
                #train_index, test_index = [], []
                #for a, b in sss.split(X2, _y2):
                #    train_index = a
                #    test_index = b                
                #train_index = np.asarray(train_index)
                #test_index = np.asarray(test_index)
                #X2_train, X2_test = X2[train_index], X2[test_index]
                #_y2_train, _y2_test = _y2[train_index], _y2[test_index]
                #
                #sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=777)
                #sss.get_n_splits(X3, _y3)
                #train_index, test_index = [], []
                #for a, b in sss.split(X3, _y3):
                #    train_index = a
                #    test_index = b                
                #train_index = np.asarray(train_index)
                #test_index = np.asarray(test_index)
                #X3_train, X3_test = X3[train_index], X3[test_index]
                #_y3_train, _y3_test = _y3[train_index], _y3[test_index]
                #
                X_train = np.concatenate((X1_train, X2_train, X3_train), axis=0)
                _y_train = np.concatenate((_y1_train, _y2_train, _y3_train), axis=0)
                
                X_test = np.concatenate((X1_test, X2_test, X3_test), axis=0)
                _y_test = np.concatenate((_y1_test, _y2_test, _y3_test), axis=0)
                                
                
                all0 = np.asarray(np.where(_y==0)[0])
                all1 = np.asarray(np.where(_y==1)[0])
                print('Train Falls/NoFalls in dataset: {}/{}, total data: {}'.format(len(all0), len(all1), X_train.shape[0]))
                all0 = np.asarray(np.where(_y2==0)[0])
                all1 = np.asarray(np.where(_y2==1)[0])
                print('Test Falls/NoFalls in dataset: {}/{}, total data: {}'.format(len(all0), len(all1), X_test.shape[0]))
                
                extracted_features = Input(shape=(4096,), dtype='float32', name='input')
                #x = ELU(alpha=1.0)(extracted_features)
                x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(extracted_features)
                x = Activation('relu')(x)
                x = Dropout(0.9)(x)
                x = Dense(nb_neurons, name='fc2', init='glorot_uniform')(x)
                #x = ELU(alpha=1.0)(x)
                x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)
                x = Activation('relu')(x)
                x = Dropout(0.8)(x)
                x = Dense(1, name='predictions', init='glorot_uniform')(x)
                x = Activation('sigmoid')(x)
                classifier = Model(input=extracted_features, output=x, name='classifier')
                classifier.compile(optimizer=adam, loss='binary_crossentropy',  metrics=param['metrics'])
                class_weight = {0:weight_0, 1:weight_1}
                #print('Data after stratify: {} + {} = {}'.format(a.shape[0], c.shape[0], a.shape[0]+c.shape[0]))
                #print(class_weight)
                history = classifier.fit(X_train, _y_train, validation_data=(X_test, _y_test), batch_size=1024, nb_epoch=3000, shuffle=True, class_weight=class_weight, callbacks=[e])
               
                #predicted = classifier.predict(X2)
                #for i in range(len(predicted)):
                #   if predicted[i] < threshold:
                #       predicted[i] = 0
                #   else:
                #       predicted[i] = 1
                #predicted = np.asarray(predicted).astype(int)
                #cm = confusion_matrix(_y2, predicted,labels=[0,1])
                #tp = cm[0][0]
                #fn = cm[0][1]
                #fp = cm[1][0]
                #tn = cm[1][1]
                #tpr = tp/float(tp+fn)
                ##fpr = fp/float(fp+tn)
                #fnr = fn/float(fn+tp)
                #tnr = tn/float(tn+fp)
                #print('TP: {}, TN: {}, FP: {}, FN: {}'.format(tp,tn,fp,fn))
                #print('TPR: {}, TNR: {}, FPR: {}, FNR: {}'.format(tpr,tnr,fpr,fnr))
                #recall = tp/float(tp+fn)
                #specificity = tn/float(tn+fp)
                #print('Sensitivity/Recall: {}'.format(recall))
                #print('Specificity: {}'.format(specificity))
                #sensitivities.append(recall)
                #specificities.append(specificity)
                #precision = tp/float(tp+fp)
                #print('Precision: {}'.format(precision))
                #print('F1-measure: {}'.format(2*float(precision*recall)/float(precision+recall)))
                #fpr, tpr, _ = roc_curve(_y2, predicted)
                #roc_auc = auc(fpr, tpr)
                #aucs.append(roc_auc)
                #print('AUC: {}'.format(roc_auc))
                #print(classifier.predict(_X[0:1]))
                plot_training_info('prueba', param['metrics'] + ['loss'], param['save_plots'], history.history)
                #print(classifier.evaluate(_X2,_y2, batch_size=batch_size))
              
                   
                print('======= CALCULO SCIKIT GENERAL')
                # Confusion matrix
                predicted = classifier.predict(X_test)
                for i in range(len(predicted)):
                   if predicted[i] < threshold:
                       predicted[i] = 0
                   else:
                       predicted[i] = 1
                predicted = np.asarray(predicted).astype(int)
                cm = confusion_matrix(_y_test, predicted,labels=[0,1])
                tp = cm[0][0]
                fn = cm[0][1]
                fp = cm[1][0]
                tn = cm[1][1]
                tpr = tp/float(tp+fn)
                fpr = fp/float(fp+tn)
                fnr = fn/float(fn+tp)
                tnr = tn/float(tn+fp)
                print('TP: {}, TN: {}, FP: {}, FN: {}'.format(tp,tn,fp,fn))
                print('TPR: {}, TNR: {}, FPR: {}, FNR: {}'.format(tpr,tnr,fpr,fnr))
                print('Sensitivity/Recall: {}'.format(tp/float(tp+fn)))
                print('Specificity: {}'.format(tn/float(tn+fp)))  
                print('Accuracy: {}'.format(accuracy_score(_y_test, predicted)))
                sensitivities_general.append(tp/float(tp+fn))
                specificities_general.append(tn/float(tn+fp))
                  
                print('======= CALCULO SCIKIT URFALL')
                predicted = classifier.predict(X2_test)
                for i in range(len(predicted)):
                    if predicted[i] < threshold:
                        predicted[i] = 0
                    else:
                        predicted[i] = 1
                predicted = np.asarray(predicted).astype(int)
                cm = confusion_matrix(_y2_test, predicted,labels=[0,1])
                tp = cm[0][0]
                fn = cm[0][1]
                fp = cm[1][0]
                tn = cm[1][1]
                tpr = tp/float(tp+fn)
                fpr = fp/float(fp+tn)
                fnr = fn/float(fn+tp)
                tnr = tn/float(tn+fp)
                print('TP: {}, TN: {}, FP: {}, FN: {}'.format(tp,tn,fp,fn))
                print('TPR: {}, TNR: {}, FPR: {}, FNR: {}'.format(tpr,tnr,fpr,fnr))
                print('Sensitivity/Recall: {}'.format(tp/float(tp+fn)))
                print('Specificity: {}'.format(tn/float(tn+fp)))
                print('Accuracy: {}'.format(accuracy_score(_y2_test, predicted)))
                sensitivities_urfall.append(tp/float(tp+fn))
                specificities_urfall.append(tn/float(tn+fp))
                
                
                print('======= CALCULO SCIKIT MULTICAM')
                predicted = classifier.predict(X1_test)
                for i in range(len(predicted)):
                    if predicted[i] < threshold:
                        predicted[i] = 0
                    else:
                        predicted[i] = 1
                predicted = np.asarray(predicted).astype(int)
                cm = confusion_matrix(_y1_test, predicted,labels=[0,1])
                tp = cm[0][0]
                fn = cm[0][1]
                fp = cm[1][0]
                tn = cm[1][1]
                tpr = tp/float(tp+fn)
                fpr = fp/float(fp+tn)
                fnr = fn/float(fn+tp)
                tnr = tn/float(tn+fp)
                print('TP: {}, TN: {}, FP: {}, FN: {}'.format(tp,tn,fp,fn))
                print('TPR: {}, TNR: {}, FPR: {}, FNR: {}'.format(tpr,tnr,fpr,fnr))
                print('Sensitivity/Recall: {}'.format(tp/float(tp+fn)))
                print('Specificity: {}'.format(tn/float(tn+fp)))
                print('Accuracy: {}'.format(accuracy_score(_y1_test, predicted)))
                sensitivities_multicam.append(tp/float(tp+fn))
                specificities_multicam.append(tn/float(tn+fp))
                
                print('======= CALCULO SCIKIT FDD')
                predicted = classifier.predict(X3_test)
                for i in range(len(predicted)):
                    if predicted[i] < threshold:
                        predicted[i] = 0
                    else:
                        predicted[i] = 1
                predicted = np.asarray(predicted).astype(int)
                cm = confusion_matrix(_y3_test, predicted,labels=[0,1])
                tp = cm[0][0]
                fn = cm[0][1]
                fp = cm[1][0]
                tn = cm[1][1]
                tpr = tp/float(tp+fn)
                fpr = fp/float(fp+tn)
                fnr = fn/float(fn+tp)
                tnr = tn/float(tn+fp)
                print('TP: {}, TN: {}, FP: {}, FN: {}'.format(tp,tn,fp,fn))
                print('TPR: {}, TNR: {}, FPR: {}, FNR: {}'.format(tpr,tnr,fpr,fnr))
                print('Sensitivity/Recall: {}'.format(tp/float(tp+fn)))
                print('Specificity: {}'.format(tn/float(tn+fp)))
                print('Accuracy: {}'.format(accuracy_score(_y3_test, predicted)))
                sensitivities_fdd.append(tp/float(tp+fn))
                specificities_fdd.append(tn/float(tn+fp))
                
            # FIN DEL BUCLE
            print('LEAVE-ONE-OUT RESULTS ===================')
            
            print("Sensitivity General: %.2f%% (+/- %.2f%%)" % (np.mean(sensitivities_general), np.std(sensitivities_general)))
            print("Specificity General: %.2f%% (+/- %.2f%%)\n" % (np.mean(specificities_general), np.std(specificities_general)))
            print("Sensitivity UR Fall: %.2f%% (+/- %.2f%%)" % (np.mean(sensitivities_urfall), np.std(sensitivities_urfall)))
            print("Specificity UR Fall: %.2f%% (+/- %.2f%%)\n" % (np.mean(specificities_urfall), np.std(specificities_urfall)))
            print("Sensitivity Multicam: %.2f%% (+/- %.2f%%)" % (np.mean(sensitivities_multicam), np.std(sensitivities_multicam)))
            print("Specificity Multicam: %.2f%% (+/- %.2f%%)\n" % (np.mean(specificities_multicam), np.std(specificities_multicam)))
            print("Sensitivity Multicam: %.2f%% (+/- %.2f%%)" % (np.mean(sensitivities_fdd), np.std(sensitivities_fdd)))
            print("Specificity Multicam: %.2f%% (+/- %.2f%%)" % (np.mean(specificities_fdd), np.std(specificities_fdd)))
            print(exp)
            print(len(sensitivities_general))
            
            
                   #sys.exit()
                   #clf = svm.SVC()
                   #clf.fit(a,np.asarray(b).argmax(1))
                   #svm.score(c,np.asarray(d).argmax(1))
                   #SVC(C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None,
                    #   verbose=False, max_iter=-1, decision_function_shape=None, random_state=None)
                   #print(classifier.evaluate(X2,Y2, batch_size=batch_size))
                   #sys.exit()
                   
                       #recortar = False
            #if recortar:
            #    if len(all0) < len(all1):
            #        all1 = np.random.choice(all1, len(all0), replace=False)
            #    else:
            #        all0 = np.random.choice(all0, len(all1), replace=False)
            #    allin = np.concatenate((all0.flatten(),all1.flatten()))
            #    allin.sort()
            #    X = X[allin,...]
            #    _y = _y[allin]
            #    print('Train (after) Falls/NoFalls in dataset: {}/{}, total data: {}'.format(len(all0), len(all1), X.shape[0]))
            #all1 = all1[:len(all1)/2]
            #Y = Y[all1]
       
    sys.exit()
        
if __name__ == '__main__':
    parameter_file = '/home/anunez/project/fall_experiments.json'
    with open(parameter_file) as data_file:    
        param = json.load(data_file)
        
    i = 0
    model_file = '/home/anunez/project/models/exp_'
    weights_file = '/home/anunez/project/weights/exp_'
    for learning_rate in [0.01, 0.01, 0.005, 0.0001]:
        for batch_size in [1024, 256, 128, 32, 16]: # en reserva 256,
            for batch_norm in [True, False]:
                for dropout in [[0.9,0.8]]: #descartamos 0.5-0.5
                    for class_weights in [[2,1]]:
                        for nb_neurons in [4096]: #descartamos 500, 2048 y 8192
                            exp = '{}_lr{}_batchs{}_batchnorm{}_drop{}-{}_class_weight{}-{}_nbneuro{}'.format(i, learning_rate, batch_size, batch_norm, dropout[0],dropout[1], class_weights[0],class_weights[1], nb_neurons)
                            main(learning_rate, batch_size, dropout, batch_norm, class_weights[0],class_weights[1], nb_neurons, exp, model_file, weights_file)
                            i += 1

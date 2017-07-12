# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 11:08:21 2017

@author: adrian
"""

from __future__ import print_function
import sys
#import caffe
sys.path.insert(0, '/home/adrian/caffe/python')
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import os
import time
import urllib2
from zipfile import ZipFile
from PIL import Image
import io
from sklearn.model_selection import StratifiedShuffleSplit
from functions import load_gazeplus_dataset, load_adl_dataset, load_model, save_model, createGenerator
#from keras.applications.vgg16 import VGG16
from vgg16module import VGG16
from keras.applications.resnet50 import ResNet50

from keras.models import Model, model_from_json, model_from_yaml, Sequential
from keras.layers import Input, Convolution2D, MaxPooling2D, LSTM, Reshape, Merge, TimeDistributed, Flatten, Activation, Dense, Dropout, merge, AveragePooling2D, ZeroPadding2D, Lambda
from keras.regularizers import l2, activity_l2
from keras.optimizers import Adam, SGD
from keras.layers.normalization import BatchNormalization 
from keras import backend as K
K.set_image_dim_ordering('th')
#from attention import SpatialTransformer
from keras.utils import np_utils
from keras.utils.np_utils import probas_to_classes
from sklearn.metrics import confusion_matrix, accuracy_score
from skimage.io import imsave
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from keras.utils.np_utils import to_categorical
import json
from scipy.ndimage import minimum, maximum, imread
import math
from mpl_toolkits.axes_grid1 import make_axes_locatable
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

from keras.layers.advanced_activations import ELU
#import transcaffe as tc

def global_average_pooling(x):
    return K.mean(x, axis = (2, 3))

def global_average_pooling_shape(input_shape):
    return input_shape[0:2]

def get_caffe_params(netname, paramname):
   net = caffe.Net(netname, paramname, caffe.TEST)
   net.save_hdf5('/home/adrian/project/caffedata.h5')
   params = OrderedDict()
   for layername in net.params:
      caffelayer = net.params[layername]
      params[layername] = []
      for sublayer in caffelayer:
         params[layername].append(sublayer.data)
      print("layer " +layername+ " has " +str(len(caffelayer))+ " sublayers, shape "+str(params[layername][0].shape))
   return params, net

def make_mosaic(imgs, nrows, ncols, border=1):
    """
    Given a set of images with all the same shape, makes a
    mosaic with nrows and ncols
    """
    nimgs = imgs.shape[0]
    imshape = imgs.shape[1:]
    
    mosaic = ma.masked_all((nrows * imshape[0] + (nrows - 1) * border,
                            ncols * imshape[1] + (ncols - 1) * border),
                            dtype=np.float32)
    
    paddedh = imshape[0] + border
    paddedw = imshape[1] + border
    for i in xrange(nimgs):
        row = int(np.floor(i / ncols))
        col = i % ncols
        
        mosaic[row * paddedh:row * paddedh + imshape[0],
               col * paddedw:col * paddedw + imshape[1]] = imgs[i]
    return mosaic

def nice_imshow(ax, data, vmin=None, vmax=None, cmap=None):
    """Wrapper around pl.imshow"""
    if cmap is None:
        cmap = cm.jet
    if vmin is None:
        vmin = data.min()
    if vmax is None:
        vmax = data.max()
    #divider = make_axes_locatable(ax)
    #cax = divider.append_axes("right", size="5%", pad=0.05)
    fig = plt.figure()
    plt.plot(data)
    #im = ax.imshow(data, vmin=vmin, vmax=vmax, interpolation='nearest', cmap=cmap)
    plt.savefig('imagen.jpg')
    plt.gcf().clear()
    plt.close(fig)

class printbatch(Callback):
    def on_batch_end(self, epoch, logs={}):
        print(logs)
        
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
 
def countData():
    data_folder = '/ssd_drive/data/'
    parts = np.zeros((101, 3))

    activity_folders = [f for f in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, f))]
    activity_folders.sort()
        
    total_data = []
    for i in range(3):
        total_data.append([])
    
    for (activity_folder, nb_activity_folder) in zip(activity_folders, range(len(activity_folders))):
        path1 = data_folder + activity_folder + '/'
        video_folders = [f for f in os.listdir(path1) if os.path.isdir(os.path.join(path1, f))]
        video_folders.sort()
        l = len(video_folders)
        for i in range(3):
            total_data[i].append([])
        #for i in range(3):
         #   total_data[nb_activity_folder].append([])
        # TRAINING
        video_folders_aux = video_folders[:int(0.7*l)]
        parts[nb_activity_folder, 0] += len(video_folders_aux)
        for video_folder in video_folders_aux:
            path2 = path1 + video_folder + '/'
            images = glob.glob(path2 + 'flow_x*.jpg')
            total_data[0][nb_activity_folder].append(len(images))
        # VALIDATION
        video_folders_aux = video_folders[int(l*0.7):int(l*0.85)]
        parts[nb_activity_folder, 1] += len(video_folders_aux)
        for video_folder in video_folders_aux:
            path2 = path1 + video_folder + '/'
            images = glob.glob(path2 + 'flow_x*.jpg')
            total_data[1][nb_activity_folder].append(len(images))
        # TEST
        video_folders_aux = video_folders[int(l*0.85):]
        parts[nb_activity_folder, 2] += len(video_folders_aux)
        for video_folder in video_folders_aux:
            path2 = path1 + video_folder + '/'
            images = glob.glob(path2 + 'flow_x*.jpg')
            total_data[2][nb_activity_folder].append(len(images))
    return (total_data, parts)
    
def generator(folder1,folder2):
    for x,y in zip(folder1,folder2):
        yield x,y
          
def saveFeatures(param, max_label, batch_size, phase, amount_of_data, parts, save_features, feature_extractor, classifier, features_file, labels_file, train_split, test_split):
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
            
def prueba(model, classifier, path2):
    f = h5py.File('/home/anunez/project/prueba.h5','w')
    f.create_dataset('prueba', shape=(20,25088))
    mean_file = '/ssd_drive/flow_mean.mat'
    print(path2 + 'flow_x_*')
    x_images = glob.glob(path2 + 'flow_x_*')
    x_images.sort()
    y_images = glob.glob(path2 + 'flow_y_*')
    y_images.sort()
        
    j = 0
    dim = (256,340,20,len(x_images))
    print(len(x_images))
    cont = 0
    i = 0
    flow = np.zeros(dim, dtype=np.float64)
    for flow_x_file, flow_y_file in zip(x_images, y_images):
        img_x = cv2.imread(flow_x_file, cv2.IMREAD_GRAYSCALE)
        img_y = cv2.imread(flow_y_file, cv2.IMREAD_GRAYSCALE)
        
        img_x = cv2.resize(img_x, dim[1::-1])
        img_y = cv2.resize(img_y, dim[1::-1])
        flow[:,:,j*2  ,0] = img_x
        flow[:,:,j*2+1,0] = img_y

        j += 1
        cont += 1
        if j == 10:
            j = 0
            d = sio.loadmat(mean_file)
            flow_mean = d['image_mean']
            flow_1 = flow[:224, :224, :,:]
            flow_1 = flow_1 - np.tile(flow_mean[...,np.newaxis], (1, 1, 1, flow.shape[3]))
            flow_1 = np.transpose(flow_1, (3,2,0,1))
            prediction = model.predict(np.expand_dims(flow_1[0,...],0))
            prediction = np.reshape(prediction, (prediction.shape[0], prediction.shape[1]*prediction.shape[2]*prediction.shape[3]))
            predicted_class = classifier.predict(prediction)
            print(np.argmax(predicted_class), predicted_class[0,3])
            #f['prueba'][i] = prediction
            i+=1
            
def prueba2(model, features, labels):
    h5features = h5py.File(features)
    #print(h5features.keys())
    total = 0
    for num, key in zip(range(len(h5features.keys())), h5features.keys()):
        #print('='*20)
        #print('Clase {}'.format(key))
        #print('='*20)
        aciertos = 0.0
        for i in range(h5features[key].shape[0]):
            clase = np.argmax(model.predict(np.expand_dims(h5features[key][i,:],0)))
            if clase == num:
                aciertos += 1.0
        #print(float(h5features[key].shape[0]))
        #print('Accuracy: {}'.format(float(aciertos)/float(h5features[key].shape[0])))
        total += aciertos
    print('Accuracy: {}'.format(float(total/100.0)))

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
    features_file = '/home/anunez/project/features_multicam_final2.h5'
    labels_file = '/home/anunez/project/labels_multicam_final2.h5'
    #features_file = '/home/anunez/project/features/features_multicam_final2.h5'
    #labels_file = '/home/anunez/project/labels/labels_multicam_final2.h5'
    features_file_test = '/home/anunez/project/features/features_urfall_final3.h5'
    labels_file_test = '/home/anunez/project/labels/labels_urfall_final3.h5'
    features_file_test2 = '/home/anunez/project/features/features_fdd_final3.h5'
    labels_file_test2 = '/home/anunez/project/labels/labels_fdd_final3.h5'
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
    cross_val = True
    if do_training:
        if do_training_with_features:
            h5features = h5py.File(features_file, 'r')
            h5labels = h5py.File(labels_file, 'r')
            
            if cross_val:
                h5features_test = h5py.File(features_file_test, 'r')
                h5labels_test = h5py.File(labels_file_test, 'r')
                h5features_test2 = h5py.File(features_file_test2, 'r')
                h5labels_test2 = h5py.File(labels_file_test2, 'r')
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
                        
                sensitivities = []
                specificities = []
                aucs = []
                accuracies = []
            # LEAVE-ONE-OUT
            for cam in range(8):
                print('='*30)
                print('LEAVE-ONE-OUT STEP {}/8'.format(cam))
                print('='*30)
                test_x = cams_x[cam]
                test_y = cams_y[cam]
                train_x = cams_x[0:cam] + cams_x[cam+1:]
                train_y = cams_y[0:cam] + cams_y[cam+1:]
                if cross_val:
                    X = []
                    _y = []
                    for cam_x, cam_y in zip(train_x, train_y):
                        print(cam_x.shape, cam_y.shape)
                        all0 = np.asarray(np.where(cam_y==0)[0])
                        all1 = np.asarray(np.where(cam_y==1)[0])
                        all1 = np.random.choice(all1, len(all0), replace=False)
                        allin = np.concatenate((all0.flatten(),all1.flatten()))
                        allin.sort()
                        X.append(np.asarray(cam_x[allin,...]))
                        _y.append(np.asarray(cam_y[allin]))
                    X = np.asarray(np.concatenate(X,axis=0))
                    _y = np.asarray(np.concatenate(_y,axis=0))
                    X2 = np.asarray(test_x)
                    _y2 = np.asarray(test_y)
                else:
                    if False:
                        temp_X = np.asarray(np.concatenate(cams_x,axis=0))
                        temp_y = np.asarray(np.concatenate(cams_y,axis=0))
                        
                        sss = StratifiedShuffleSplit(n_splits=2, test_size=0.2, random_state=777)
                        sss.get_n_splits(temp_X, temp_y)
                        train_index, test_index = [], []
                        for a, b in sss.split(temp_X, temp_y):
                            train_index = a
                            test_index = b                
                        train_index = np.asarray(train_index)
                        test_index = np.asarray(test_index)
                        X, X2 = temp_X[train_index], temp_X[test_index]
                        _y, _y2 = temp_y[train_index], temp_y[test_index]
                    if True:
                        X = np.asarray(h5features['train'])
                        _y = np.asarray(h5labels['train'])
                        X2 = np.asarray(h5features['test']) 
                        _y2 = np.asarray(h5labels['test'])
                all0 = np.asarray(np.where(_y==0)[0])
                all1 = np.asarray(np.where(_y==1)[0])
                recortar = False
                if recortar:
                    if len(all0) < len(all1):
                        all1 = np.random.choice(all1, len(all0), replace=False)
                    else:
                        all0 = np.random.choice(all0, len(all1), replace=False)
                    allin = np.concatenate((all0.flatten(),all1.flatten()))
                    allin.sort()
                    X = X[allin,...]
                    _y = _y[allin]
                                
                all0 = np.asarray(np.where(_y==0)[0])
                all1 = np.asarray(np.where(_y==1)[0])
                print('Train Falls/NoFalls in dataset: {}/{}, total data: {}'.format(len(all0), len(all1), X.shape[0]))
                all0 = np.asarray(np.where(_y2==0)[0])
                all1 = np.asarray(np.where(_y2==1)[0])
                print('Test Falls/NoFalls in dataset: {}/{}, total data: {}'.format(len(all0), len(all1), X2.shape[0]))
                
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
                if batch_size == 0:
                    history = classifier.fit(X, _y, validation_data=(X2, _y2), batch_size=X.shape[0], nb_epoch=3000, shuffle=True, class_weight=class_weight, callbacks=[e])
                    #history = classifier.fit(X, _y, validation_data=(X2, _y2), batch_size=1024, nb_epoch=3000, shuffle=True, class_weight=class_weight)
                else:
                    history = classifier.fit(X, _y, validation_data=(X2, _y2), batch_size=batch_size, nb_epoch=3000, shuffle=True, class_weight=class_weight, callbacks=[e])
                    #history = classifier.fit(X, _y, validation_data=(X2, _y2), batch_size=1024, nb_epoch=3000, shuffle=True, class_weight=class_weight)
                #scores = classifier.evaluate(test_x, test_y, verbose=0)
                #cvscores.append(scores[1] * 100)
                #all0 = np.asarray(np.where(test_y==0)[0])
                #all1 = np.asarray(np.where(test_y==1)[0])
                #fall_period = classifier.predict(test_x[all0])
                #no_fall_period = classifier.predict(test_x[all1])
                #for i in range(len(fall_period)):
                #   if fall_period[i] < threshold:
                #       fall_period[i] = 0
                #   else:
                #       fall_period[i] = 1
                #fall_period = np.asarray(fall_period).astype(int)
                #for i in range(len(no_fall_period)):
                #   if no_fall_period[i] < threshold:
                #       no_fall_period[i] = 0
                #   else:
                #       no_fall_period[i] = 1
                #no_fall_period = np.asarray(no_fall_period).astype(int)
            #
                #fn, tp = 0, 0
                #for i in range(len(fall_period)):
                ##    if fall_period[i] == 0:
                ##        tp = 1
                #        break
                #if tp != 1:
                #    fn = 1
                    
                #fp, tn = 0, 0
                #for i in range(len(no_fall_period)):
                #    if no_fall_period[i] == 0:
                ##        fp = 1
                #        break
                #if fp != 1:
                #    tn = 1
#
                #fp = cm[1][0]
                #tn = cm[1][1]
                predicted = classifier.predict(X2)
                for i in range(len(predicted)):
                   if predicted[i] < threshold:
                       predicted[i] = 0
                   else:
                       predicted[i] = 1
                predicted = np.asarray(predicted).astype(int)
                cm = confusion_matrix(_y2, predicted,labels=[0,1])
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
                recall = tp/float(tp+fn)
                specificity = tn/float(tn+fp)
                print('Sensitivity/Recall: {}'.format(recall))
                print('Specificity: {}'.format(specificity))
                if cross_val:
                    sensitivities.append(recall)
                    specificities.append(specificity)
                    
                precision = tp/float(tp+fp)
                print('Precision: {}'.format(precision))
                print('F1-measure: {}'.format(2*float(precision*recall)/float(precision+recall)))
                fpr, tpr, _ = roc_curve(_y2, predicted)
                roc_auc = auc(fpr, tpr)
                acc = accuracy_score(_y2, predicted)
                print(acc)
                if cross_val:
                    aucs.append(roc_auc)
                    accuracies.append(acc)
                print('AUC: {}'.format(roc_auc))
                #print(classifier.predict(_X[0:1]))
                plot_training_info('prueba', param['metrics'] + ['loss'], param['save_plots'], history.history)
                #print(classifier.evaluate(_X2,_y2, batch_size=batch_size))
                if compute_metrics:
                   
                   #indices = np.where(Y==0)[0]
                   predicted = classifier.predict(np.asarray(X2))
                   ind0 = np.where(np.asarray(_y2)<threshold)[0]
                   ind1 = np.where(np.asarray(_y2)>=threshold)[0]
                   print(len(ind0), len(ind1), len(ind0)+len(ind1))
               
               
                # Compute ROC curve and ROC area for each class  
                if compute_roc_curve:
                   fpr = dict()
                   tpr = dict()
                   roc_auc = dict()
                   _y2 = np.asarray(_y2).astype(int)
                   
                   ground_truth = to_categorical(_y2, 2)
                   for i in range(len(predicted)):
                       if predicted[i] < threshold:
                           predicted[i] = 0
                       else:
                           predicted[i] = 1
                   predicted = np.asarray(predicted).astype(int)
                   
                   scores = to_categorical(predicted, 2)
                   for i in range(2):
                       fpr[i], tpr[i], _ = roc_curve(ground_truth[:, i], scores[:, i])
                       roc_auc[i] = auc(fpr[i], tpr[i])
    
                   # Compute micro-average ROC curve and ROC area
                   fpr["micro"], tpr["micro"], _ = roc_curve(ground_truth.ravel(), scores.ravel())
                   roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
                   plt.figure()
                   lw = 2
                   plt.plot(fpr[0], tpr[0], color='darkorange',
                             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[0])
                   plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
                   plt.xlim([0.0, 1.0])
                   plt.ylim([0.0, 1.05])
                   plt.xlabel('False Positive Rate')
                   plt.ylabel('True Positive Rate')
                   plt.title('ROC Curve')
                   plt.legend(loc="lower right")
                   plt.savefig('/home/anunez/project/roc/' + exp + 'roc.png')
                   plt.gcf().clear()
               
                   plt.figure()
                   plt.hist(ind0)
                   plt.title('Predicted as 0')
                   plt.savefig('/home/anunez/project/threshold/' + exp + 'ind0.png')
                   plt.gcf().clear()
                   
                   plt.figure()
                   plt.hist(ind1)
                   plt.title('Predicted as 1')
                   plt.savefig('/home/anunez/project/threshold/' + exp + 'ind1.png')
                   plt.gcf().clear()
                   
                   
                   with open('/home/anunez/project/metrics/' + exp + 'metrics.txt', 'w') as f:
                   # Confusion matrix
                       for i in range(len(predicted)):
                           if predicted[i] < threshold:
                               predicted[i] = 0
                           else:
                               predicted[i] = 1
                       predicted = np.asarray(predicted).astype(int)
                       cm = confusion_matrix(_y2, predicted,labels=[0,1])
                       print(cm)
                       #print('======= CALCULO MANUAL')
                       #tp = float(sum((1-predicted)[ind0]))/float(len(ind0))
                       #tn = float(sum(predicted[ind1]))/float(len(ind1))
                       #fp = float(sum((1-predicted)[ind1]))/float(len(ind1))
                       #fn = float(sum(predicted[ind0]))/float(len(ind0))
                       #print('TP: {}, TN: {}, FP: {}, FN: {}'.format(tp,tn,fp,fn))
                       #print('Sensitivity/Recall: {}'.format(tp/(tp+fn)))
                       #print('Specificity: {}'.format(tn/(tn+fp)))
                       
                       
                       print('======= CALCULO SCIKIT')
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
                       f.write('{}\t{}\n{}\t{}\n'.format(tp,fp,fn,tn))
                       f.write('TP: {}, TN: {}, FP: {}, FN: {}\n'.format(tp,tn,fp,fn))
                       
                       f.write('TPR: {}, TNR: {}, FPR: {}, FNR: {}\n'.format(tpr,tnr,fpr,fnr))
                       f.write('Sensitivity/Recall: {}\n'.format(tp/float(tp+fn)))
                       f.write('Specificity: {}\n'.format(tn/float(tn+fp)))
                       
                       
                       print('Accuracy: {}'.format(accuracy_score(_y2, predicted)))
                       f.write('Accuracy: {}\n'.format(accuracy_score(_y2, predicted)))
                       print(cam)
            # FIN DEL BUCLE
            print('LEAVE-ONE-OUT RESULTS ===================')
            print(len(sensitivities))
            print("Sensitivity: %.2f%% (+/- %.2f%%)" % (np.mean(sensitivities), np.std(sensitivities)))
            print("Specificity: %.2f%% (+/- %.2f%%)" % (np.mean(specificities), np.std(specificities)))
            print("AUC: %.2f%% (+/- %.2f%%)" % (np.mean(aucs), np.std(aucs)))
            print("ACCURACY: %.2f%% (+/- %.2f%%)" % (np.mean(accuracies), np.std(accuracies)))
            print(exp)
            
            
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

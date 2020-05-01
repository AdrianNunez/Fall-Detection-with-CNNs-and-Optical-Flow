from __future__ import print_function
from numpy.random import seed
seed(1)
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import os
import h5py
import scipy.io as sio
import cv2
import glob
import gc

from keras.models import load_model, Model, Sequential
from keras.layers import (Input, Conv2D, MaxPooling2D, Flatten,
		 	  Activation, Dense, Dropout, ZeroPadding2D)
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization 
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.model_selection import KFold, StratifiedShuffleSplit
from keras.layers.advanced_activations import ELU

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# CHANGE THESE VARIABLES ---
data_folder = '/home/anunez/Downloads/Multicam_OF/'
mean_file = '/home/anunez/flow_mean.mat'
vgg_16_weights = 'weights.h5'
save_features = False
save_plots = True

# Set to 'True' if you want to restore a previous trained models
# Training is skipped and test is done
use_checkpoint = False
# --------------------------

best_model_path = 'models/'
plots_folder = 'plots/'
checkpoint_path = best_model_path + 'fold_'

saved_files_folder = 'saved_features/'
features_file = saved_files_folder + 'features_multicam_tf.h5'
labels_file = saved_files_folder + 'labels_multicam_tf.h5'
features_key = 'features'
labels_key = 'labels'

num_cameras = 8
L = 10
num_features = 4096
batch_norm = True
learning_rate = 0.01
mini_batch_size = 0
weight_0 = 1
epochs = 6000
use_validation = False
# After the training stops, use train+validation to train for 1 epoch
use_val_for_training = False
val_size = 100
# Threshold to classify between positive and negative
threshold = 0.5

# Name of the experiment
exp = 'multicam_lr{}_batchs{}_batchnorm{}_w0_{}'.format(learning_rate,
					       mini_batch_size,
					       batch_norm,
					       weight_0)
        
def plot_training_info(case, metrics, save, history):
    '''
    Function to create plots for train and validation loss and accuracy
    Input:
    * case: name for the plot, an 'accuracy.png' or 'loss.png' 
	will be concatenated after the name.
    * metrics: list of metrics to store: 'loss' and/or 'accuracy'
    * save: boolean to store the plots or only show them.
    * history: History object returned by the Keras fit function.
    '''
    val = False
    if 'val_acc' in history and 'val_loss' in history:
        val = True
    plt.ioff()
    if 'accuracy' in metrics:     
        fig = plt.figure()
        plt.plot(history['acc'])
        if val: plt.plot(history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        if val: 
            plt.legend(['train', 'val'], loc='upper left')
        else:
            plt.legend(['train'], loc='upper left')
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
        if val: plt.plot(history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        #plt.ylim(1e-3, 1e-2)
        plt.yscale("log")
        if val: 
            plt.legend(['train', 'val'], loc='upper left')
        else:
            plt.legend(['train'], loc='upper left')
        if save == True:
            plt.savefig(case + 'loss.png')
            plt.gcf().clear()
        else:
            plt.show()
        plt.close(fig)
 
def generator(list1, lits2):
    '''
    Auxiliar generator: returns the ith element of both given list with
	 each call to next() 
    '''
    for x,y in zip(list1,lits2):
        yield x, y
          
def saveFeatures(feature_extractor,
		 features_file,
		 labels_file,
		 features_key, 
		 labels_key):
    '''
    Function to load the optical flow stacks, do a feed-forward through the
	 feature extractor (VGG16) and
    store the output feature vectors in the file 'features_file' and the 
	labels in 'labels_file'.
    Input:
    * feature_extractor: model VGG16 until the fc6 layer.
    * features_file: path to the hdf5 file where the extracted features are
	 going to be stored
    * labels_file: path to the hdf5 file where the labels of the features
	 are going to be stored
    * features_key: name of the key for the hdf5 file to store the features
    * labels_key: name of the key for the hdf5 file to store the labels
    '''

    if not os.path.exists(saved_files_folder):
        os.makedirs(saved_files_folder)
    
    class0 = 'Falls'
    class1 = 'NotFalls'
  
    # Load the mean file to subtract to the images
    d = sio.loadmat(mean_file)
    flow_mean = d['image_mean']
   
    h5features = h5py.File(features_file,'w')
    h5labels = h5py.File(labels_file,'w')
    
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
    for i in range(1,25):
        stages.append('chute{:02}'.format(i))
    
    for stage, nb_stage in zip(stages, range(len(stages))):
        h5features.create_group(stage)
        h5labels.create_group(stage)  
        path = data_folder + stage
        for nb_camera in range(1,num_cameras+1):
            h5features[stage].create_group('cam{}'.format(nb_camera))
            h5labels[stage].create_group('cam{}'.format(nb_camera))
            not_falls = glob.glob(
                path + '/NotFalls/camera{}*'.format(nb_camera)
            )
            not_falls.sort()
        
            for not_fall in not_falls:
                label = 1
                name = not_fall[not_fall.rfind('/')+1:]
                x_images = glob.glob(not_fall + '/flow_x*.jpg')
                x_images.sort()
                y_images = glob.glob(not_fall + '/flow_x*.jpg')
                y_images.sort()
                nb_stacks = int(len(x_images))-L+1
                
                features_notfall = h5features[stage][
					'cam{}'.format(nb_camera)
					].create_dataset(
					 name,
					 shape=(nb_stacks, num_features),
					 dtype='float64')
                labels_notfall = h5labels[stage][
					'cam{}'.format(nb_camera)
					].create_dataset(
					 name,
					 shape=(nb_stacks, 1),
					 dtype='float64')
                
                # NO FALL
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
                flow = flow - np.tile(flow_mean[...,np.newaxis],
				     (1, 1, 1, flow.shape[3]))
                flow = np.transpose(flow, (3, 0, 1, 2)) 
                predictions = np.zeros((flow.shape[0], num_features),
					dtype=np.float64)
                truth = np.zeros((flow.shape[0], 1), dtype=np.float64)
                for i in range(flow.shape[0]):
                    prediction = feature_extractor.predict(
						np.expand_dims(flow[i, ...],0))
                    predictions[i, ...] = prediction
                    truth[i] = label
                features_notfall[:,:] = predictions
                labels_notfall[:,:] = truth
                del (predictions, truth, flow, features_notfall,
		     labels_notfall, x_images, y_images, nb_stacks)
                gc.collect()
            
            if stage == 'chute24':
                continue

            falls = glob.glob(
                path + '/Falls/camera{}'.format(nb_camera)
            )
            falls.sort()
            for fall in falls:     
                label = 0
                name = fall[fall.rfind('/')+1:]
                x_images = glob.glob(fall + '/flow_x*.jpg')
                x_images.sort()
                y_images = glob.glob(fall + '/flow_y*.jpg')
                y_images.sort()
                nb_stacks = int(len(x_images))-L+1
                
                features_fall = h5features[stage][
					'cam{}'.format(nb_camera)
					].create_dataset(
					 name,
					 shape=(nb_stacks, num_features),
					 dtype='float64')
                labels_fall = h5labels[stage][
					'cam{}'.format(nb_camera)
					].create_dataset(
					 name,
					 shape=(nb_stacks, 1),
					 dtype='float64')
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
                flow = flow - np.tile(flow_mean[...,np.newaxis],
				      (1, 1, 1, flow.shape[3]))
                flow = np.transpose(flow, (3, 0, 1, 2)) 
                predictions = np.zeros((flow.shape[0], num_features),
					dtype=np.float64)
                truth = np.zeros((flow.shape[0], 1), dtype=np.float64)
                for i in range(flow.shape[0]):
                    prediction = feature_extractor.predict(
						np.expand_dims(flow[i, ...],0))
                    predictions[i, ...] = prediction
                    truth[i] = label
                features_fall[:,:] = predictions
                labels_fall[:,:] = truth
                del predictions, truth, flow, features_fall, labels_fall

    h5features.close()
    h5labels.close()

def load_dataset():
    h5features = h5py.File(features_file, 'r')
    h5labels = h5py.File(labels_file, 'r')
  
    # Load the data separated by cameras for cross-validation
    stages = []
    for i in range(1,25):
        stages.append('chute{:02}'.format(i))
    cams_x = []
    cams_y = []
    for stage, nb_stage in zip(stages, range(len(stages))):   
        for cam, nb_cam in zip(h5features[stage].keys(), range(8)):
            temp_x = []
            temp_y = []
            for key in h5features[stage][cam].keys():
                temp_x.append(np.asarray(h5features[stage][cam][key]))
                temp_y.append(np.asarray(h5labels[stage][cam][key]))
            temp_x = np.concatenate(temp_x,axis=0)
            temp_y = np.concatenate(temp_y,axis=0)
            if nb_stage == 0:
                cams_x.append(temp_x)
                cams_y.append(temp_y)
            else:
                cams_x[nb_cam] = np.concatenate([cams_x[nb_cam],
						 temp_x], axis=0)
                cams_y[nb_cam] = np.concatenate([cams_y[nb_cam],
						 temp_y], axis=0)
    return cams_x, cams_y

def main():          
    # ========================================================================
    # VGG-16 ARCHITECTURE
    # ========================================================================
    model = Sequential()
    
    model.add(ZeroPadding2D((1, 1), input_shape=(224, 224, 20)))
    model.add(Conv2D(64, (3, 3), activation='relu', name='conv1_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu', name='conv1_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(128, (3, 3), activation='relu', name='conv2_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(128, (3, 3), activation='relu', name='conv2_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, (3, 3), activation='relu', name='conv3_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, (3, 3), activation='relu', name='conv3_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, (3, 3), activation='relu', name='conv3_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu', name='conv4_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu', name='conv4_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu', name='conv4_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu', name='conv5_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu', name='conv5_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu', name='conv5_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    
    model.add(Flatten())
    model.add(Dense(num_features, name='fc6',
		    kernel_initializer='glorot_uniform'))
   
    # ========================================================================
    # WEIGHT INITIALIZATION
    # ========================================================================
    layerscaffe = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1',
		   'conv3_2', 'conv3_3', 'conv4_1', 'conv4_2', 'conv4_3',
		   'conv5_1', 'conv5_2', 'conv5_3', 'fc6', 'fc7', 'fc8']
    i = 0
    h5 = h5py.File(vgg_16_weights)
   
    layer_dict = dict([(layer.name, layer) for layer in model.layers])

    # Copy the weights stored in the 'vgg_16_weights' file to the
    # feature extractor part of the VGG16
    for layer in layerscaffe[:-3]:
        w2, b2 = h5['data'][layer]['0'], h5['data'][layer]['1']
        w2 = np.transpose(np.asarray(w2), (2,3,1,0))
        w2 = w2[::-1, ::-1, :, :]
        b2 = np.asarray(b2)
        layer_dict[layer].set_weights((w2, b2))

    # Copy the weights of the first fully-connected layer (fc6)
    layer = layerscaffe[-3]
    w2, b2 = h5['data'][layer]['0'], h5['data'][layer]['1']
    w2 = np.transpose(np.asarray(w2), (1,0))
    b2 = np.asarray(b2)
    layer_dict[layer].set_weights((w2, b2))

    # ========================================================================
    # FEATURE EXTRACTION
    # ========================================================================
    if save_features:
        saveFeatures(model, features_file,
		    labels_file, features_key,
		    labels_key)

    # ========================================================================
    # TRAINING
    # =======================================================================

    adam = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999,
		        epsilon=1e-08, decay=0.0005)
    model.compile(optimizer=adam, loss='categorical_crossentropy',
		          metrics=['accuracy'])  

    cams_x, cams_y = load_dataset()
                
    sensitivities = []
    specificities = []
    aucs = []
    accuracies = []
    
    # LEAVE-ONE-CAMERA-OUT CROSS-VALIDATION
    for cam in range(num_cameras):
        print('='*30)
        print('LEAVE-ONE-OUT STEP {}/8'.format(cam+1))
        print('='*30)
        # cams_x[nb_cam] contains all the optical flow stacks of
        # the 'cam' camera (where 'cam' is an integer from 0 to 24)
        test_x = cams_x[cam]
        test_y = cams_y[cam]
        train_x = cams_x[0:cam] + cams_x[cam+1:]
        train_y = cams_y[0:cam] + cams_y[cam+1:]
        # Flatten to 1D arrays
        train_x = np.asarray([train_x[i][j] 
            for i in range(len(train_x)) for j in range(len(train_x[i]))])
        train_y = np.asarray([train_y[i][j] 
            for i in range(len(train_y)) for j in range(len(train_y[i]))])
        
        # Create a validation subset from the training set
        zeroes = np.asarray(np.where(train_y==0)[0])
        ones = np.asarray(np.where(train_y==1)[0])
        trainval_split_0 = StratifiedShuffleSplit(n_splits=1,
                        test_size=val_size/2,
                        random_state=7)
        indices_0 = trainval_split_0.split(train_x[zeroes,...],
                        np.argmax(train_y[zeroes,...], 1))
        trainval_split_1 = StratifiedShuffleSplit(n_splits=1,
                        test_size=val_size/2,
                        random_state=7)
        indices_1 = trainval_split_1.split(train_x[ones,...],
                        np.argmax(train_y[ones,...], 1))
        train_indices_0, val_indices_0 = indices_0.next()
        train_indices_1, val_indices_1 = indices_1.next()

        _X_train = np.concatenate([train_x[zeroes,...][train_indices_0,...],
                    train_x[ones,...][train_indices_1,...]],axis=0)
        _y_train = np.concatenate([train_y[zeroes,...][train_indices_0,...],
                    train_y[ones,...][train_indices_1,...]],axis=0)
        X_val = np.concatenate([train_x[zeroes,...][val_indices_0,...],
                    train_x[ones,...][val_indices_1,...]],axis=0)
        y_val = np.concatenate([train_y[zeroes,...][val_indices_0,...],
                    train_y[ones,...][val_indices_1,...]],axis=0)
        y_val = np.squeeze(y_val)
        _y_train = np.squeeze(np.asarray(_y_train))

        # Balance the positive and negative samples           
        all0 = np.where(_y_train==0)[0]
        all1 = np.where(_y_train==1)[0]
        
        all1 = np.random.choice(all1, len(all0), replace=False)
        allin = np.concatenate((all0.flatten(), all1.flatten()))
        X_train = np.asarray(_X_train[allin,...])
        y_train = np.asarray(_y_train[allin])
        X_test = np.asarray(test_x)
        y_test = np.asarray(test_y)

        # ==================== CLASSIFIER ========================               
        extracted_features = Input(shape=(num_features,),
                    dtype='float32', name='input')
        if batch_norm:
            x = BatchNormalization(axis=-1, momentum=0.99,
                    epsilon=0.001)(extracted_features)
            x = Activation('relu')(x)
        else:
            x = ELU(alpha=1.0)(extracted_features)
        
        x = Dropout(0.9)(x)
        x = Dense(4096, name='fc2', init='glorot_uniform')(x)
        if batch_norm:
            x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)
            x = Activation('relu')(x)
        else:
            x = ELU(alpha=1.0)(x)
        x = Dropout(0.8)(x)
        x = Dense(1, name='predictions', init='glorot_uniform')(x)
        x = Activation('sigmoid')(x)
        
        classifier = Model(input=extracted_features,
                output=x, name='classifier')
        fold_best_model_path = best_model_path + 'multicam_fold_{}'.format(
                                cam)
        classifier.compile(optimizer=adam, loss='binary_crossentropy',
                metrics=['accuracy'])
        
        if not use_checkpoint:
            # ==================== TRAINING ========================     
            # weighting of each class: only the fall class gets
            # a different weight
            class_weight = {0: weight_0, 1: 1}

            callbacks = None
            if use_validation:
                # callback definition
                metric = 'val_loss'
                e = EarlyStopping(monitor=metric, min_delta=0, patience=100,
                        mode='auto')
                c = ModelCheckpoint(fold_best_model_path, monitor=metric,
                            save_best_only=True,
                            save_weights_only=False, mode='auto')
                callbacks = [e, c]
            validation_data = None
            if use_validation:
                validation_data = (X_val,y_val)
            _mini_batch_size = mini_batch_size
            if mini_batch_size == 0:
                _mini_batch_size = X_train.shape[0]

            history = classifier.fit(
                X_train, y_train, 
                validation_data=validation_data,
                batch_size=_mini_batch_size,
                nb_epoch=epochs,
                shuffle=True,
                class_weight=class_weight,
                callbacks=callbacks
            )

            if not use_validation:
                classifier.save(fold_best_model_path)

            plot_training_info(plots_folder + exp, ['accuracy', 'loss'],
                    save_plots, history.history)

            if use_validation and use_val_for_training:
                classifier = load_model(fold_best_model_path)
                
                # Use full training set (training+validation)
                X_train = np.concatenate((X_train, X_val), axis=0)
                y_train = np.concatenate((y_train, y_val), axis=0)

                history = classifier.fit(
                    X_train, y_train, 
                    validation_data=validation_data,
                    batch_size=_mini_batch_size,
                    nb_epoch=epochs,
                    shuffle='batch',
                    class_weight=class_weight,
                    callbacks=callbacks
                )

                classifier.save(fold_best_model_path)
            
        # ==================== EVALUATION ========================  

        # Load best model
        print('Model loaded from checkpoint')
        classifier = load_model(fold_best_model_path)

        predicted = classifier.predict(X_test)
        for i in range(len(predicted)):
            if predicted[i] < threshold:
                predicted[i] = 0
            else:
                predicted[i] = 1
        # Array of predictions 0/1
        predicted = np.asarray(predicted).astype(int)
        
        # Compute metrics and print them
        cm = confusion_matrix(y_test, predicted,labels=[0,1])
        tp = cm[0][0]
        fn = cm[0][1]
        fp = cm[1][0]
        tn = cm[1][1]
        tpr = tp/float(tp+fn)
        fpr = fp/float(fp+tn)
        fnr = fn/float(fn+tp)
        tnr = tn/float(tn+fp)
        precision = tp/float(tp+fp)
        recall = tp/float(tp+fn)
        specificity = tn/float(tn+fp)
        f1 = 2*float(precision*recall)/float(precision+recall)
        accuracy = accuracy_score(y_test, predicted)
        fpr, tpr, _ = roc_curve(y_test, predicted)
        roc_auc = auc(fpr, tpr)
        
        print('FOLD/CAMERA {} results:'.format(cam))
        print('TP: {}, TN: {}, FP: {}, FN: {}'.format(tp,tn,fp,fn))
        print('TPR: {}, TNR: {}, FPR: {}, FNR: {}'.format(
                        tpr,tnr,fpr,fnr))   
        print('Sensitivity/Recall: {}'.format(recall))
        print('Specificity: {}'.format(specificity))
        print('Precision: {}'.format(precision))
        print('F1-measure: {}'.format(f1))
        print('Accuracy: {}'.format(accuracy))
        print('AUC: {}'.format(roc_auc))
        
        # Store the metrics for this epoch
        sensitivities.append(tp/float(tp+fn))
        specificities.append(tn/float(tn+fp))
        aucs.append(roc_auc)
        accuracies.append(accuracy)

    print('LEAVE-ONE-OUT RESULTS ===================')
    print("Sensitivity: %.2f%% (+/- %.2f%%)" % (np.mean(sensitivities),
                            np.std(sensitivities)))
    print("Specificity: %.2f%% (+/- %.2f%%)" % (np.mean(specificities),
                            np.std(specificities)))
    print("Accuracy: %.2f%% (+/- %.2f%%)" % (np.mean(accuracies),
                            np.std(accuracies)))
    print("AUC: %.2f%% (+/- %.2f%%)" % (np.mean(aucs), np.std(aucs)))
    
if __name__ == '__main__':
    if not os.path.exists(best_model_path):
        os.makedirs(best_model_path)
    if not os.path.exists(plots_folder):
        os.makedirs(plots_folder)
        
    main()

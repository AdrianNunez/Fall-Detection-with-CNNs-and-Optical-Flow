from __future__ import print_function
from numpy.random import seed
seed(1)
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import os
from keras.models import Model, Sequential
from keras.layers import Input, Convolution2D, MaxPooling2D, Flatten, Activation, Dense, Dropout, ZeroPadding2D
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization 
from keras import backend as K
K.set_image_dim_ordering('th')
from sklearn.metrics import confusion_matrix, accuracy_score
import h5py
import scipy.io as sio
import cv2
import glob
import gc
from sklearn.model_selection import KFold
from keras.layers.advanced_activations import ELU

data_folder = '/ssd_drive/UR_Fall_OF/'
mean_file = '/ssd_drive/flow_mean.mat'
L = 10
num_features = 4096
        
def plot_training_info(case, metrics, save, history):
    '''
    Function to create plots for train and validation loss and accuracy
    Input:
    * case: name for the plot, an 'accuracy.png' or 'loss.png' will be concatenated after the name.
    * metrics: list of metrics to store: 'loss' and/or 'accuracy'
    * save: boolean to store the plots or only show them.
    * history: History object returned by the Keras fit function.
    '''
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
 
def generator(list1, lits2):
    '''
    Auxiliar generator: returns the ith element of both given list with each call to next() 
    '''
    for x,y in zip(list1,lits2):
        yield x, y
          
def saveFeatures(feature_extractor, features_file, labels_file):
    '''
    Function to load the optical flow stacks, do a feed-forward through the feature extractor (VGG16) and
    store the output feature vectors in the file 'features_file' and the labels in 'labels_file'.
    Input:
    * feature_extractor: model VGG16 until the fc6 layer.
    * features_file: path to the hdf5 file where the extracted features are going to be stored
    * labels_file: path to the hdf5 file where the labels of the features are going to be stored
    * features_key: name of the key for the hdf5 file to store the features
    * labels_key: name of the key for the hdf5 file to store the labels
    '''
    data_folder = '/ssd_drive/MultiCam_OF2/'
    mean_file = '/ssd_drive/flow_mean.mat'
    L = 10 
    
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
    
    idx_falls, idx_nofalls = 0, 0
    # For each stage7scenario
    for stage, nb_stage in zip(stages, range(len(stages))):
        h5features.create_group(stage)
        h5labels.create_group(stage)  
        cameras = glob.glob(data_folder + stage + '/cam*')
        cameras.sort()
        for camera, nb_camera in zip(cameras, range(1, len(cameras)+1)):
            h5features[stage].create_group('cam{}'.format(nb_camera))
            h5labels[stage].create_group('cam{}'.format(nb_camera))
            not_falls = glob.glob(camera + '/NotFalls/notfall*'.format(nb_camera))
            not_falls.sort()
            print(camera + '/NotFalls/notfall*'.format(nb_camera), len(not_falls))
            for not_fall in not_falls:
                label = 1
                x_images = glob.glob(not_fall + '/flow_x*.jpg')
                x_images.sort()
                y_images = glob.glob(not_fall + '/flow_x*.jpg')
                y_images.sort()
                nb_stacks = int(len(x_images))-L+1
                
                features_notfall = h5features[stage]['cam{}'.format(nb_camera)].create_dataset('notfall{:04}'.format(idx_nofalls), shape=(nb_stacks, num_features), dtype='float64')
                labels_notfall = h5labels[stage]['cam{}'.format(nb_camera)].create_dataset('notfall{:04}'.format(idx_nofalls), shape=(nb_stacks, 1), dtype='float64')
                idx_nofalls += 1
                
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
                del predictions, truth, flow, features_notfall, labels_notfall, x_images, y_images, nb_stacks
                gc.collect()
                
            if stage == 'chute24':
                continue
            
            falls = glob.glob(camera + '/Falls/fall*'.format(nb_camera))
            print(camera + '/Falls/fall*'.format(nb_camera), len(falls))
            falls.sort()
            h5features.close()
            h5labels.close()
            h5features = h5py.File(features_file,'a')
            h5labels = h5py.File(labels_file,'a')
            for fall in falls:     
                label = 0
                x_images = glob.glob(fall + '/flow_x*.jpg')
                x_images.sort()
                y_images = glob.glob(fall + '/flow_y*.jpg')
                y_images.sort()
                nb_stacks = int(len(x_images))-L+1
                
                features_fall = h5features[stage]['cam{}'.format(nb_camera)].create_dataset('fall{:04}'.format(idx_falls), shape=(nb_stacks, num_features), dtype='float64')
                labels_fall = h5labels[stage]['cam{}'.format(nb_camera)].create_dataset('fall{:04}'.format(idx_falls), shape=(nb_stacks, 1), dtype='float64')
                idx_falls += 1
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
                features_fall[:,:] = predictions
                labels_fall[:,:] = truth
                del predictions, truth, flow, features_fall, labels_fall
               
    h5features.close()
    h5labels.close()

def main(learning_rate, mini_batch_size, batch_norm, weight_0, epochs, model_file, weights_file): 
    exp = 'lr{}_batchs{}_batchnorm{}_w0_{}'.format(learning_rate, mini_batch_size, batch_norm, w0)
    vgg_16_weights = '/home/anunez/project/weights.h5'
    balance_dataset = True
    save_plots = True
    num_features = 4096
    features_file = '/home/anunez/project/features_multicam.h5'
    labels_file = '/home/anunez/project/labels_multicam.h5'
    save_features = True
           
    # =============================================================================================================
    # VGG-16 ARCHITECTURE
    # =============================================================================================================
    model = Sequential()
    
    model.add(ZeroPadding2D((1, 1), input_shape=(20, 224, 224)))
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
   
    layerscaffe = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'conv3_3', 'conv4_1', 'conv4_2', 'conv4_3', 'conv5_1', 'conv5_2', 'conv5_3', 'fc6', 'fc7', 'fc8']
    i = 0
    h5 = h5py.File(vgg_16_weights)
   
    layer_dict = dict([(layer.name, layer) for layer in model.layers])

    for layer in layerscaffe[:-3]:
        w2, b2 = h5['data'][layer]['0'], h5['data'][layer]['1']
        w2 = np.transpose(np.asarray(w2), (0,1,2,3))
        w2 = w2[:, :, ::-1, ::-1]
        b2 = np.asarray(b2)
        layer_dict[layer].W.set_value(w2)
        layer_dict[layer].b.set_value(b2)
        i += 1
        
    layer = layerscaffe[-3]
    w2, b2 = h5['data'][layer]['0'], h5['data'][layer]['1']
    w2 = np.transpose(np.asarray(w2), (1,0))
    b2 = np.asarray(b2)
    layer_dict[layer].W.set_value(w2)
    layer_dict[layer].b.set_value(b2)
    i += 1
    
    adam = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0005)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    # =============================================================================================================
    # FEATURE EXTRACTION
    # =============================================================================================================
    if save_features:
        saveFeatures(model, features_file, labels_file)

    # =============================================================================================================
    # TRAINING
    # =============================================================================================================

    do_training = True
    compute_metrics = True
    threshold = 0.5
    e = EarlyStopping(monitor='val_loss', min_delta=0, patience=100, verbose=0, mode='auto')
    
    if do_training:
        h5features = h5py.File(features_file, 'r')
        h5labels = h5py.File(labels_file, 'r')
  
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
                    cams_x[nb_cam] = np.concatenate([cams_x[nb_cam], temp_x], axis=0)
                    cams_y[nb_cam] = np.concatenate([cams_y[nb_cam], temp_y], axis=0)
                
        sensitivities = []
        specificities = []
        aucs = []
        accuracies = []
        # LEAVE-ONE-OUT
        for cam in range(8):
            print('='*30)
            print('LEAVE-ONE-OUT STEP {}/8'.format(cam+1))
            print('='*30)
            test_x = cams_x[cam]
            test_y = cams_y[cam]
            train_x = cams_x[0:cam] + cams_x[cam+1:]
            train_y = cams_y[0:cam] + cams_y[cam+1:]
           
            X = []
            _y = []
            for cam_x, cam_y in zip(train_x, train_y):
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
            
            # ==================== CLASSIFIER ========================               
            extracted_features = Input(shape=(4096,), dtype='float32', name='input')
            if batch_norm:
                x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(extracted_features)
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
            
            classifier = Model(input=extracted_features, output=x, name='classifier')
            classifier.compile(optimizer=adam, loss='binary_crossentropy',  metrics=['accuracy'])
            
            # ==================== TRAINING ========================         
            class_weight = {0:weight_0, 1:1}

            if mini_batch_size == 0:
                history = classifier.fit(X, _y, validation_data=(X2, _y2), batch_size=X.shape[0], nb_epoch=epochs, shuffle=True, class_weight=class_weight, callbacks=[e])
            else:
                history = classifier.fit(X, _y, validation_data=(X2, _y2), batch_size=mini_batch_size, nb_epoch=epochs, shuffle=True, class_weight=class_weight, callbacks=[e])
            
            # ==================== EVALUATION ========================        
            if compute_metrics:
               predicted = classifier.predict(X2)
               print(len(predicted))
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
               precision = tp/float(tp+fp)
               recall = tp/float(tp+fn)
               specificity = tn/float(tn+fp)
               f1 = 2*float(precision*recall)/float(precision+recall)
               accuracy = accuracy_score(_y2, predicted)
               fpr, tpr, _ = roc_curve(_y2, predicted)
               roc_auc = auc(fpr, tpr)
               print('TP: {}, TN: {}, FP: {}, FN: {}'.format(tp,tn,fp,fn))
               print('TPR: {}, TNR: {}, FPR: {}, FNR: {}'.format(tpr,tnr,fpr,fnr))   
               print('Sensitivity/Recall: {}'.format(recall))
               print('Specificity: {}'.format(specificity))
               print('Precision: {}'.format(precision))
               print('F1-measure: {}'.format(f1))
               print('Accuracy: {}'.format(accuracy))
               print('AUC: {}'.format(roc_auc))
               sensitivities.append(tp/float(tp+fn))
               specificities.append(tn/float(tn+fp))
               aucs.append(roc_auc)
               accuracies.append(accuracy)

        print('LEAVE-ONE-OUT RESULTS ===================')
        print("Sensitivity: %.2f%% (+/- %.2f%%)" % (np.mean(sensitivities), np.std(sensitivities)))
        print("Specificity: %.2f%% (+/- %.2f%%)" % (np.mean(specificities), np.std(specificities)))
        print("AUC: %.2f%% (+/- %.2f%%)" % (np.mean(aucs), np.std(aucs)))
        print("accuracy: %.2f%% (+/- %.2f%%)" % (np.mean(accuracies), np.std(accuracies)))
        
if __name__ == '__main__':
    if not os.path.exists('models'):
        os.makedirs('models')
    if not os.path.exists('weights'):
        os.makedirs('weights')
    model_file = 'models/exp_'
    weights_file = 'weights/exp_'
    batch_norm = True
    learning_rate = 0.01
    mini_batch_size = 1024
    w0 = 1
    epochs = 3000
  
    main(learning_rate, mini_batch_size, batch_norm, w0, epochs, model_file, weights_file)

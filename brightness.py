import cv2
import os
import sys
import numpy as np

data_folder = '/home/adrian/project/FALL/FDD_Fall/'
output_path = '/home/adrian/project/FALL/FDD_Fall_Dynamic_2/'

if not os.path.exists(output_path):
    os.makedirs(output_path)

folders = [f for f in os.listdir(data_folder + 'Falls') if os.path.isdir(os.path.join(data_folder + 'Falls', f))]
brightness = 50 # 50, 100
step = 0.1

brightness_add = np.zeros((224,224,3))
brightness_add.fill(brightness)
n = 0

for folder in folders:
    print('=========== {}'.format(folder))
    images = [f for f in os.listdir(data_folder + 'Falls/' + folder) if os.path.isfile(os.path.join(data_folder + 'Falls/' + folder, f))]
    images.sort()
    n = 0
    t = True
    for image, nb_image in zip(images, range(len(images))):
        x = cv2.imread(data_folder + 'Falls/' + folder + '/' + image)
        if brightness*np.sin(n)<0:
            t = False
        if t:
            x = np.add(x, brightness_add*np.sin(n))
            n += step
        if not os.path.exists(output_path + 'Falls/' + folder):
            os.makedirs(output_path + 'Falls/' + folder)
        cv2.imwrite(output_path + 'Falls/' + folder + '/' + image, x, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                
folders = [f for f in os.listdir(data_folder + 'NotFalls') if os.path.isdir(os.path.join(data_folder + 'NotFalls', f))]

for folder in folders:
    print('=========== {}'.format(folder))
    images = [f for f in os.listdir(data_folder + 'NotFalls/' + folder) if os.path.isfile(os.path.join(data_folder + 'NotFalls/' + folder, f))]
    images.sort()
    n = 0
    t = True
    for image, nb_image in zip(images, range(len(images))):
        x = cv2.imread(data_folder + 'NotFalls/' + folder + '/' + image) 
        if brightness*np.sin(n)<0:
            t = False
        if t:
            x = np.add(x, brightness_add*np.sin(n))
            n += step
        if not os.path.exists(output_path + 'NotFalls/' + folder):
            os.makedirs(output_path + 'NotFalls/' + folder)
        cv2.imwrite(output_path + 'NotFalls/' + folder + '/' + image, x, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
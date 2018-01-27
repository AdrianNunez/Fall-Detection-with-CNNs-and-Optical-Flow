import cv2
import os
import sys
import numpy as np

## VARIABLES TO CHANGE
data_folder = 'FDD_Fall/'
output_path = 'FDD_Fall_Dynamic/'
mode = 'darken'

if not os.path.exists(output_path):
    os.makedirs(output_path)

# EXPERIMENT 4.5.1: Matrix to be subtracted to the images to make them darker
darkness = 100
darkness_add = np.zeros((224,224,3))
darkness_add.fill(brightness)

# EXPERIMENT 4.5.2: Lighting change, the sin function is applied to the variable n, step is added to n after each frame
# n_start is the starting value of n
# Therefore, n=n_start in the frame 0, n=n_start+step in frame 1, n=n_start+2*step in frame 2 and so on
# The final transformation is the matrix brightness*sin(n) added to the original image
brightness = 50
brightness_add = np.zeros((224,224,3))
brightness_add.fill(brightness)

step = 0.1
n_start = 0

# Process the folders inside the 'Falls' directory
folders = [f for f in os.listdir(data_folder + 'Falls') if os.path.isdir(os.path.join(data_folder + 'Falls', f))]
for folder in folders:
    images = [f for f in os.listdir(data_folder + 'Falls/' + folder) if os.path.isfile(os.path.join(data_folder + 'Falls/' + folder, f))]
    images.sort()
    n = n_start
    stop = False

    # Process all the images of a video
    for image, nb_image in zip(images, range(len(images))):
        x = cv2.imread(data_folder + 'Falls/' + folder + '/' + image)
	if mode == 'darken':
	    x = np.add(x, darkness_add)
	else:
	    # if sin(n) is below 0 (the light is switched off) remove it
	    if brightness*np.sin(n) < 0:
	        stop = True
	    if stop:
	        x = np.add(x, brightness_add*np.sin(n))
	        n += step
        if not os.path.exists(output_path + 'Falls/' + folder):
            os.makedirs(output_path + 'Falls/' + folder)
        cv2.imwrite(output_path + 'Falls/' + folder + '/' + image, x, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

# Process the folders inside the 'NotFalls' directory          
folders = [f for f in os.listdir(data_folder + 'NotFalls') if os.path.isdir(os.path.join(data_folder + 'NotFalls', f))]
for folder in folders:
    images = [f for f in os.listdir(data_folder + 'NotFalls/' + folder) if os.path.isfile(os.path.join(data_folder + 'NotFalls/' + folder, f))]
    images.sort()
    n = 0
    stop = True
    # Process all the images of a video
    for image, nb_image in zip(images, range(len(images))):
        x = cv2.imread(data_folder + 'NotFalls/' + folder + '/' + image)      
	if mode == 'darken':
	    x = np.add(x, darkness_add)
	else:
	    # if sin(n) is below 0 (the light is switched off) remove it
	    if brightness*np.sin(n) < 0:
	        stop = True
	    if stop:
	        x = np.add(x, brightness_add*np.sin(n))
	        n += step
        if not os.path.exists(output_path + 'NotFalls/' + folder):
            os.makedirs(output_path + 'NotFalls/' + folder)
        cv2.imwrite(output_path + 'NotFalls/' + folder + '/' + image, x, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

print('=== END ===')

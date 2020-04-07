
import os
import cv2
import glob
import sys
from tqdm import tqdm

data_folder = 'Multicam_images/'
output_path = 'Multicam_OF/'

if not os.path.exists(output_path):
    os.mkdir(output_path)

folders = [f for f in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, f))]
folders.sort()
for folder in tqdm(folders):
    camera_folders = [f for f in os.listdir(data_folder + folder) if os.path.isdir(os.path.join(data_folder + folder + '/', f))]
    camera_folders.sort()
    for camera_folder in camera_folders:
        path = data_folder + folder + '/' + camera_folder
        event_folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path + '/', f))]
        event_folders.sort()
        for event_folder in event_folders:
            path = data_folder + folder + '/' + camera_folder + '/' + event_folder
            flow = output_path + folder + '/' + camera_folder + '/' + event_folder
            if not os.path.exists(flow):
                os.makedirs(flow)
            os.system('dense_flow2/build/extract_cpu -f={} -x={} -y={} -i=tmp/image -b=20 -t=1 -d=0 -s=1 -o=dir'.format(path, flow + '/flow_x', flow + '/flow_y'))
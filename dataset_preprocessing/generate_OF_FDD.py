
import os
import cv2
import glob
import sys
sys.path.append('/usr/local/lib/python2.7/site-packages/')
#os.system('/home/adrian/dense_flow/build/extract_cpu -f={} -x={} -y={} -i=tmp/image -b 20 -t 1 -d 3 -o=dir'.format('test.avi', '/flow_x', '/flow_y'))

data_folder = 'FDD_images/'
output_path = 'FDD_OF/'
i = 0

if not os.path.exists(output_path):
    os.mkdir(output_path)

folders = [f for f in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, f))]
folders.sort()
for folder in folders:
    event_folders = [f for f in os.listdir(data_folder + folder) if os.path.isdir(os.path.join(data_folder + folder + '/', f))]
    event_folders.sort()
    for event_folder in event_folders: 
	    path = data_folder + folder + '/' + event_folder
	    flow = output_path + folder + '/' + event_folder
	    if not os.path.exists(flow):
		    os.makedirs(flow)
        os.system('/home/anunez/dense_flow2/build/extract_cpu -f={} -x={} -y={} -i=tmp/image -b=20 -t=1 -d=0 -s=1 -o=dir'.format(path, flow + '/flow_x', flow + '/flow_y'))

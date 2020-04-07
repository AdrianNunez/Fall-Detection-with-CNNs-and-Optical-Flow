
import os
import cv2
import glob
import sys
sys.path.append('/usr/local/lib/python2.7/site-packages/')
#os.system('/home/adrian/dense_flow/build/extract_cpu -f={} -x={} -y={} -i=tmp/image -b 20 -t 1 -d 3 -o=dir'.format('test.avi', '/flow_x', '/flow_y'))

data_folder = '/home/anunez/UR_Fall/'
output_path = '/home/anunez/UR_Fall_opticalflow/'
i = 0

#activity_folders.sort()
#for actor_folder in actor_folders:
#    print(i, actor_folder)
#    i += 1
if not os.path.exists(output_path):
    os.mkdir(output_path)
    #video_folders = [f for f in os.listdir(data_folder + activity_folder + '/Videos') if os.path.isfile(os.path.join(data_folder + activity_folder + '/Videos/', f))]
    #print(data_folder + activity_folder + '/Videos')
    #video_folders.sort()
    #print(len(video_folders))
#    action_folders = [f for f in os.listdir(data_folder + actor_folder) if os.path.isdir(os.path.join(data_folder + actor_folder, f))]
#    action_folders.sort()


folders = [f for f in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, f))]
folders.sort()
for folder in folders:
    #if actor_folder != 'Ahmad': continue
    event_folders = [f for f in os.listdir(data_folder + folder) if os.path.isdir(os.path.join(data_folder + folder + '/', f))]
    event_folders.sort()
    for event_folder in event_folders:
        
	    path = data_folder + folder + '/' + event_folder
	    #vid_file = data_folder + activity_folder  + '/Videos/' + video_folder.replace(' ','')
	    flow = output_path + folder + '/' + event_folder
	    if not os.path.exists(flow):
		os.makedirs(flow)
	    print(flow, path)
            #print(vid_file, flow)
            #if not os.path.exists(flow):
            #    os.makedirs(flow)
            #else:
            #    files = [f for f in os.listdir(flow) if os.path.isfile(os.path.join(flow + '/', f))]       
            #    if len(files) > 0:
            #        continue
            #os.system('/home/adrian/dense_flow/build/extract_cpu -f={} -x={} -y={} -i=tmp/image -b=20 -t=1 -d=0 -s=1 -o=dir'.format(video_folder, flow + '/flow_x', flow + '/flow_y'))
            os.system('/home/anunez/dense_flow2/build/extract_cpu -f={} -x={} -y={} -i=tmp/image -b=20 -t=1 -d=0 -s=1 -o=dir'.format(path, flow + '/flow_x', flow + '/flow_y'))

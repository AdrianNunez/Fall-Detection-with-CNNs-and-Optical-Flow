import os
import cv2
import sys
import zipfile

# Path where the videos are stored (in zip format, as in a fresh download)
data_folder = '/home/user/Downloads/'
dst_folder = '' # folder where the dataset is going to be unzipped
output_base_path = 'FDD_images/'
files = ['Coffee_room_01.zip', 'Coffee_room_02.zip',
         'Home_01.zip', 'Home_02.zip']
W, H = 224, 224 # shape of new images (resize is applied)

if not os.path.exists(dst_folder):
    os.makedirs(dst_folder)

# Extract zip files (if necessary)
for f in files:
    filepath = data_folder + f
    zfile = zipfile.ZipFile(filepath)
    pos = filepath.rfind('/')
    if not os.path.exists(dst_folder + filepath[pos+1:-4]):
        zfile.extractall(dst_folder)

# Process each scenario
scenario_folders = [f for f in os.listdir(dst_folder) 
                if os.path.isdir(os.path.join(dst_folder, f))]
nb_fall_frames, nb_notfall_frames = 0, 0
num_frames = 0
for scenario_folder in scenario_folders:
    path = dst_folder + scenario_folder + '/'
    videos = sorted([f for f in os.listdir(path + 'Videos/') 
        if os.path.isfile(os.path.join(path + 'Videos/', f))])
    annotations = sorted([f for f in os.listdir(path + 'Annotation_files/') 
        if os.path.isfile(os.path.join(path + 'Annotation_files/', f))])
    
    # For each video and its corresponding labels
    for video, annotation in zip(videos, annotations):
        #print(video, annotation)
        lines = open(
                path + 'Annotation_files/' + annotation, 'r'
        ).readlines()
        # Get frames of start and end of the fall (if any, otherwise 0)
        fall_start, fall_end = 0, 0
        start_found = False
        for line in lines:
            #print(line, fall_start, fall_end, start_found)
            if len(line.split(',')) == 1:
                if not start_found:
                    fall_start = int(line.strip())
                    start_found = True
                else:
                    fall_end = int(line.strip())
                    break
        if fall_end > 0:
            num_frames += (fall_end-fall_start+1)
        fall_start -= 1
        fall_end -= 1
        #print(path + 'Videos/' + video)
        
        cap = cv2.VideoCapture(path + 'Videos/' + video)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Extract pre-fall frames
        num = video[video.find('(')+1:video.find(')')]
        output_path = output_base_path + 'NotFalls/{}_folder_video_{}/'.format(
            scenario_folder, num)
        pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
        if not os.path.exists(output_path): os.makedirs(output_path)
        while pos < fall_start:
            ret, frame = cap.read()
            # Save frame
            cv2.imwrite(output_path + 'img_{:05d}.jpg'.format(int(pos+1)),
                        cv2.resize(frame, (W,H)),
                        [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            nb_notfall_frames += 1
            pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
        # Extract fall frames
        if fall_end > 0:
            output_path = output_base_path + 'Falls/{}_folder_video_{}/'.format(
                scenario_folder, num)
            if not os.path.exists(output_path): os.makedirs(output_path)
            #cap.set(cv2.CAP_PROP_POS_FRAMES, fall_start)
        
        while pos <= fall_end:
            #print(scenario_folder,video, length, fall_start, pos, fall_end)
            ret, frame = cap.read()
            # Save frame
            cv2.imwrite(output_path + 'img_{:05d}.jpg'.format(int(pos+1)),
                        cv2.resize(frame, (W,H)),
                        [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            nb_fall_frames += 1
            pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
        # Extract post-fall frames
        output_path = output_base_path + 'NotFalls/{}_folder_video_{}/'.format(
            scenario_folder, num)
        #cap.set(cv2.CAP_PROP_POS_FRAMES, fall_end)
        while pos < length:
            ret, frame = cap.read()
            # Save frame
            cv2.imwrite(output_path + 'img_{:05d}.jpg'.format(int(pos+1)),
                        cv2.resize(frame, (W,H)),
                        [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            nb_notfall_frames += 1
            pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
        cap.release()
        cv2.destroyAllWindows()

print('Number of fall frames: {}, not fall frames: {}'.format(
    nb_fall_frames, nb_notfall_frames
))
print(num_frames)
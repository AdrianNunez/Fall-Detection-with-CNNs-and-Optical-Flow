import os
import cv2
import sys
import json
import glob
import shutil
import codecs
import zipfile

# Path where the videos are stored (in zip format, as in a fresh download)
data_folder = '/home/user/Downloads/'
dst_folder = '' # folder where the dataset is going to be unzipped
output_base_path = 'Multicam_images/'
fall_annotations_file = 'annotations_multicam.json'
delays_file = 'delays_multicam.json'
num_scenarios = 24
num_cameras = 8
W, H = 224, 224 # shape of new images (resize is applied)

# Extract all files and organise them (if necessary)
if len(glob.glob(dst_folder + '*')) == 0:
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    filepath = data_folder + 'dataset.zip'
    zfile = zipfile.ZipFile(filepath)
    pos = filepath.rfind('/')

    zfile.extractall(dst_folder)
    dir = glob.glob(dst_folder + '*')
    for path in glob.glob(dir[0] + '/*'):
        save_path = dst_folder + path[path.rfind('/')+1:]
        os.makedirs(save_path)
        for filePath in glob.glob(path + '/*'):
            shutil.move(filePath, save_path)
    shutil.rmtree(dir[0])

with open(fall_annotations_file, 'r') as json_file:
    annotations = json.load(json_file)
with open(delays_file, 'r') as json_file:
    delays = json.load(json_file)

# For each scenario
for s in range(1,num_scenarios+1):
    # Get all videos (one per camera)
    videos = glob.glob(dst_folder + 'chute{:02}/*'.format(s))
    videos.sort()
    starts = annotations['scenario{}'.format(s)]['start']
    ends = annotations['scenario{}'.format(s)]['end']

    for cam, video in enumerate(videos, 1):
        cap = cv2.VideoCapture(video)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Delay of this camara for this scenario
        delay = delays['camera{}'.format(cam)][str(s)]
        for start, end in zip(starts, ends):
            # Apply the delay
            start += delay
            end += delay
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            # Read the pre-fall part
            pos = 0
            while pos < start:
                ret, frame = cap.read()
                pos += 1
                output_path = (
                    output_base_path +
                    'chute{:02}/NotFalls/camera{}_pre/'.format(
                    s, cam
                ))
                if not os.path.exists(output_path):
                    os.makedirs(output_path)
                cv2.imwrite(output_path + 'img_{:05d}.jpg'.format(int(pos)),
                        cv2.resize(frame, (W,H)),
                        [int(cv2.IMWRITE_JPEG_QUALITY), 95])
  
            # Read the fall part
            assert cap.get(cv2.CAP_PROP_POS_FRAMES) == start
            while pos <= end:
                ret, frame = cap.read()
                pos += 1
                output_path = (
                    output_base_path +
                    'chute{:02}/Falls/camera{}/'.format(
                    s, cam
                ))
                if not os.path.exists(output_path):
                    os.makedirs(output_path)
                cv2.imwrite(output_path + 'img_{:05d}.jpg'.format(int(pos)),
                        cv2.resize(frame, (W,H)),
                        [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            # Read the post-fall part
            assert cap.get(cv2.CAP_PROP_POS_FRAMES) == end + 1
            while pos < length:
                ret, frame = cap.read()
                pos += 1
                output_path = (
                    output_base_path +
                    'chute{:02}/NotFalls/camera{}_post/'.format(
                    s, cam
                ))
                if not os.path.exists(output_path):
                    os.makedirs(output_path)
                cv2.imwrite(output_path + 'img_{:05d}.jpg'.format(int(pos)),
                        cv2.resize(frame, (W,H)),
                        [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        # If there was no fall
        if len(starts) == 0 and len(ends) == 0:
            pos = 0
            while pos < length:
                ret, frame = cap.read()
                pos += 1
                output_path = (
                    output_base_path +
                    'chute{:02}/NotFalls/camera{}_full/'.format(
                    s, cam
                ))
                if not os.path.exists(output_path):
                    os.makedirs(output_path)
                cv2.imwrite(output_path + 'img_{:05d}.jpg'.format(int(pos)),
                        cv2.resize(frame, (W,H)),
                        [int(cv2.IMWRITE_JPEG_QUALITY), 95])
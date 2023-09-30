import torch
from s3dg import S3D
import cv2
import numpy as np
import argparse

# load video
parser = argparse.ArgumentParser()
parser.add_argument("--video_path", required=True)
parser.add_argument("--num_frames", required=True)
args = parser.parse_args()

VIDEO_PATH = args.video_path
NUM_FRAMES = args.num_frames


cap = cv2.VideoCapture(VIDEO_PATH)
frames = []
while True:
    ret, frame = cap.read()
    if not ret or len(frames) == NUM_FRAMES:
        break
    frame = cv2.resize(frame, (224, 224))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame / 255.0
    frames.append(frame)


cap.release()

video = np.array(frames)
video = np.transpose(video, (3, 0, 1, 2))
video = video.reshape((-1, 3, 32, 224, 224))
video = torch.from_numpy(video).float()

print(video.shape)


net = S3D('s3d_dict.npy', 512)
net.load_state_dict(torch.load('s3d_howto100m.pth'))

net = net.eval()
video_output = net(video)

print(video_output["video_embedding"].shape)
import torch
from s3dg import S3D
import cv2
import numpy as np
import argparse
from tqdm import tqdm

# load video
parser = argparse.ArgumentParser()
parser.add_argument("--video_path", required=True)
parser.add_argument("--num_frames", required=True, type=int)
parser.add_argument("--output_path", default="features/output.npy")
args = parser.parse_args()

VIDEO_PATH = args.video_path
NUM_FRAMES = args.num_frames
OUTPUT_PATH = args.output_path

if NUM_FRAMES % 32 != 0:
    raise ValueError("NUM_FRAMES must be a multiple of 32!")

net = S3D('s3d_dict.npy', 1024)
net.load_state_dict(torch.load('s3d_howto100m.pth'))
net = net.eval()

cap = cv2.VideoCapture(VIDEO_PATH)
outputs = []
for i in tqdm(range(NUM_FRAMES//32)):
    frames = []
    for j in range(32):
        ret, frame = cap.read()
        frame = cv2.resize(frame, (224, 224))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame / 255.0
        frames.append(frame)
    video = np.array(frames)
    video = np.transpose(video, (3, 0, 1, 2))
    video = video.reshape((-1, 3, 32, 224, 224))
    video = torch.from_numpy(video).float()
    with torch.no_grad():
        output_chunk = net(video)
    outputs.append(output_chunk["video_embedding"])

cap.release()

video_output_embedding = torch.cat(outputs, dim=0)
print(video_output_embedding.shape)

np.save(OUTPUT_PATH, video_output_embedding.cpu().numpy())
import sys
import os
import torch
import cv2
from torchvision.models.detection.keypoint_rcnn import KeypointRCNNPredictor
import torchvision
import numpy as np
from torchvision.io import read_image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.CustomKeypointRCNN import CustomKeypointRCNN
from utils.MiDaSInferenceLocal import MiDaSInferenceLocal
from utils.Helper_functions import (
    load_intrinsic_parameters_from_json,
    draw_detections_and_keypoints,
    get_keypoint_depths,
    combine_keypoints_with_depth,
    plot_3d_points_with_lines,
    convert_to_3d_camera_coords,
)

cam_params = load_intrinsic_parameters_from_json(
    "./Camera_Calibrations\intrinsic_parameters.json"
)

# model_2d_pose = CustomKeypointRCNN()
# model_2d_pose.save('./models/keypoint_rcnn/model.pt')a
# model_2d_pose.load(".\models\keypoint_rcnn\model.ptmodel.pt")
model_2d_pose = torchvision.models.detection.keypointrcnn_resnet50_fpn(
    weights="KeypointRCNN_ResNet50_FPN_Weights.DEFAULT"
)

model_2d_pose.eval()
model_2d_pose.to("cuda")
model_midas = MiDaSInferenceLocal()
# cap = cv2.VideoCapture(
#     "C:\\Users\\ksa_j\\Documents\\python\\SDAIA_Capstone_Project\\Dataset\\HumanSC3D\\train\\s01\\videos\\50591643\\001.mp4"
# )
cap = cv2.VideoCapture(1)


def pose_2d_handler(pose: dict, frame):
    return pose["labels"], pose["boxes"], pose["keypoints"]


fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

while True:
    ret, frame = cap.read()
    if ret:
        frame = cv2.resize(frame, (800, 800))
        frame_copy = frame.copy()
        frame = frame.astype(np.float32) / 255
        frame = torch.tensor(frame.transpose((2, 0, 1))).float().unsqueeze(0)
        frame = frame.to("cuda")
        with torch.no_grad():
            pose = model_2d_pose(frame)
            depth = model_midas.infer(frame_copy)
        frame = frame.squeeze().permute(1, 2, 0).cpu().numpy()
        frame, keypoints = draw_detections_and_keypoints(frame, pose[0])
        depth = model_midas.postprocess(depth)
        pose_3d = combine_keypoints_with_depth(keypoints, depth)
        pose_3d = convert_to_3d_camera_coords(pose_3d, cam_params)
        # print(pose_3d)
        plot_3d_points_with_lines(ax, pose_3d)
        # print(pose_3d)
        # tensor to numpy
        # frame = frame.squeeze().permute(1, 2, 0).cpu().numpy()
        # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        print(frame.shape)
        print(depth.shape)
        cv2.imshow("Image", frame)
        cv2.imshow("Depth", depth)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


cap.release()

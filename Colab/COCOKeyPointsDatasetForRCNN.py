import os
from pycocotools.coco import COCO
import skimage.io as io
import torch
import torchvision.transforms.functional as F
from torch.utils.data import Dataset
from skimage.transform import resize
import cv2
import torchvision.transforms.functional as F
import numpy as np


def convert_coco_bbox_to_xyxy(bbox):
    return [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]


# def process_image_and_annotations(
#     image, boxes_list, keypoints, image_width, image_height
# ):
#     # Resize the image
#     image = cv2.resize(image, (512, 512))

#     # Calculate scaling factors for width and height
#     width_scale = 512 / image_width
#     height_scale = 512 / image_height

#     # Adjust bounding boxes
#     boxes_list = [
#         [
#             box[0] * width_scale,
#             box[1] * height_scale,
#             box[2] * width_scale,
#             box[3] * height_scale,
#         ]
#         for box in boxes_list
#     ]

#     # Adjust keypoints
#     scaling_tensor = torch.tensor([width_scale, height_scale, 1])
#     for i in range(len(keypoints)):
#         keypoints_tensor = torch.tensor(keypoints[i])
#         for j in range(0, len(keypoints_tensor), 3):
#             keypoints_tensor[j : j + 3] = keypoints_tensor[j : j + 3] * scaling_tensor
#         keypoints[i] = keypoints_tensor.tolist()

#     return image, boxes_list, keypoints


class COCOKeyPointsDatasetForRCNN(Dataset):
    def __init__(self, annotation_file, root_dir, transform=None):
        """
        Args:
            annotation_file (string): Path to the COCO annotations file.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.coco = COCO(annotation_file)
        self.root_dir = root_dir
        self.transform = transform

        # Get image IDs for 'person' category
        person_cat_ids = self.coco.getCatIds(catNms=["person"])
        person_image_ids = self.coco.getImgIds(catIds=person_cat_ids)

        # Filter image IDs to include only those that have keypoint annotations
        self.image_ids = person_image_ids

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        # Load image
        image_path = os.path.join(
            self.root_dir, "{:012}.jpg".format(self.image_ids[idx])
        )
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Get annotations for the image
        ann_ids = self.coco.getAnnIds(imgIds=self.image_ids[idx])
        annotations = self.coco.loadAnns(ann_ids)

        # Initialize lists to store bounding boxes, labels, masks, and keypoints
        boxes_list = []
        labels_list = []
        keypoints_list = []

        for annotation in annotations:
            # Bounding boxes
            boxes_list.append(convert_coco_bbox_to_xyxy(annotation["bbox"]))

            # Labels
            labels_list.append(annotation["category_id"])

            # Keypoints
            keypoints_list.append(annotation["keypoints"])

        # Convert lists to tensors
        boxes = torch.as_tensor(boxes_list, dtype=torch.float32)
        labels = torch.as_tensor(labels_list, dtype=torch.int64)
        keypoints = torch.as_tensor(keypoints_list, dtype=torch.float32)

        # Call the function to process the image and annotations
        # image, boxes_list, keypoints_list = process_image_and_annotations(
        #     image, boxes_list, keypoints_list, image.shape[1], image.shape[0]
        # )
        keypoints = keypoints.view(-1, 17, 3)
        # Create target dictionary
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["keypoints"] = keypoints

        # Convert image to tensor and return
        image = torch.as_tensor(image, dtype=torch.float32).permute(2, 0, 1)
        return image, target

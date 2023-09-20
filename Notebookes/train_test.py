import sys
import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.CustomKeypointRCNN import CustomKeypointRCNN
from utils.COCOKeyPointsDatasetForRCNN import COCOKeyPointsDatasetForRCNN

import os
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch


def collate_fn(batch):
    images, targets = zip(*batch)

    # Convert images to a list
    images = list(images)

    # Create the new targets list using individual tensors for each entry (not stacked)
    new_targets = []
    for target in targets:
        new_target = {}
        for k, v in target.items():
            new_target[k] = v
        new_targets.append(new_target)

    return images, new_targets


if __name__ == "__main__":
    train_annotation_file = ".//Dataset//annotations//person_keypoints_train2017.json"
    train_image_folder = ".//Dataset//train2017"
    val_annotation_file = ".//Dataset//annotations//person_keypoints_val2017.json"
    val_image_folder = ".//Dataset//val2017"

    train_set = COCOKeyPointsDatasetForRCNN(train_annotation_file, train_image_folder)
    val_set = COCOKeyPointsDatasetForRCNN(val_annotation_file, val_image_folder)

    print("Number of samples in training set: {}".format(len(train_set)))
    print("Number of samples in validation set: {}".format(len(val_set)))

    train_loader = DataLoader(
        train_set, batch_size=8, shuffle=True, num_workers=8, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_set, batch_size=8, shuffle=True, num_workers=8, collate_fn=collate_fn
    )

    model = CustomKeypointRCNN(num_keypoints=17)

    optimizer = Adam(model.model.parameters(), lr=0.0001)
    save_path = ".//models//keypoint_rcnn//"

    model.train(train_loader, val_loader, 10, optimizer, save_path)
    # model.save(save_path)

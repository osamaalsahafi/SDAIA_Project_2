import sys
import os
from torch.utils.data import Subset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from CustomKeypointRCNN import CustomKeypointRCNN
from COCOKeyPointsDatasetForRCNN import COCOKeyPointsDatasetForRCNN

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
    train_annotation_file = (
        "./content/Dataset/annotations/person_keypoints_train2017.json"
    )
    train_image_folder = "./content/Dataset/train2017"
    val_annotation_file = "./content/Dataset/annotations/person_keypoints_val2017.json"
    val_image_folder = "./content/Dataset/val2017"

    train_set = COCOKeyPointsDatasetForRCNN(train_annotation_file, train_image_folder)
    val_set = COCOKeyPointsDatasetForRCNN(val_annotation_file, val_image_folder)
    print("*" * 30)
    print("Number of entire in training set: {}".format(len(train_set)))
    print("Number of entire in validation set: {}".format(len(val_set)))
    # Define the size of the subset for training and validation
    subset_size_train = 500
    subset_size_val = 250

    # Generate random indices for train and validation sets
    train_indices = torch.randperm(len(train_set))[:subset_size_train]
    val_indices = torch.randperm(len(val_set))[:subset_size_val]

    train_subset = Subset(train_set, train_indices)
    val_subset = Subset(val_set, val_indices)
    print("*" * 30)
    print("Number of samples in training subset: {}".format(len(train_subset)))
    print("Number of samples in validation subset: {}".format(len(val_subset)))
    print("*" * 30)

    train_loader = DataLoader(
        train_subset, batch_size=8, shuffle=True, num_workers=8, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_subset, batch_size=8, shuffle=True, num_workers=8, collate_fn=collate_fn
    )

    model = CustomKeypointRCNN(num_keypoints=17)

    optimizer = Adam(model.model.parameters(), lr=0.00001)
    save_path = "./content/models/keypoint_rcnn"

    model.train(train_loader, val_loader, 10, optimizer, save_path)
    model.save(save_path)

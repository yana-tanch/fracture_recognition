import random

from pathlib import Path

import cv2
import torch
import numpy as np
import pandas as pd

from torch.utils import data
from sklearn.model_selection import train_test_split


class DataSet(data.Dataset):
    def __init__(self, params, X, t, augment):
        super().__init__()

        self.params = params
        self.augment = augment

        self.X = X
        self.t = t

    def __len__(self):
        return len(self.t)

    def __getitem__(self, item):
        X = self.X[item]
        t = self.t[item]

        bbox = X['bbox']
        image = cv2.imread(X['image'], 0)

        source = np.array([t, bbox[0, 0], bbox[0, 1], bbox[1, 0], bbox[1, 1]])

        if not self.augment:
            return torch.from_numpy(image), torch.from_numpy(source)

        degrees = self.params.augmentation.degrees
        scale = self.params.augmentation.scale
        translate = self.params.augmentation.translate

        # Center
        C = np.eye(3)

        C[0, 2] = -image.shape[1] / 2
        C[1, 2] = -image.shape[0] / 2

        # Rotation and Scale
        R = np.eye(3)

        a = random.uniform(-degrees, degrees)
        s = random.uniform(1 - scale, 1 + scale)

        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

        # Translation
        T = np.eye(3)

        width = image.shape[1]
        height = image.shape[0]

        T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width
        T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height

        # Combined  matrix
        M = T @ R @ C  # order of operations (right to left) is IMPORTANT

        # affine image
        warped_image = cv2.warpAffine(image, M[:2], dsize=(width, height), borderValue=(0, 0, 0))

        xy = bbox @ M.T
        x = xy[:, 0]
        y = xy[:, 1]

        x = x.clip(0, width - 1)
        y = y.clip(0, height - 1)

        x1 = x.min()
        y1 = y.min()
        x2 = x.max()
        y2 = y.max()

        target = np.array([t, x1, y1, x2, y2])

        return torch.from_numpy(warped_image), torch.from_numpy(target)


def get_datasets(params):
    dataset_dir = Path(params.dataset_dir)
    annotation = pd.read_csv(dataset_dir / 'annotation.csv')

    X = []
    t = []

    for target, label in enumerate(params.labels):
        for _, row in annotation.loc[annotation.label == label].iterrows():
            image_path = dataset_dir / 'images' / (str(row.id).zfill(6) + '.png')

            rec = dict(image=str(image_path),
                       bbox=np.array([[row.x1, row.y1, 1], [row.x2, row.y2, 1]])
                       )

            X.append(rec)
            t.append(target)

    X_train, X_test, t_train, t_test = train_test_split(X, t,
                                                        train_size=params.train_size,
                                                        random_state=params.seed,
                                                        stratify=t)

    train_dataset = DataSet(params, X_train, t_train, True)
    test_dataset = DataSet(params, X_test, t_test, False)

    return train_dataset, test_dataset


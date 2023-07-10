import csv
from pathlib import Path

import cv2
import hydra
import numpy as np
import pandas as pd
import pydicom
import pyrootutils

from tqdm import tqdm

pyrootutils.setup_root(__file__, indicator='.project', pythonpath=True)

from src.utils import make_dir


@hydra.main(version_base='1.3', config_path='../conf', config_name='config.yaml')
def main(cfg):
    params = cfg.prepare_dataset

    src_dir = Path(params.data_dir)
    dst_dir = Path(params.dataset_dir)

    class_annotation = pd.read_csv(src_dir / 'train.csv')
    boxes_annotation = pd.read_csv(src_dir / 'train_bounding_boxes.csv')

    ids = boxes_annotation['StudyInstanceUID'].unique()

    images_dir = make_dir(dst_dir / 'images')
    annot_file = open(dst_dir / 'annotation.csv', 'w')

    writer = csv.DictWriter(
        annot_file,
        fieldnames=['id', 'x1', 'y1', 'x2', 'y2', 'label']
    )
    writer.writeheader()

    classes = dict()
    counter = 0

    for patient_id in tqdm(ids, leave=False):
        labels = class_annotation.loc[class_annotation.StudyInstanceUID == patient_id,
                                    ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']]
        labels = list(labels.values[0])

        label = int(labels[0] * 64 +
                    labels[1] * 32 +
                    labels[2] * 16 +
                    labels[3] * 8 +
                    labels[4] * 4 +
                    labels[5] * 2 +
                    labels[6])

        df = boxes_annotation.loc[boxes_annotation.StudyInstanceUID == patient_id, :]

        for row_index, row in df.iterrows():
            x, y, w, h, s = row.x, row.y, row.width, row.height, row.slice_number

            image_path = src_dir / 'train_images' / patient_id / (str(s) + '.dcm')

            dicom_file = pydicom.dcmread(image_path)
            image = dicom_file.pixel_array

            if (image.shape[0] != 512) or (image.shape[1] != 512):
                continue

            # https://en.wikipedia.org/wiki/Hounsfield_scale
            image = image.clip(0)  # (300, 1900)
            image = (image - image.min()) / (image.max() - image.min())
            image = (255 * image).astype(np.uint8)

            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            image = clahe.apply(image)

            file_id = str(counter).zfill(6)

            out_image_path = images_dir / (file_id + '.png')
            cv2.imwrite(str(out_image_path), image)

            writer.writerow({'id': file_id,
                             'x1': int(x),
                             'y1': int(y),
                             'x2': int(x + w),
                             'y2': int(y + h),
                             'label': label}
            )

            if label not in classes:
                classes[label] = 1
            else:
                classes[label] += 1

            counter += 1

    writer = csv.DictWriter(
        open(dst_dir / 'stats.csv', 'w'),
        fieldnames=['label', 'num_images']
    )

    writer.writeheader()

    for label, num_images in classes.items():
        writer.writerow({'label': label, 'num_images': num_images})


if __name__ == '__main__':
    main()


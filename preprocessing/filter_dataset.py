import csv
import os
from typing import Union
from pathlib import Path
from tqdm import tqdm

import cv2


def filter_dataset(data_dir: str, threshold: Union[float, int] = 100.0) -> None:
    data_dir = Path(data_dir)
    threshold = threshold

    bad_images = []

    for d, dirs, files in tqdm(os.walk(data_dir)):
        for x in files:
            if x.endswith((".jpg", ".JPG", ".png", ".PNG")):
                img = cv2.imread((os.path.join(d, x)), 0)  # Read in as grayscale to check for blur
                img = cv2.equalizeHist(img)
                val = cv2.Laplacian(img, cv2.CV_64F).var()
                if threshold > val:
                    bad_images.append([(os.path.join(d, x)), val, (img > 10.0).any(), (img < 200.0).any()])
                elif not (img > 10.0).any() or not (img < 200.0).any():  # Ensuring image isn't white or black
                    bad_images.append([(os.path.join(d, x)), val, (img > 10.0).any(), (img < 200.0).any()])

    with open('output3.csv', 'w') as result_file:
        wr = csv.writer(result_file)
        wr.writerows(bad_images)


filter_dataset("/media/matt/4TBInternal/zzFootball/Games/")

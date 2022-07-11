import os
import numpy as np

from preprocessing.modification import blur_image, brightness_image, contrast_image, coloring_image

rng = np.random.default_rng(1234)
val_range = np.concatenate((np.linspace(1.1, 2), np.linspace(0.1, 1, endpoint=False)))

dir_path = "/media/matt/4TBInternal/zzFootball/1s"

for path, subdirs, files in os.walk(dir_path):
    for file in files:
        if file.endswith((".jpg", ".JPG", ".png", ".PNG")):
            level = rng.choice(val_range)
            blur_image(os.path.join(path, file), blur_type="box", radius=level)
            blur_image(os.path.join(path, file), blur_type="gauss", radius=level)
            brightness_image(os.path.join(path, file), level=level)
            contrast_image(os.path.join(path, file), level=level)
            coloring_image(os.path.join(path, file), level=level)

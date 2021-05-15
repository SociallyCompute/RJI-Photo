import os.path, sys, warnings

import numpy as np
from PIL import ImageFile
from skimage import feature

ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings('ignore')
sys.path.append(os.path.split(sys.path[0])[0])


def canny_edge_detection(image, sigma=None):
    gray_image = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140]) if image.ndim == 3 else image
    return feature.canny(gray_image)

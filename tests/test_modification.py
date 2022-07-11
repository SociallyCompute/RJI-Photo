from unittest import TestCase

from PIL import Image
from preprocessing.modification import blur_image, contrast_image, brightness_image, coloring_image


class Test(TestCase):
    def test_blur_image(self):
        test_dir = r'/media/matt/4TBInternal/zzFootball/20160910_MUvsEasternMichigan/For_Upload/'
        test_file = test_dir + '20160910_MUvsEasternMichigan_EC_0001.JPG'
        blur_image(test_file, "box", 5)

    def test_contrast_image(self):
        test_dir = r'/media/matt/4TBInternal/zzFootball/20160910_MUvsEasternMichigan/For_Upload/'
        test_file = test_dir + '20160910_MUvsEasternMichigan_EC_0001.JPG'
        contrast_image(test_file, 0.5)

    def test_brightness_image(self):
        test_dir = r'/media/matt/4TBInternal/zzFootball/20160910_MUvsEasternMichigan/For_Upload/'
        test_file = test_dir + '20160910_MUvsEasternMichigan_EC_0001.JPG'
        brightness_image(test_file, 0.5)

    def test_coloring_image(self):
        test_dir = r'/media/matt/4TBInternal/zzFootball/20160910_MUvsEasternMichigan/For_Upload/'
        test_file = test_dir + '20160910_MUvsEasternMichigan_EC_0001.JPG'
        coloring_image(test_file, 0.5)

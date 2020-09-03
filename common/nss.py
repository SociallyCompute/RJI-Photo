from PIL import Image
import os
import numpy as np

class ConvolutionalNSS:
    """
    Builds a mask and performs a convolution over a set of image. This can take
    the given values to form a convolutional map and then apply natural scene 
    statistics (NSS) techniques on the convolutional image.

    :param root: (String) path to root image folder
    :param mask_dim: (int) single dimension of mask
    :param mask_center: (int, int) coordinates locating center of 2d mask
    :param mask: (ndarray) mask_dim x mask_dim array (if even number is chosen use 3)
    :param image_path_list: (list) list of image paths
    :param image_dims: (int, int) max width and height that fits all loaded images   
    """

    def __init__(self, root, mask_dim):
        self.root = root
        self.mask_dim = mask_dim if mask_dim % 2 == 0 else 3
        self.mask_center = (mask_dim/2, mask_dim/2)
        self.mask = _generate_blank_mask()
        self.image_path_list = _generate_image_paths_list()
        self.image_dims = generate_image_dims(image_path_list[0])

    @staticmethod
    def generate_nparray_from_path(img_path, rgb=None):
        """
        method that takes an image path and generates a np array representing it
        static to generalize for any use to get an nparray from an image path

        :param img_path: (String) full path to image
        :param rgb: (String) identify if RGB image by setting to 'RGB'. Otherwise assumed black/white
        :rtype img_arr: (ndarray) matrix describing image
        """
        img = Image.open(img_path)
        img.convert(rgb) if rgb == 'RGB' else img.convert('LA')
        return np.array(img)

    def _generate_blank_mask(self):
        """
        Generate a mask to travel over each image. Defined as a perfect square
        and specified by the mask_dim variable.

        :rtype np_arr: (ndarray) returns array filled with 1 through (2*mask_dim)
        """
        t = []
        t2 = []
        for i in range(self.mask_dim): #generate all rows
            for j in range(self.mask_dim): #generate single row
                t.append((j+1) + (i*self.mask_dim))
            t2.append(np.asarray(t))
        np_arr = np.asarray(t2)
        return np_arr

    def _generate_image_paths_list(self):
        """
        Simply take the root path and recursively iterate through adding
        each of the files ending in PNG or JPG to an image_path_list
        This is done to keep track of each Image but avoid having to save
        them in large chunks of memory.

        :param root_path: (String) root path to folder
        """
        self.root = os.path.expanduser(self.root)
        for r, _, fnames in os.walk(self.root):
                for fname in sorted(fnames):
                    path = os.path.join(r, fname)
                    if path.lower().endswith(('.png', '.jpg')):
                        image_path_list.append(path)
        return image_path_list

    def generate_image_dims(self, img_path):
        """
        Generate base image dims based on the first image in the list. If a picture
        doesn't fit to these dims (i.e. too small), then this resizes the class 
        image dims so we get the largest possible dims to fit every picture

        :param img_path: (String) full path to image
        :rtype (width, height): (int, int) dimensions describing the largest dimensions to fit all images
        """
        img = Image.open(img_path)
        width, height = img.size
        if self.image_dims is not None: 
            width = self.image_dims[0] if self.image_dims[0] < width else width
            height = self.image_dims[1] if self.image_dims[1] < height else height
        return (width, height)

    def generate_convolution_image(self, img):
        """
        travel across the image and create a convolutional image. Start with center of mask
        in top left of image and travel across where last convolution is center on top right.
        Restart from 2nd row on left all the way down to bottom right.

        :param img: (ndarray) numpy array describing image
        :rtype conv_img: (ndarray) numpy array describing convolution performed on image
        """
        sum_total = 0
        conv_img = np.zeros(img.shape)

        if img.ndim == 3:
            for i in range(img.ndim):
                width, height = img[i].shape
                for j in range(width):
                    for k in range(height):
                        for x in range(mask_dim):
                            for y in range(mask_dim):
                                if (img[i])[(j-self.mask_center[0])+x][(k-self.mask_center[1])+y] is not None:
                                    sum_total += (self.mask[x][y] * (img[i])[(j-self.mask_center[0])+x][(k-self.mask_center[1])+y])
                        conv_img[i][j][k] = sum_total
        else:
            width, height = img.shape
            for j in range(width):
                for k in range(height):
                    for x in range(mask_dim):
                        for y in range(mask_dim):
                            if img[(j-self.mask_center[0])+x][(k-self.mask_center[1])+y] is not None:
                                sum_total += (self.mask[x][y] * img[(j-self.mask_center[0])+x][(k-self.mask_center[1])+y])
                    conv_img[j][k] = sum_total

        return conv_img

    def generate_convolution_image_list(self):
        """
        Build the list of convolution mappings

        :rtype conv_list: (list) contains convolutional mappings of each of the images specified by image_path_list
        """
        conv_list = []
        for im_path in self.image_path_list:
            conv_list.append(self.generate_convolution_image(generate_nparray_from_path(im_path, 'RGB')))
        return conv_list
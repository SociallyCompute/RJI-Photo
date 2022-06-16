import os
import warnings
from os import listdir
from os.path import isfile, join, isdir
from typing import Optional, Union, List, Dict

from PIL import Image


class MetaDataReader(object):
    def __init__(self, img_dir: Optional[str] = None):
        if img_dir is not None:
            if not isdir(img_dir):
                raise NotADirectoryError("The provided path must be an image directory")
            self.img_dir = img_dir
        else:
            self.img_dir = None

    def get_cc_from_file(self, path: str = None) -> Union[List[str], Dict[str, str], str]:
        """
        Either returns a list of possible files loaded to the class or returns the exif data for the provided file
        :param path: Filepath to Image File, can be either from home root or from root directory when executed
        :return: Either a list of files found that would work (if no path specified) or the exif data for the file
        """
        if path is None:
            warnings.warn('No path was specified, returning a list of possible images')
            if self.img_dir is None:
                raise NotADirectoryError('No directory specified for searching images')
            files = [f for f in listdir(self.img_dir) if isfile(join(self.img_dir, f))]
            return files
        if not isfile(path):
            raise FileNotFoundError('File not found')
        if path.endswith(('.png', '.PNG')):
            img = Image.open(path)
            img.load()
            exif = img.info
            img.close()
            return exif
        elif path.endswith(('.jpg', '.JPG')):
            with open(path, 'rb') as f:
                img = f.read()
            s = str(img).find('photomechanic:ColorClass')
            if s != -1:
                e = str(img).find('photomechanic:Tagged')
                if s != e:
                    cc = str(img)[s:e]
                    return cc[26]
            return "0"
        else:
            raise NotImplementedError('This file is not a JPG or PNG and thus not implemented')

    def get_cc_from_dir(self, dir_path: Optional[str] = None) -> Dict[str, str]:
        """
        Get color codes for all images in a directory and its subdirectories
        :param dir_path: Optional - path to directory, however if not provided
                         must have provided during class initialization
        :return: Dictionary of Color Codes mapping file name to color code
        """
        if dir_path is None:
            if self.img_dir is None:
                raise ValueError("No Directory Provided")
            dir_path = self.img_dir

        cc_dict = {}
        for path, subdirs, files in os.walk(dir_path):
            for file in files:
                if file.endswith((".jpg", ".JPG", ".png", ".PNG")):
                    cc_dict[os.path.join(path, file)] = self.get_cc_from_file(os.path.join(path, file))
        return cc_dict

    def save_meta_data(self, path: str = None) -> None:
        if path is None:
            raise FileNotFoundError("No file was specified")
        exif = self.get_cc_from_file(path)
        # TODO
        """
        Search for image id based on the path (paths should be unique)
        
        If found, Update the Meta Data according to the image id
        
        If not found, Create new row and generate new image id (possibly return?)
        """

from sklearn.neighbors import NearestNeighbors
import numpy as np
from PIL import Image

#create a dictionary of pictures
pics = {}

def run_knn():
    return

def add_to_list(loc,f):
    im = Image.open(loc+f)
    #this should never fire, it would mean there is a duplicate picture
    if(f in pics):
        return
    pics[f] = im
    return
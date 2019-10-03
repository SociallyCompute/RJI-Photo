# RJI-Photo
Reynold's Journalism Institute Photo Archive Automated Assessment System

### Purpose:
This project is intended to classify images via quality. Initially we shall try unsupervised clustering algorithms to see if the machines pick up on any data we are missing. This is also being attempted due to our lack of consistent labels. Our datasets are private property distributed to us by the Reynold's Journalism Institute Photo Archive and therefore not avaliable for public use. 

### Required Packages:
Numpy (matrix manipulation)
Sci-Kit Learn (running KMeans)
Matplotlib (graphing)
pillow (image processing)

### How to Run:
This is currently being run in a Python 3.7.1 environment. In order to run the script you execute the run_files.py script after adjusting the root_dir variable located in the main of the file. In addition you need to supply 1 argument during runtime specifying which directory to halt searching.

### KMeans Process:
read in images using pillow and convert to black and white then to numpy array. Store this in a list:

```python
im = im.convert('1') #convert to black and white
pics[f] = im #store in dictionary with filename as key
mat = np.array(im) 
np_pics.append(mat) #add to list of numpy arrays n1 x n2
```
[
    [ [0 0 0 ... 0 0 0]
      .
      .
      .
      [0 0 0 ... 0 0 0] ],
    [ [0 0 0 ... 0 0 0]
      .
      .
      .
      [0 0 0 ... 0 0 0] ]
]

the matrix of each image is variable, n1 x n2. If it is not a black and white image the color has a 3rd dimension.

```python
km = KMeans(n_clusters=15)
km.fit(reduced_pics)
index_set = {i: np.where(km.labels_ == i)[0] for i in range(km.n_clusters)} #index of pictures in data
```

fit them into KMeans clusters (will change depending on how it seems to split) then find the indicies of each of the pictures

### Problems:
- reformatting pictures: at least 1999 has different sizes
  * lossless compression?
  * just run PCA on the pixels to reduce everything to a more managable size?
    - this could be an issue judging quality later on
- color vs black and white: 2D vs 3D arrays
  * going to try running black and white as color with empty arrays
  * combine color dimension and flatten
- transforming images: rotation invariance?
  * not sure how to deal with this, but perhaps just constant feeding in multiple orientations
  * maybe we can insure that inputs are rotated correctly at first
- hyperdimensional display: with this many features, how to represent it visually
  * show grouping based on value as opposed to graphically
- permissions issues with pictures
  * waiting on chmod to be run (read permission denied)
  * over 15000 images in 2017
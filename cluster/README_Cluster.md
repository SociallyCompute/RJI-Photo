## Clustering info

### Required Packages:
Numpy (matrix manipulation)
Sci-Kit Learn (running KMeans)
Matplotlib (graphing)
pillow (image processing)

### How to Run:
This is currently being run in a Python 3.7.1 environment. In order to run the script you execute the run_files.py script. It will prompt more user inputs and following the menu will determine the appropriate start, stop, reduction, and clustering numbers. Root implies the base directory.
The menu is:
```
Which algorithm would you like to train?
1: K-Means
2: KNN
3: Convolusional Neural Network
Your choice: 
```
Then:
```
What dataset would you like to use?
1: root
2: Custom path
Your choice: 
```
Finally
```
What folder do you want to stop on?
Your choice: 
```

### Image Processing:
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

### K-Means:
Flatten the matricies into vectors and stack into a single 2D matrix. Follow that with PCA feature selection and run K-Means on this set.

```python
km = KMeans(n_clusters=n)
km.fit(reduced_pics)
index_set = {i: np.where(km.labels_ == i)[0] for i in range(km.n_clusters)} #index of pictures in data
```
fit them into KMeans clusters (will change depending on how it seems to split) then find the indicies of each of the pictures

### KNN:
Flatten the matricies into vectors and stack into a single 2D matrix. Follow with PCA feature selection and run KNN on this set.

```python
p = np.vstack(reduced_pics)
k = knn(clusters)
k.fit(p, range(p.shape[1])) 
```

### Splitting Images:
Part of running the python script on KNN splits the images into several files based on pixel size. This can be used to prevent loading images repeatedly and wasting processing speed. Written file syntax are full path names followed by \n. Current file names are:
- '22080files.txt' : 4912 x 7360 x 3
- '18048files.txt' : 4016 x 6016 x 3
- '9024files.txt' : 2008 x 3008 x 3

### Convolusional Neural Network:
Primarily using Pytorch for building the CNN. We need to develop stronger labels than the provided file system for usable data.

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


# Login on server as newmatt
```
To run a command as administrator (user "root"), use "sudo <command>".
See "man sudo_root" for details.

newmatt@mudcats:~$ source rji/bin/activate
(rji) newmatt@mudcats:~$ cd github/RJI-Photo/
(rji) newmatt@mudcats:/home/newmatt/github/RJI-Photo$ git status
On branch goggins
Your branch is up to date with 'origin/goggins'.

nothing to commit, working tree clean
(rji) newmatt@mudcats:/home/newmatt/github/RJI-Photo$ 
```

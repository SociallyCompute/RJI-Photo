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
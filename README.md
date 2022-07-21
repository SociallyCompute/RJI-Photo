# RJI-Photo
Reynolds Journalism Institute Photo Archive Automated Assessment System

### Goals:
The Reynolds Journalism Institute has several terrabytes worth of photographs taken over the years. The topics covered
span across all possible topics including sports, public debate, animals, etc. Specifically the request has been to make
it easier for the RJI archivist to sort through images and determine good images to keep and images that should be
discarded. Following up this problem, it was requested a program be written to determine better quality photos from a 
set to show to editors making their life easier.


Therefore, there are 2 key goals to this project:
1. Remove easy to recognize bad quality images (blurry, black screens, etc.)
2. Rank images in similar groupings 

### Process:
To go about this is has been determined that the program be split into 3 separate components:
1. Filter Out the Easy to Recognize "Bad" Images
   1. Apply histogram equalization
   2. Apply a Laplacian filter 
   3. Take the variance of the resulting matrix versus a set threshold
2. Cluster Remaining Images
   1. Dimension Reduction Method:
      1. Resize Image to 720x420
      2. Apply PCA to take the eigenvectors corresponding to 70% of the variance
      3. Cluster the transformed images using DBSCAN
   2. Deep Learning Method:
      1. Apply ResNet without the final layer
      2. Take the resulting feature map and cluster using DBSCAN
3. Rank Images In Clusters:
   1. TBD

### Project Outline
Most of the project runs through python scripts. The required libraries are listed in the 
[requirements.txt](requirements.txt). A setup.py shall be implemented in the future.

#### How to Run
Needs to be optimized!

#### Tests
Needs to be optimized!

#### Exploratory Data Analysis
EDA is located in these two jupyter notebooks [1](EDA.ipynb) and [2](HoG.ipynb). They reveal that there is not much
difference between prelabeled 1s and 7s.

#### Results
Results are stored in both csvs and tensorboard logging
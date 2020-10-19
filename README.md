# RJI-Photo
Reynold's Journalism Institute Photo Archive Automated Assessment System

### Purpose:
This project is intended to classify images via quality. Initially we shall try unsupervised clustering algorithms to see if the machines pick up on any data we are missing. This is also being attempted due to our lack of consistent labels. Our datasets are private property distributed to us by the Reynold's Journalism Institute Photo Archive and therefore not avaliable for public use. 

### Datasets:
* Training - [AVA Dataset](https://www.dpchallenge.com/) 
* Testing - Combination of AVA and a Private Dataset

### Project Breakdown:
This project is divided into multiple parts:
1. Unsupervised Clustering of Images via modern shallow ML methods
2. Deriving Quality Ranking Labels for Images Given Previously Existing Lingustic Variable Rankings in Metadata 
3. Rating Scale of Images through modern Convolutional Neural Network while preserving an Uncertainty through the machine 
Because of the nature and scope of this project more specific documentation and executables exist in each subfolder with an overall executable existing at the root of the repo with run.py. 

### Image Preprocessing:
* Canny Edge Detection - reduce features to compare in DBSCAN

### Unsupervised:
DBSCAN - being optimized so that 

### Deep Learning:
CNN

### Natural Scene Statistics:
NSS

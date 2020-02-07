## Neural Net Info

### Overview
The CNN used in this project is an adapted variant of the state-of-the-art VGG-16 model trained and provided in the [pytorch](https://pytorch.org/docs/stable/_modules/torchvision/models/vgg.html) pre-trained model set. This is a similar idea to what was achieved by Google.AI's Perceptron Team in their [2018 NIMA Paper](https://arxiv.org/pdf/1709.05424.pdf). In that paper, the team used the AVA dataset to classify the aesthetics of images. In their paper they attempted to calculate standard deviation and average score of images where we are trying to rank images against each other with our separate data set. The model is outlined below.

### Final Fully Connected Layer
So we pulled the entire VGG-16 model from pytorch then primarily changed the last layers. For starters we pulled the softmax layer and last 1x1x1000 layer off and applied our own 1x1x10 layer without softmax. This is done because we want to rank on a 1-10 scale and be able to compare them natively with one another. 

### Ranking Scheme
We first need to scale the images to a useable range. This is done through a simple Z-Score calculation. This calculation is done via StandardScaler in [sci-kit learn](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) and makes our values useable and comparable. From there, we dump them into a dataframe and retrieve the classification ranking for each image. 

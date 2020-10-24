### NSS 
#### Pros:
* can be more forceful and explainable
* break apart into luminance and edges
    * Canny vs luminence filter pass
    * Any other passes?
    * Apply a scaling factor based on number of photos in set?
* Requires a lot less training
* Learning the threshold for the "good" vs "bad"
    * instead of CNN learning the kernels
    * Apply SVM
    * Maybe KNNs on a hyperdimensional space labeled by good/bad?
        * Problem here is that we want to choose the best out of a set of similar pictures
        * This could classify all similar images the same instead of choosing the best
    * Apply cross validation to the results
* Once we have the results from each filter pass, we can combine the results to get an overall result
    * Fuzzy rules:
        * If luminance is good and edges are good => good = 10
        * If luminance is ok and edges are good => good = 8
        * If luminance is bad and edges are good => ok = 4
        * If luminance is good and edges are ok => good = 8
        * If luminance is ok and edges are ok => ok = 6 Consistency and balance is better in this case
        * If luminance is bad and edges are ok => bad = 2
        * If luminance is good and edges are bad => ok = 4 
        * If luminance is ok and edges are bad => bad = 2
        * If luminance is bad and edges are bad => bad = 1

#### Cons:
* Fuzzy logic can be a bit sporadic in implementation
* Don't necessarily know all the needed filters
    * i.e. if its more beneficial to apply a smoothing filter, if there are optimal edge filters, etc.
* Lots of possible classifiers to go through ranging from Naive Bayes to SVMs, especially since images are large matrix inputs
    * i.e. hyperdimensionality
* Since the testing set is sampled (albeit in a sort of bad way) we would need a way to convert the fuzzy rules to crisp values
    * The test images use the photoshop tool to assign a color to good/bad images, very few mediocre images

-----------------------------------------------------------------------------------------------------------------------------------------------------------------

### ConvNets
#### Pros:
* deep learning has proven effective for classification problems
    * what we are doing is not quite classification, but adjacent?
* We don't have to worry about fuzzy logic
* Lots of literature for this currently
* We don't have to guess at the filters because they are being updated by the machine for "good" vs "bad"
    * i.e. we don't have to say luminance and edges are filters
* Can use a final fully connected layer with one-hot encoding for classification
* Everything would be self-contained

#### Cons:
* Training time
* Seriously, training time
* Might be overkill - don't need a deep architecture and might actually muddy the results
* Currently struggling with what appears to be vanishing gradients despite architecture
    * built in ReLUs, residuals, and batch normalization - aka this shouldn't be an issue yet it still is
    * currently using pre-trained model from the paper, and training on top of them. might need to change and initialize weights ourselves
* Not explainable - bad XAI for photo journalists

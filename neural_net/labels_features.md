## How do we determine labels
#### Aesthetic Labels
- Could we possibly use fuzzy set theory to determine membership?
    - Prompt users to determine if an image is good/bad/mediocre/etc.
    - Create labels from these values 
- Labels are difficult to determine because there is no standard (human perception of "good", "bad", etc)
- We might be able to match files to edited files? Is there enough edited files for the training data size?
    - ~40 photoshoots/photographer and anywhere from 100-1000 images per shoot in the dump files
    - ~100 edited images

#### Technical Labels
- We can use total pixel count/clarity?
- Pull all starting shot images that aren't intented for human consumption

## Features
#### Colored Images
- Color images have 3 channels (they are a x1 x x2 x 3 matrix)
- Back and White images have 1 channel (they are a x1 x x2 x 1 matrix)
- Would it be better to flip them into grayscale to compare? How will it impact aesthetic rankings?

#### Various Image Sizes
- These need to be standardized so we can successfully compare all images regardless of initial size
    - Try https://pytorch.org/docs/stable/torchvision/transforms.html#torchvision.transforms.Resize
    - Try https://pytorch.org/docs/stable/torchvision/transforms.html#torchvision.transforms.Pad 
        - Pad based on size. Maybe combine padding and resize to perfect square?

#### Memory Issues
- There are significant memory issues with the size when converting from 3D matrix to 2D. Perhaps we can cut the pixel count in half? It might affect the aesthetics but how much time could we save?

#### Possible Feature Reduction
- PCA reduction? 
    - If so, how many features?

#### EXIF Metadata
- Using PIL to pull metadata out of edited images
    - Compare to dump files


## What's Next
- Resize all images
- explore image metadata
- form clusters/create subtitle for clusters
- Possibly apply fuzzy set theory to determine memberships to "good" or "bad" images?

# Plant Growth Performance Rating Tool
This repository contains all the code relevant for my Master's thesis. The goal of the project was to design a pipeline that gives the user feedback on their plant's growth performance, based on a simple smartphone image:

I've compacted this pipeline in a little tkinter application that allows users to perform the estimation and see the results in a windowed format.

## Directories
### Application
This folder contains the scripts and joblib files required to run the biomass estimator. Currently, segmentation is handled by the Segment Anything Model (https://github.com/facebookresearch/segment-anything) and can be done with either text prompts (https://github.com/luca-medeiros/lang-segment-anything) or coordinates through clicking.

### Biomass Estimation
Contains all the scripts needed to extract information from the segmentation masks and fit a linear regression model (used in the application)

### Deeplabv3 Segmentation
Within this folder you find everything related to perform finetuning of the DeepLabV3 Model, with either the ResNet101 or MobileNet_V3_large backbone.
The code in this section is a modified version of this repository:
https://github.com/jnkl314/DeepLabV3FineTuning+

### Misc
This folder contains additional scripts I've used, that are not neccessary to run the applicaton

## Related Dataset
The application has been designed to work with the dataset I've generated during my Master's thesis (though it could be adjusted to work with any basil/plant dataset).
Please read the description of the dataset on kaggle to understand what it is meant for, and what it can't do.
https://kaggle.com/datasets/67d9266f5c2668ef352266e709902c48e5ace70be2c5c160127cae4c3fe50ccf

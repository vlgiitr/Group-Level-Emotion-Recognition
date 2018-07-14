# Group-Level Emotion Recognition

This repository contains the code of our model submitted for the ICMI 2018 EmotiW Group-Level Emotion Recognition Challenge. The model was ranked **4th** in the challenge.

Short Paper of our challenge submission can be found [here](https://arxiv.org/abs/1807.03380).

## Contents

1. [Summary of the Model](#1-summary-of-the-model)
    1. [Corresponding Code for Models explained in Short Paper](#11-corresponding-code-for-models-explained-in-short-paper)
2. [Setup Instructions and Dependencies](#2-setup-instructions-and-dependencies)
3. [Repository Overview](#3-repository-overview)
4. [Dataset Folder Overview](#4-dataset-folder-overview)
5. [Credits](#5-credits)
6. [To-do](#6-to-do)
7. [Guidelines for Contributors](#7-guidelines-for-contributors)
    1. [Reporting Bugs and Opening Issues](#71-reporting-bugs-and-opening-issues)
    2. [Pull Requests](#72-pull-requests)
8. [License](#8-license)
## 1. Summary of the Model

We propose an end-to-end model for jointly learning the scene and facial features of an image for group-level emotion recognition. An overview of the approach is presented in the following figure.

![model_overview](https://raw.githubusercontent.com/aarushgupta/Group-Level-Emotion-Recognition/master/assets/Figure1.jpg?token=AWCUo3tpz-Vu5NLon6smI5Ss4RFTRwJnks5bU4pHwA%3D%3D)

Our model is composed of two branches. The first branch is a **global-level CNN** that detects emotions on the basis of the image as a whole. The second is a **local-level CNN** that detects emotions on the basis of the faces present in the image. The content of each face is merged into a single representation by an **attention mechanism**. This single representation of the facial features is then concatenated with the image feature vector from the Global-Level CNN to build an end-to-end trainable model.

There are four different types of attention mechanisms that we use:
1. Average Features
2. Attention A: Global Image Feature Vector
3. Attention B: Intermediate Feature Vector
4. Attention C: Feature Score

The following figure gives an overview of the different attention mechanisms stated above.

![attention_mechanism_overview](https://raw.githubusercontent.com/aarushgupta/Group-Level-Emotion-Recognition/master/assets/Figure2.jpg?token=AWCUo2CbU5qyhnRdDagzffkXMnN-S-euks5bU4pcwA%3D%3D)

More details of the model and our approach towards the challenge can be found in our [short paper](https://arxiv.org/abs/1807.03380).

### 1.1. Corresponding Code for Models explained in Short Paper



|  S. No. | Model in Paper | Code File in Repository|
| --- | --- | --- |
| 1     | Global_Simple     | DenseNet161_emotiW     |
| 2     | Global_EmotiC     | Densenet_Emotiw_PretrainEmotiC_lr001     |
| 3     | Local     | AlignedModel_EmotiW_lr01_Softmax     |
| 4     | Local_FineTune     | AlignedModelTrainerSoftmax_AlignedModel_EmotiW_lr01_Softmax     |
| 5     | Local_FineTune_LSoftmax     | AlignedModelTrainerLSoftmax_AlignedModel_EmotiW_lr001     |
| 6     | Average     | PretrainedDenseNetAvgFaceFeatures_FineTune_2208_3_NoSoftmax     |
| 7     | Attention_A     | FaceAttention_AlignedModel_FullTrain_lr001_dropout_BN_SoftmaxLr01     |
| 8     | Attention_B     | FaceAttention_AlignedModel_FullTrain_3para_lr001_dropout_BN_SoftmaxLr01     |
| 9     | Attention_B_EmotiC     | FaceAttention_AlignedModel_FullTrain_3para_lr001_dropout_BN_SoftmaxLr01_EmotiC     |
| 10     | Attention_C     | FaceAttention_AlignedModel_FullTrain_4para_lr01_dropout_BN_SoftmaxLr01     |
| 11     | Attention_C_EmotiC     | FaceAttention_AlignedModel_FullTrain_4para_lr001_dropout_BN_SoftmaxLr01_EmotiC     |


For our best performing model, we use an ensemble of the 14 models defined in the repository.


## 2. Setup Instructions and Dependencies

You may setup the repository on your local machine by either downloading it or running the following line on `cmd prompt`.

``` Batchfile
git clone https://github.com/aarushgupta/Group-Level-Emotion-Recognition.git
```

Due to the large sizes of `Dataset` and `TrainedModels`, they have been stored on Google Drive.  You may go to the Google Drive links given in the respective folders to download them. 

The following dependencies are required by the repository:

+ PyTorch v0.4
+ TorchVision v0.2.1
+ NumPy
+ SciPy
+ Scikit-Learn
+ Matplotlib
+ PIL
+ Pickle

## 3. Repository Overview

The repository has the following directories and files:

1. **Dataset**: Contains various datasets used in the model.

2. **Ensemble_Models**: This contains code for the following:
    * saving outputs of the models. 
    * evaluation of ensemble models using the saved outputs. Two kinds of ensembles are present:
        * Weights of models in ensemble determined by handpicking. 
        * Weights of models in ensemble selected by SVM.

3. **MTCNN**: This contains iPython Notebooks for extracting individual face features and images using the MTCNN face detection model.

4. **ModelOutputs**: This contains `.npz` files containing the outputs of all the models. 

5. **Models_FullTrained**: This contains the code for models trained on both the `train` and `VAL` subset of `emotiw` dataset.

6. **Models_TrainDataset**: This contains the code for models trained only on the `train` subset of `emotiw` dataset.

7. **TrainedModels**: This contains pretrained checkpoints of the models used.

8. `AlignedFaces_Extractor_Train.ipynb` and `AlignedFaces_Extractor_Test.ipynb` contains code to apply similarity transform to faces extracted from images using MTCNN model.

9. `Calculator_NumberOfFaces.ipynb` contains code to find the number of faces covering a certain percentage of `emotiw` dataset.

10. `GlobalCNN_DenseNet161_EmotiC_lr001.py` is code for the Global DenseNet-161 model trained on the EmotiC dataset.

## 4. Dataset Folder Overview

The `Dataset` folder contains the following datasets:

1. **AlignedCroppedImages**: This contains `.jpg` image files of aligned faces corresponding to each image in the `emotiw` dataset.
    
    * It is generated from `CroppedFaces` dataset using `AlignedFaces_Extractor_Train.ipynb` and `AlignedFaces_Extractor_Test.ipynb`.

2. **CroppedFaces**: This contains `.npz` files for each image corresponding to the `emotiw` dataset.

    * It is generated from `emotiw` and `FaceCoordinates` dataset using `Face_Cropper_TestDataset.ipynb` and `Face_Cropper_TrainValDataset.ipynb`.
    *  Each `.npz` file contains the following:

        * **a**:  This contains a list of the faces in the image in rgb array form
        * **b**: This contains landmark coordinates for the corresponding faces.

3. **emotic**: This contains the [EmotiC](http://sunai.uoc.edu/emotic/) dataset used for pretraining the models.  

    * Images may be downloaded from [here](http://sunai.uoc.edu/emotic/download.html).
    * `train_annotations.npz` and `val_annotations.npz` contain the following data:
        * **image**: list of image names in training subset or validation subset (corresponding to file).
        * **folder**: list of folder names corresponding to each image in `image` list.
        * **valence**: list of valence scores corresponding to each image in 'image' list. 

4. **emotiw**: This is the EmotiW 2018 Group-Level Emotion Recognition Dataset.

5. **FaceCoordinates**: This contains `.npz` files for each image corresponding to the `emotiw` dataset.

    * It is generated from `emotiw` dataset using `MTCNN/Face_Extractor_BB_Landmarks_Test.ipynb` and `MTCNN/Face_Extractor_BB_Landmarks_Train.ipynb`.  These files extract faces using MTCNN model.
    * Each `.npz` file contains the following:
        * **a**: This contains a list of bounding boxes coordinates of the faces present in an image.
        * **b**: This contains landmark coordinates for the corresponding faces.

6. **FaceFeatures**: This contains `.npz` files for each image corresponding to the `emotiw` dataset.
    * It is generated from `emoti` dataset using `Face_Extractor_Feature_Test.py` and `Face_Extractor_Feature_Train.ipynb`.  These files extract feature vector of faces in an image using MTCNN.  
    * Each `.npz` file contains the following:
        * **a**: This contains a list of 256-dimensional facial features of faces in the corresponding image extracted from the last layer of MTCNN.

7. **Removed_EmotiW**: This contains images removed from the `emotiw` dataset as they were not detected properly by the MTCNN model.

8. **test_list**: This contains a list of images from the `emotiw` dataset to be used as EVAL dataset (as mentioned in paper).

9. **val_list**: This contains a list of images from the `emotiw` dataset to be used as VAL dataset (as mentioned in paper).


## 5. Credits

1. The implementation of the MTCNN model has been adapted from [this](https://github.com/TropComplique/mtcnn-pytorch) repository.
2. The implementation of the SphereFace model (used in aligned models) has been adapted from [this](https://github.com/clcarwin/sphereface_pytorch) repository.
3. We have used the EmotiW 2018 Group-Level Emotion Recognition Challenge dataset (given in `Dataset/emotiw`), cited here:
```cite
@INPROCEEDINGS{7163151, 
author={A. Dhall and J. Joshi and K. Sikka and R. Goecke and N. Sebe}, 
booktitle={2015 11th IEEE International Conference and Workshops on Automatic Face and Gesture Recognition (FG)}, 
title={The more the merrier: Analysing the affect of a group of people in images}, 
year={2015}, 
volume={1}, 
number={}, 
pages={1-8}, 
keywords={emotion recognition;learning (artificial intelligence);social networking (online);automatic affect analysis;emotion labelled database;mood display;multiple kernel learning based hybrid affect inference model;scene context based affect inference model;social media;Computational modeling;Context;Databases;Gold;Kernel;Mood;Videos}, 
doi={10.1109/FG.2015.7163151}, 
ISSN={}, 
month={May},}
```

## 6. To-Do

1. [ ] Upload trained models of `TrainedModels/FullDataset/AlignedModel_EmotiW_lr01_Softmax` and `TrainedModels/FullDataset/PretrainedDenseNetAvgFaceFeatures_FineTune_2208_3_NoSoftmax`.
2. [ ] Review implementations of `Ensemble_Models/Model_OutputSaver_FullTrained` and `Ensemble_Models/Model_OutputSaver_TrainDataset`.

## 7. Guidelines for Contributors

### 7.1. Reporting Bugs and Opening Issues

If you'd like to report a bug or open an issue then please:

**Check if there is an existing issue.** If there is then please add any more information that you have, or give it a 👍.

When submitting an issue please describe the issue as clearly as possible, including how to reproduce the bug. If you can include a screenshot of the issues, that would be helpful.

### 7.2. Pull Requests

Please first discuss the change you wish to make via an issue.

We don't have a set format for Pull Requests, but expect you to list changes, bugs generated and other relevant things in the PR message.

## 8. License

This repository is licensed under MIT license.


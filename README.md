# CV-Project | Power Loss Estimation for Photovoltaic Modules

Computer Vision Project for Solar Pannels

## Introduction

 Solar photovoltaic (PV) modules play a crucial role in renewable energy systems. 
 However, their efficiency can be compromised by environmental factors such as dust, dirt accumulation or even natural lightning.
 This project aims to develop an AI model capable of predicting 2 regression problems, the 'loss_percentage' and 'irradiance_level' on the panels;
 The model should also be able to detect which type of dirt is on the panel, which is part of our classification problem.

This project **aims to achieve some goals like:**

 - Data transformation on the images;
 - Data augmentation techniques to add robustness to the models;
 - 2 regression predictions;
 - 1 classification with 6 different types of soil
 - Visualize predictions and compare results

### Part I

Part I is composed by the file **Preprocessing + EDA**.

On this file, we apply some preprocessing techniques to the dataset like:
1) **Breaking down the string that identifies the images** in attributes/ columns composed by the following attributes in order
to create a pandas dataframe, so we could work with their information;
2) Month, day, hour, minute, seconds, original_title, Irradiance_level and Loss_percentage where the columns extracted from the first string;
3) Mapping the month to numbers => example; map month JAN to 1, FEB to 2 and so on;
4) Datatype conversions from **strings** to **integers** so as not to run into errors; 
5) Timestamp ordering and index resetting 
6) **Visualizations** Visualize the number of pictures per month, day, hour, as well as the label distribution 
7) **Dimensionality Reduction** and **label balancing**
8) Creation of a new folder with roughly **half of the original data**
9) Manually labeling our images - soil (folder Pytorch_data/ID_COYPY)
10) Merging dataframes (both csv's with Regressions and the new information about the soil)


### Part2

 - Pipeline Implementation with Pytorch
 - **Dataset Class** to match the **labels** with the **images**;
 - **Data Augmentation** to add robustness to the trainning data; 
 - **Visualizations** (Visualize a batch of the dataloader to guaratee that everything was correctly assigned)
 - **NN Architecture** (2 CNN's that predict 2 regression targets and 1 classification target with 6 classes)
      1) **Regression:** Loss_Percentage and Irradiance_Level
      2) **Classification:** 'None', 'clay', 'whitesand', 'yellowsand', 'multiple' and 'birddrop';
 - **Training** Our training functions does the following:
      1) Calculates the loss for the train_dataloader both for the regression and the classification and averages both;
      2) Calculates the loss for the val_dataloader following the same procedure;
      3) after training each epoch validates on the val_dataloader and prints mse;
      4) Calculates the accuracy for the classification problem on the evaluation dataframe
 - **Testing** The testing function does the following:
      1) Provides the MSE and MAE for the regressions; 
      2) Accuracy for the classification along with a **classification report** with other metrics for each individual class like f1-score, recall ...
 - Plot some test images with the predictions and their corresponding real values; 

### Part 3 Conclusions

### Conclusions

We were able to come up with a model that can:
 - predict the loss_percentage and irradiance level with extremely good accuracy;
 - An accuracy of almost 100% in detecting the type of soil on the panels;
 - The model generalizes well on unseen data;
 - 


**Difficulties:**

 - Work with over 20.000 thousand images;
 - data types led to some mistakes early on;
 - reduce the data for balancing the dataframe;
 - Develop a framework to drop some images based on some constraints that we defined;
 - Hard to figure out the **"flattened feature dimension"** (128 * 48 * 48 // 4 );
 - Manually labeling the type of soil on the panels;
 - Computational Power
 - Understand how batch normalization is applied

**Future Work:**

 - Design a framework with opencv functions to detect the % which is dirty on the panels;
 - Develop the model even more to detect the % of the pannel occupied by soil;
 - Apply explainability to the images and the decisions behind the predictions;
 - Run the code on more powerful computers
 - Do some fine tuning and try different parameters like the learning rate and regularization;
 - Use pre-trained models and compare their performanece with ours;
 - 


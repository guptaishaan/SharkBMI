# Shark Health Tracker

![Project Image](/sharktraining.png)

## Introduction
Welcome to **Project Shark Health**! This project is designed to provide a novel platform for researchers to efficiently calculate large quantities of shark healths digitally in an accurate manner. 

The health of shark populations is a strong indicator for oceanic ecosystems. This research details the development and application of Convolutional Neural Networks (CNNs) with the YOLOv8 pose detection model to estimate shark health and Body Mass Index (BMI) from visual data. The approach circumvents traditional, invasive monitoring methods, facilitating less disruptive ecological assessments. Our findings demonstrate the model's robust capacity for accurate pose detection, underscoring its potential as a tool for marine biologists in the ongoing effort to preserve marine biodiversity. Accurate prediction of biological metrics is pivotal for understanding and managing diverse ecosystems. This research paper introduces a novel algorithm designed to predict the Body Mass Index (BMI) of sharks based on five key anatomical points. The study utilizes a comprehensive dataset encompassing sharks from different regions and species. Training parameters include anatomical points crucial for determining shark BMI. The algorithm's predictive accuracy is assessed by comparing its predictions with actual BMI values from the dataset. Preliminary findings based on testing data indicate a 93% accuracy rate of the model. The dataset itself is compiled from a variety of sources, including field studies, research institutions, and marine biology databases. This research aims to contribute to the field of marine biology and environmental health by providing a reliable method for predicting shark BMI, a metric integral to understanding their health and overall well-being.

### Features:
1) This product holds a binary classification algorithm for image viability analysis. In the "Binary Class" folder, you can find the model and programs for implementing the code. This algorithm will take a given image (or a folder, when modified) to analyze an image on whether it can be analyzed by researchers for the main metric of health (Body Mass Index). This algorithm is able to identify viable images 95% of the time.
   
2) This repository also holds multiple "Occulation" folders. These are trained on a Convolution Neural Network, which are able to recgonize and point out 5 key points crucial to calculating shark health, along with a bounding box of the shark itself. This recursive calculation minimizes error using 3 different methods of analysis. As seen below, all 3 different methods can be used to calculate shark health:

![Figure 1, connecting points](/sharkmeasure.png)



### Technologies Used:

![Basic visualization of our results](/resultsbasic.png)
![Basic visualization of our results](/resultsdisplay.png)
![Basic visualization of our results](/mainresult.png)
![Basic visualization of our results](/rocboundry.png)

The model’s performance was primarily evaluated using the mean Average Precision (mAP) at a 50% Intersection over Union (IoU) threshold. This metric is a standard in object detection tasks and provides a reliable measure of the model's accuracy in identifying the key anatomical points of sharks.

Throughout the training and validation phases, loss functions were closely monitored. A steady decline in these functions was essential to confirm the model’s effective learning and convergence. A separate subset of the dataset, not used in the training phase, was employed for validation. This approach was critical to assess the model's ability to generalize its learning to new, unseen data.


## Author

![Author Image](/image000000.png)

Hello! I'm Ishaan Gupta, the creator of this project. I am a research intern at the CSU Monterey Bay Marine Biology Lab, and am an avid public speaker who advocates against environmental injustices. Specifcally, I have a love for solving ecological and natural disaster based problems with modern technology. 

### About Me:
- Role: Research Intern @ CSUMB
- Interests: Environmental Engineering (with a focus in Ecology and Disaster Prevention)
- Contact: ishaangupta0408@gmail.com

## How to Navigate the Code

The project structure is as follows:

1) BinaryClass is a folder filled with the fully developed model to identify whether a given shark image is viable for analysis or not. The response here will be binary and can be sorted to a specifc location in the computer (change path name)

2) OcculationValid, OcculationValid2, and PartialValid are all folders with high metrics of accuracy for keypoint pose detection on the shark body. The shark is labelled with all keypoints. An extra layer of verification is added here, where sharks with missing keypoints will not have that given post point annotated. 


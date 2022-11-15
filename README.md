# Using Generative Adversarial Networks to Map Gastrointestinal Diseases during Endoscopy

This repository contains all the code used to implement a GAN capable of mapping gastrointestinal diseases for the KID dataset as part of an MEng final year project.


# Project Context:

Capsule endoscopy is a relatively new, low-cost procedure that can be used to diagnose diseases of the gastrointestinal tract, particularly parts that cannot be easily reached by conventional endoscopes (e.g.distal small bowel). Each year over 12,000 capsule endoscopies are conducted in the UK and the procedure consists of swallowing a video-recording capsule. Currently, a doctor has to manually sift through over 20,000 recorded frames to find often only 1 that contains a concerning pathology. This leads to significant delays in diagnosis and is prone to human error. 

![image](https://user-images.githubusercontent.com/87672746/202011697-6c81b35d-6617-49f6-9403-04d9d4612f1a.png)

Recently, significant advances have been made in the field of AI image processing due to increased computational power and availability of large datasets. This has had a particularly large impact in healthcare, where AI has been applied for patient treatment, hospital management, disease detection and surgery. This project aims to detect diseases in a labelled dataset of images taking during capsule endoscopy using AI, the KID dataset. Specifically, the student will work with neural networks capable of processing large image datasets, with increasingly complex architectures to reach the final goal. The relevant networks are: 

•	Simple classifier of the diseases present in the dataset provided

•	Variational Autoencoder

•	Pix2Pix GAN- Generative Adversarial Network


# Prerequisites:
1- Python
2- Tensorflow 2.0
3- Fundamental programming knowledge 

# Project's Structure (Objectives):
1.	Implement and test an Image Classifier using convolutional neural networks on an example dataset, i.e. Fashion MNIST dataset
2.	Implement and test an Image Classifier using convolutional neural networks on the labelled KID dataset to classify the different diseases present in it
3.	Implement and test a Variational Auto Encoder (VAE) using an example dataset, i.e. MNIST dataset. (Note: VAE is one of the neural networks that constitute the GAN to be implemented)
4.	Implement and test the Pix2Pix GAN using an example dataset, i.e. CelebA dataset 
5.	Implement and test the Pix2Pix GAN on the labelled KID dataset to map gastrointestinal diseases 
6.	Additional: Combine classification with disease mapping to create colour-coded maps that show different diseases
7.	Additional: Run the final trained model on a different, unlabelled data set on the Pix2Pix GAN to compare performance with the KID data set and discuss results

Details of each Objective can be found in the respective folder

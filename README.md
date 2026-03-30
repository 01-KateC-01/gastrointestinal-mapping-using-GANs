# Using Generative Adversarial Networks to Map Gastrointestinal Diseases during Endoscopy
This repository contains all the code used to implement AI models capable of mapping gastrointestinal diseases for the KID dataset as part of an MEng final year project (awarded 92%, EEE Project Student of the Year 2023).


# Project Context:
Capsule endoscopy is a relatively new, low-cost procedure that can be used to diagnose diseases of the gastrointestinal tract, particularly parts that cannot be easily reached by conventional endoscopes (e.g.distal small bowel). Each year over 12,000 capsule endoscopies are conducted in the UK and the procedure consists of swallowing a video-recording capsule. Currently, a doctor has to manually sift through over 20,000 recorded frames to find often only 1 that contains a concerning pathology. This leads to significant delays in diagnosis and is prone to human error.


Recently, significant advances have been made in the field of AI image processing due to increased computational power and availability of large datasets. This project applies increasingly complex neural network architectures to automate disease detection and mapping from capsule endoscopy images.

![image](https://user-images.githubusercontent.com/87672746/202011697-6c81b35d-6617-49f6-9403-04d9d4612f1a.png)
# Technologies & Frameworks
- Python (TensorFlow 2.0, NumPy, pandas, scikit-image, matplotlib)
- Architectures: U-Net, CNNs (VGG16, ResNet50, EfficientNet B4, Inception V3), Autoencoders
- Dataset: KID Dataset 2 (2,371 capsule endoscopy images, 8 classes)
- Platform: Google Colab (cloud-based GPU training)
- Validation: Stratified K-fold cross-validation, confusion matrices, TPR/TNR metrics

## Results
- Benchmarked 4 pre-trained CNN classifiers (VGG16, ResNet50, EfficientNet B4, 
  Inception V3) on the KID dataset across 8 gastrointestinal disease classes, 
  achieving up to 70% validation accuracy on an inherently imbalanced medical dataset
- Final segmentation model: U-Net architecture achieving 80%+ segmentation accuracy 
  on multi-pathology gastrointestinal images — the first proof-of-concept for 
  multi-pathology segmentation on this dataset
- Awarded EEE Project Student of the Year 2023 (92%)

# Project's Structure (Objectives):
1.	Implement and test an Image Classifier using convolutional neural networks on an example dataset, i.e. Fashion MNIST dataset
2.	Implement and test an Image Classifier using convolutional neural networks on the labelled KID dataset to classify the different diseases present in it
3.	Implement and test a Variational Auto Encoder (VAE) and U-Net architecture on the KID dataset to map gastrointestinal diseases

Details of each Objective can be found in the respective folder

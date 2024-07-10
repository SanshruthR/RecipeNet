# RecipeNet

[![Python](https://img.shields.io/badge/python-3.8%2B-brightgreen.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9.0-purple.svg)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Spaces-yellow)](https://huggingface.co/spaces/Sanshruth/RecipeNet)
[![Recipe1M](https://img.shields.io/badge/Recipe1M-Dataset-blue)](http://im2recipe.csail.mit.edu/im2recipe.pdf)
[![Ingredients 101](https://img.shields.io/badge/Recipes5k-Dataset-red)](https://arxiv.org/pdf/1707.08816v1)
[![License: MIT](https://img.shields.io/badge/License-MIT-pink.svg)](https://opensource.org/licenses/MIT)


<p align="left">
  <img src="https://github.com/SanshruthR/RecipeNet/assets/98751980/c87d7ea2-c7d7-46eb-97d6-fe51f84303de" alt="pic1" style="width:23%;"/>
  <img src="https://github.com/SanshruthR/RecipeNet/assets/98751980/c4529796-3c9b-4a48-b71f-16a75ac65d5f" alt="pic4" style="width:23%;"/>
  
  <img src="https://github.com/SanshruthR/RecipeNet/assets/98751980/8a234af6-1614-41cd-a002-cd36504918cc" alt="pic2" style="width:23%;"/>
  <img src="https://github.com/SanshruthR/RecipeNet/assets/98751980/2fe6a1b2-3088-4728-9b4e-5d60ebca3c48" alt="pic3" style="width:23%;"/>
</p>








RecipeNet is a deep learning model designed to classify food images and provide recipes, ingredients, and preparation time based on the Recipe1M dataset. The project utilizes transfer learning techniques with EfficientNet-B0 as the base model.

## Table of Contents

- [Introduction](#introduction)
- [Datasets](#datasets)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Results](#results)
- [Usage](#usage)
- [References](#references)

## Introduction

RecipeNet leverages the Recipe1M, Recipe5K, and Ingredients 101 datasets to create a robust food classification and recipe retrieval system. By using EfficientNet-B0 for transfer learning, the model achieves high accuracy in classifying food images and retrieving relevant recipes and ingredients.

## Datasets

### Recipe1M
Recipe1M is a large-scale dataset containing over one million recipes and their corresponding images. The dataset is designed to facilitate the development of AI systems capable of understanding and generating cooking recipes. More details can be found in the [Recipe1M paper](http://im2recipe.csail.mit.edu/im2recipe.pdf).

### Recipe5K
Recipe5K is another dataset used for food classification tasks, containing 5,000 images and their corresponding recipes. This dataset helps improve the generalization capabilities of the model. More details are available [here](https://arxiv.org/pdf/1707.08816v1).

### Ingredients 101
Ingredients 101 is a dataset focused on recognizing food ingredients through multi-label learning. This dataset enhances the model's ability to identify specific ingredients in food images. More details can be found in the [Ingredients 101 paper](https://paperswithcode.com/paper/food-ingredients-recognition-through-multi/review/).

## Model Architecture

RecipeNet is built using EfficientNet-B0, a state-of-the-art convolutional neural network. The model architecture includes:

- EfficientNet-B0 backbone pre-trained on ImageNet.
- Custom classifier with a dropout layer and a fully connected layer.
- Custom heads for ingredient prediction.

### Model Layers

- **Input Layer**: Image input of size (224, 224, 3).
- **EfficientNet-B0 Backbone**: Feature extraction layers.
- **Dropout Layer**: Dropout probability of 0.2.
- **Fully Connected Layer**: 1280 input features and 101 output features for food categories.
- **Heads**: Linear layer with 768 input features and 101 output features for ingredient classification.

### Parameters

- **Optimizer**: Adam
- **Learning Rate**: 1e-4
- **Loss Function**: Cross-Entropy Loss
- **Epochs**: 50
- **Batch Size**: 32

## Training

The model was trained on a combination of the Recipe1M, Recipe5K, and Ingredients 101 datasets. The training process included data augmentation, learning rate scheduling, and early stopping to prevent overfitting.

## Results

RecipeNet achieved high accuracy on the test set, demonstrating its effectiveness in food classification and recipe retrieval tasks. The model can predict the food category, preparation time, ingredients, and provide a link to the recipe.

## Usage

To use RecipeNet, visit the following links:
- [RecipeNet Web Application](https://recipenetai.netlify.app/)
- [RecipeNet on Hugging Face Spaces](https://huggingface.co/spaces/Sanshruth/RecipeNet)

## References

- [Recipe1M: A Dataset for Learning Cross-Modal Embeddings for Cooking Recipes and Food Images](http://im2recipe.csail.mit.edu/im2recipe.pdf)
- [Recipes5K Dataset](https://arxiv.org/pdf/1707.08816v1)
- [Ingredients 101: Food Ingredients Recognition Through Multi-label Learning](https://paperswithcode.com/paper/food-ingredients-recognition-through-multi/review/)


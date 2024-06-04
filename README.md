# Video Summarization using Deep Learning

## Overview

This repository contains the code for my graduation thesis project titled "Research and Application of Deep Learning in Video Summarization". The project focuses on the SUM-GAN-AED model to summarize videos into shorter versions. The model employs Convolutional Neural Networks (CNN) for feature extraction, Self-Attention mechanisms for weighting each frame, and a combination of Variational Autoencoders (VAE) and Generative Adversarial Networks (GAN) to enhance the performance of Self-Attention and enable unsupervised learning.

## Introduction

This project aims to develop a deep learning model capable of summarizing videos effectively. The SUM-GAN-AED model integrates several advanced techniques to achieve this goal:

- **CNN (Convolutional Neural Networks)**: Used for feature extraction from video frames.
- **Self-Attention**: Helps in assigning weights to individual frames, highlighting the important parts.
- **VAE (Variational Autoencoders) + GAN (Generative Adversarial Networks)**: Enhances the performance of the Self-Attention mechanism and supports unsupervised learning.

## Model Architecture

The SUM-GAN-AED model is structured as follows:

1. **Feature Extraction**: CNNs are used to extract features from each frame of the video.
2. **Self-Attention**: This mechanism assigns weights to each frame based on its importance, determined by the extracted features.
3. **VAE and GAN**: The VAE generates a latent representation of the frames, while the GAN refines this representation to improve the quality of the summarization.


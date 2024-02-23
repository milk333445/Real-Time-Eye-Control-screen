# Project Name: Real-Time Eye-Tracking Control Screen pipeline

## Introduction

-   This project is an innovative real-time eye-tracking system designed to control the computer cursor by analyzing the user's eye movements. It can be applied in a wide range of scenarios, such as assisting individuals with mobility impairments to interact more easily with computers, or offering a more natural mode of interaction in gaming and virtual reality environments.

## Technical Overview

1. Computer Vision: Utilizes the OpenCV library for real-time video stream processing and facial feature detection. Captures the user's facial image through a camera and extracts the eye region in real-time.

2. Deep Learning: Applies Convolutional Neural Networks (CNNs), particularly the MobileNet model variants, to analyze eye images and predict the direction of gaze. The models are pretrained on extensive datasets to recognize different eye movement patterns.

## File Explanation

1. ./detectors:Includes detectors for various pose estimates
2. eyetracking.py:Realization of real-time eye control system
3. ./models:Deep Learning Models
4. righteye_img_capture.py:Mainly used for capturing eye training data.
5. train.py:Training 2D mapping model by collecting eyeball data

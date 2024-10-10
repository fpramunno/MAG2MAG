# Magnetogram-to-Magnetogram: Generative Forecasting of Solar Evolution

## Abstract
Investigating the solar magnetic field is crucial to understand the physical processes in the solar interior as well as their effects on the interplanetary environment. We introduce a cutting-edge method to predict the evolution of the solar line-of-sight (LoS) magnetogram using image-to-image translation with Denoising Diffusion Probabilistic Models (DDPMs). Our approach combines "computer science metrics" for image quality and "physics metrics" for physical accuracy to evaluate model performance. The results indicate that DDPMs are highly effective in maintaining the structural integrity, the dynamic range of solar magnetic fields, the magnetic flux and other physical features such as the size of the active regions, surpassing traditional persistence models, also in flaring situation, which could . We aim to use deep learning not only for visualisation but as an integrative tool for telescopes, enhancing our understanding of unknown physical events like solar flares. Future studies will aim to integrate more diverse solar data to refine the accuracy and applicability of our generative model.

## Introduction
This repository contains the code for training the Mag2Mag paper. The paper introduces a cutting-edge method to predict the evolution of the solar line-of-sight magnetogram using Denoising Diffusion Probabilistic Models (DDPMs), assessing its performance with metrics that reflect both image quality and physical accuracy, ultimately aiming to enhance our understanding and visualization of solar phenomena.
- `training.py`: The main script for training the model.
- `diffusion.py`: Contains the DDPM class and utility functions.
- `modules.py`: Includes essential modules and classes used in the model.
- `physics_param.py`: Script to compute the physics metrics.
- `util.py`: Includes all the utility functions for the physics metrics computations.

## Examples
![](https://github.com/fpramunno/MAG2MAG/blob/main/pred.png)

### Prerequisites
List of libraries to run the project:

### Contact
Contact me at francesco.ramunno@fhnw.ch

### References
Paper presented at the first ESA AI (SPAICE) Conference 17 19 September 2024: https://arxiv.org/abs/2407.11659


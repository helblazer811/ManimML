# Manim Machine Learning

<img src="examples/ManimMLLogo.gif">

Manim Machine Learning is a project focused on providing animations and visualizations of common machine learning concepts with the [Manim Community Library](https://www.manim.community/). We want this project to be a compilation of primitive visualizations that can be easily combined to create videos about complex machine learning concepts. Additionally, we want to provide a set of abstractions which allow users to focus on explanations instead of software engineering.


## Table of Contents

## Getting Started 
First you will want to [install manim](https://docs.manim.community/en/stable/installation.html). Then you can run the following to generate the example videos. 

`make video`

or 

`manim -pqh src/vae.py VAEScene`

## Examples

Checkout the ```examples``` directory for some example videos with source code. 

### Variational Autoencoders

This is a visualization of a Variational Autoencoder. 

<img src="examples/VAEScene.gif" width="600">

### VAE Disentanglement 

This is a visualization of disentanglement with a Variational Autoencoder

<img src="examples/DisentanglementScene.gif" width="600">

### Neural Networks

This is a visualization of a Neural Network. 

<img src="examples/TestNeuralNetworkScene.gif" width="600">

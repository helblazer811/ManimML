# ManimML
<a href="https://github.com/helblazer811/ManimMachineLearning">
    <img src="examples/media/ManimMLLogo.gif">
</a>

[![GitHub license](https://img.shields.io/github/license/helblazer811/ManimMachineLearning)](https://github.com/helblazer811/ManimMachineLearning/blob/main/LICENSE.md)
[![GitHub tag](https://img.shields.io/github/v/release/helblazer811/ManimMachineLearning)](https://img.shields.io/github/v/release/helblazer811/ManimMachineLearning)
![Pypi Downloads](https://img.shields.io/pypi/dm/manim-ml)
[![Follow Twitter](https://img.shields.io/twitter/follow/alec_helbling?style=social)](https://twitter.com/alec_helbling)

ManimML is a project focused on providing animations and visualizations of common machine learning concepts with the [Manim Community Library](https://www.manim.community/). We want this project to be a compilation of primitive visualizations that can be easily combined to create videos about complex machine learning concepts. Additionally, we want to provide a set of abstractions which allow users to focus on explanations instead of software engineering.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Examples](#examples)

## Getting Started 
First you will want to [install manim](https://docs.manim.community/en/stable/installation.html). 

Then install the package form source or
`pip install manim_ml`

Then you can run the following to generate the example videos from python scripts. 

`manim -pqh src/vae.py VAEScene`

## Examples

Checkout the ```examples``` directory for some example videos with source code. 

### Neural Networks

This is a visualization of a Variational Autoencoder made using ManimML. It has a Pytorch style list of layers that can be composed in arbitrary order. The following video is made with the code from below.  

<img src="examples/media/VAEScene.gif">

```python
class VariationalAutoencoderScene(Scene):

    def construct(self):
        embedding_layer = EmbeddingLayer(dist_theme="ellipse").scale(2)
        
        image = Image.open('images/image.jpeg')
        numpy_image = np.asarray(image)
        # Make nn
        neural_network = NeuralNetwork([
            ImageLayer(numpy_image, height=1.4),
            FeedForwardLayer(5),
            FeedForwardLayer(3),
            embedding_layer,
            FeedForwardLayer(3),
            FeedForwardLayer(5),
            ImageLayer(numpy_image, height=1.4),
        ], layer_spacing=0.1)

        neural_network.scale(1.3)

        self.play(Create(neural_network))
        self.play(neural_network.make_forward_pass_animation(run_time=15))
```

### Generative Adversarial Network

This is a visualization of a Generative Adversarial Network made using ManimML. 

<img src="examples/media/GANScene.gif">

### VAE Disentanglement 

This is a visualization of disentanglement with a Variational Autoencoder

<img src="examples/media/DisentanglementScene.gif">


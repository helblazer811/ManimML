Design Doc: Neural Network Rendering Pipeline
=============================================

=======
Neural Network Layer Scope and Mobject Ownership
=======

An important consideration when designing this pipeline 
was to figure out a consistent design philosophy answering the 
following questions:

1. What content is owned by an individual layer?
2. Does a layer describe an operation, a piece of data, or both?


=======
Constructing More Elaborate Animations
=======

Sometimes we want to construct custom animations that span more 
than a single central layer and its two surrounding layers. 

One possible solution to this is to allow for the passing 
of neural networks into larger neural networks and treating them as layers.
This way a person can make a class that extends NeuralNetwork, which then
can implement its own forward pass animation, and then that neural network
can then be passed into a larger neural network and treated as its own 
individual layer. 

This may be a good workaround in certain situations, but this does not 
solve the problem of allowing for multiple animations that each require
the context of multiple layers. 
"""
    Code for making a dropout animation for the
    feed forward layers of a neural network. 
"""
from manim import *
import random

from manim_ml.neural_network.layers.feed_forward import FeedForwardLayer
from manim_ml.neural_network.layers.feed_forward_to_feed_forward import (
    FeedForwardToFeedForward,
)


class XMark(VGroup):
    def __init__(self, stroke_width=1.0, color=GRAY):
        super().__init__()
        line_one = Line(
            [-0.1, 0.1, 0],
            [0.1, -0.1, 0],
            stroke_width=1.0,
            stroke_color=color,
            z_index=4,
        )
        self.add(line_one)
        line_two = Line(
            [-0.1, -0.1, 0],
            [0.1, 0.1, 0],
            stroke_width=1.0,
            stroke_color=color,
            z_index=4,
        )
        self.add(line_two)


def get_edges_to_drop_out(layer: FeedForwardToFeedForward, layers_to_nodes_to_drop_out):
    """Returns edges to drop out for a given FeedForwardToFeedForward layer"""
    prev_layer = layer.input_layer
    next_layer = layer.output_layer
    # Get the nodes to dropout in previous layer
    prev_layer_nodes_to_dropout = layers_to_nodes_to_drop_out[prev_layer]
    next_layer_nodes_to_dropout = layers_to_nodes_to_drop_out[next_layer]
    # Compute the edges to dropout
    edges_to_dropout = []
    edge_indices_to_dropout = []
    for edge_index, edge in enumerate(layer.edges):
        prev_node_index = int(edge_index / next_layer.num_nodes)
        next_node_index = edge_index % next_layer.num_nodes
        # Check if the edges should be dropped out
        if (
            prev_node_index in prev_layer_nodes_to_dropout
            or next_node_index in next_layer_nodes_to_dropout
        ):
            edges_to_dropout.append(edge)
            edge_indices_to_dropout.append(edge_index)

    return edges_to_dropout, edge_indices_to_dropout


def make_pre_dropout_animation(
    neural_network,
    layers_to_nodes_to_drop_out,
    dropped_out_color=GRAY,
    dropped_out_opacity=0.2,
):
    """Makes an animation that sets up the NN layer for dropout"""
    animations = []
    # Go through the network and get the FeedForwardLayer instances
    feed_forward_layers = neural_network.filter_layers(
        lambda layer: isinstance(layer, FeedForwardLayer)
    )
    # Go through the network and get the FeedForwardToFeedForwardLayer instances
    feed_forward_to_feed_forward_layers = neural_network.filter_layers(
        lambda layer: isinstance(layer, FeedForwardToFeedForward)
    )
    # Get the edges to drop out
    layers_to_edges_to_dropout = {}
    for layer in feed_forward_to_feed_forward_layers:
        layers_to_edges_to_dropout[layer], _ = get_edges_to_drop_out(
            layer, layers_to_nodes_to_drop_out
        )
    # Dim the colors of the edges
    dim_edge_colors_animations = []
    for layer in layers_to_edges_to_dropout.keys():
        edges_to_drop_out = layers_to_edges_to_dropout[layer]
        # Make color dimming animation
        for edge_index, edge in enumerate(edges_to_drop_out):
            """
            def modify_edge(edge):
                edge.set_stroke_color(dropped_out_color)
                edge.set_stroke_width(0.6)
                edge.set_stroke_opacity(dropped_out_opacity)
                return edge

            dim_edge = ApplyFunction(
                modify_edge,
                edge
            )
            """

            dim_edge_colors_animations.append(FadeOut(edge))
    dim_edge_colors_animation = AnimationGroup(
        *dim_edge_colors_animations, lag_ratio=0.0
    )
    # Dim the colors of the nodes
    dim_nodes_animations = []
    x_marks = []
    for layer in layers_to_nodes_to_drop_out.keys():
        nodes_to_dropout = layers_to_nodes_to_drop_out[layer]
        # Make an X over each node
        for node_index, node in enumerate(layer.node_group):
            if node_index in nodes_to_dropout:
                x_mark = XMark()
                x_marks.append(x_mark)
                x_mark.move_to(node.get_center())
                create_x = Create(x_mark)
                dim_nodes_animations.append(create_x)

    dim_nodes_animation = AnimationGroup(*dim_nodes_animations, lag_ratio=0.0)

    animation_group = AnimationGroup(
        dim_edge_colors_animation,
        dim_nodes_animation,
    )

    return animation_group, x_marks


def make_post_dropout_animation(
    neural_network,
    layers_to_nodes_to_drop_out,
    x_marks,
):
    """Returns the NN to normal after dropout"""
    # Go through the network and get the FeedForwardLayer instances
    feed_forward_layers = neural_network.filter_layers(
        lambda layer: isinstance(layer, FeedForwardLayer)
    )
    # Go through the network and get the FeedForwardToFeedForwardLayer instances
    feed_forward_to_feed_forward_layers = neural_network.filter_layers(
        lambda layer: isinstance(layer, FeedForwardToFeedForward)
    )
    # Get the edges to drop out
    layers_to_edges_to_dropout = {}
    for layer in feed_forward_to_feed_forward_layers:
        layers_to_edges_to_dropout[layer], _ = get_edges_to_drop_out(
            layer, layers_to_nodes_to_drop_out
        )
    # Remove the x marks
    uncreate_animations = []
    for x_mark in x_marks:
        uncreate_x_mark = Uncreate(x_mark)
        uncreate_animations.append(uncreate_x_mark)

    uncreate_x_marks = AnimationGroup(*uncreate_animations, lag_ratio=0.0)
    # Add the edges back
    create_edge_animations = []
    for layer in layers_to_edges_to_dropout.keys():
        edges_to_drop_out = layers_to_edges_to_dropout[layer]
        # Make color dimming animation
        for edge_index, edge in enumerate(edges_to_drop_out):
            edge_copy = edge.copy()
            edges_to_drop_out[edge_index] = edge_copy
            create_edge_animations.append(FadeIn(edge_copy))

    create_edge_animation = AnimationGroup(*create_edge_animations, lag_ratio=0.0)

    return AnimationGroup(uncreate_x_marks, create_edge_animation, lag_ratio=0.0)


def make_forward_pass_with_dropout_animation(
    neural_network,
    layers_to_nodes_to_drop_out,
):
    """Makes forward pass animation with dropout"""
    layer_args = {}
    # Go through the network and get the FeedForwardLayer instances
    feed_forward_layers = neural_network.filter_layers(
        lambda layer: isinstance(layer, FeedForwardLayer)
    )
    # Go through the network and get the FeedForwardToFeedForwardLayer instances
    feed_forward_to_feed_forward_layers = neural_network.filter_layers(
        lambda layer: isinstance(layer, FeedForwardToFeedForward)
    )
    # Iterate through network and get feed forward layers
    for layer in feed_forward_layers:
        layer_args[layer] = {"dropout_node_indices": layers_to_nodes_to_drop_out[layer]}
    for layer in feed_forward_to_feed_forward_layers:
        _, edge_indices = get_edges_to_drop_out(layer, layers_to_nodes_to_drop_out)
        layer_args[layer] = {"edge_indices_to_dropout": edge_indices}

    return neural_network.make_forward_pass_animation(layer_args=layer_args)


def make_neural_network_dropout_animation(
    neural_network, dropout_rate=0.5, do_forward_pass=True
):
    """
    Makes a dropout animation for a given neural network.

    NOTE Does dropout only on feed forward layers.

    1. Does dropout
    2. If `do_forward_pass` then do forward pass animation
    3. Revert network to pre-dropout appearance
    """
    # Go through the network and get the FeedForwardLayer instances
    feed_forward_layers = neural_network.filter_layers(
        lambda layer: isinstance(layer, FeedForwardLayer)
    )
    # Go through the network and get the FeedForwardToFeedForwardLayer instances
    feed_forward_to_feed_forward_layers = neural_network.filter_layers(
        lambda layer: isinstance(layer, FeedForwardToFeedForward)
    )
    # Get random nodes to drop out for each FeedForward Layer
    layers_to_nodes_to_drop_out = {}
    for feed_forward_layer in feed_forward_layers:
        num_nodes = feed_forward_layer.num_nodes
        nodes_to_drop_out = []
        # Compute random probability that each node is dropped out
        for node_index in range(num_nodes):
            dropout_prob = random.random()
            if dropout_prob < dropout_rate:
                nodes_to_drop_out.append(node_index)
        # Add the mapping to the dict
        layers_to_nodes_to_drop_out[feed_forward_layer] = nodes_to_drop_out
    # Make the animation
    pre_dropout_animation, x_marks = make_pre_dropout_animation(
        neural_network, layers_to_nodes_to_drop_out
    )
    if do_forward_pass:
        forward_pass_animation = make_forward_pass_with_dropout_animation(
            neural_network, layers_to_nodes_to_drop_out
        )
    else:
        forward_pass_animation = AnimationGroup()

    post_dropout_animation = make_post_dropout_animation(
        neural_network, layers_to_nodes_to_drop_out, x_marks
    )
    # Combine the animations into one
    return Succession(
        pre_dropout_animation, forward_pass_animation, post_dropout_animation
    )

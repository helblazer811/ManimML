"""
    Module for visualizing decision trees in Manim. 
    It parses a decision tree classifier from sklearn. 

    TODO return a map from nodes to split animation for BFS tree expansion
    TODO reimplement the decision 2D decision tree surface drawing. 
"""
from manim import *
from manim_ml.decision_tree.decision_tree_surface import (
    compute_decision_areas,
    merge_overlapping_polygons,
)
import manim_ml.decision_tree.helpers as helpers

import numpy as np
from PIL import Image

class LeafNode(Group):
    """Leaf node in tree"""

    def __init__(
        self, class_index, display_type="image", class_image_paths=[], class_colors=[]
    ):
        super().__init__()
        self.display_type = display_type
        self.class_image_paths = class_image_paths
        self.class_colors = class_colors
        assert self.display_type in ["image", "text"]
        if self.display_type == "image":
            self._construct_image_node(class_index)
        else:
            raise NotImplementedError()

    def _construct_image_node(self, class_index):
        """Make an image node"""
        # Get image
        image_path = self.class_image_paths[class_index]
        pil_image = Image.open(image_path)
        node = ImageMobject(pil_image)
        node.scale(1.5)
        rectangle = Rectangle(
            width=node.width + 0.05,
            height=node.height + 0.05,
            color=self.class_colors[class_index],
            stroke_width=6,
        )
        rectangle.move_to(node.get_center())
        rectangle.shift([-0.02, 0.02, 0])
        self.add(rectangle)
        self.add(node)

class SplitNode(VGroup):
    """Node for splitting decision in tree"""

    def __init__(self, feature, threshold):
        super().__init__()
        node_text = f"{feature}\n<=  {threshold:.2f} cm"
        # Draw decision text
        decision_text = Text(node_text, color=WHITE)
        # Draw the surrounding box
        bounding_box = SurroundingRectangle(decision_text, buff=0.3, color=WHITE)
        self.add(bounding_box)
        self.add(decision_text)

class DecisionTreeDiagram(Group):
    """Decision Tree Diagram Class for Manim"""

    def __init__(
        self,
        sklearn_tree,
        feature_names=None,
        class_names=None,
        class_images_paths=None,
        class_colors=[RED, GREEN, BLUE],
    ):
        super().__init__()
        self.tree = sklearn_tree
        self.feature_names = feature_names
        self.class_names = class_names
        self.class_image_paths = class_images_paths
        self.class_colors = class_colors
        # Make graph container for the tree
        self.tree_group, self.nodes_map, self.edge_map = self._make_tree()
        self.add(self.tree_group)

    def _make_node(
        self,
        node_index,
    ):
        """Make node"""
        is_split_node = (
            self.tree.children_left[node_index] != self.tree.children_right[node_index]
        )
        if is_split_node:
            node_feature = self.tree.feature[node_index]
            node_threshold = self.tree.threshold[node_index]
            node = SplitNode(self.feature_names[node_feature], node_threshold)
        else:
            # Get the most abundant class for the given leaf node
            # Make the leaf node object
            tree_class_index = np.argmax(self.tree.value[node_index])
            node = LeafNode(
                class_index=tree_class_index,
                class_colors=self.class_colors,
                class_image_paths=self.class_image_paths,
            )
        return node

    def _make_connection(self, top, bottom, is_leaf=False):
        """Make a connection from top to bottom"""
        top_node_bottom_location = top.get_center()
        top_node_bottom_location[1] -= top.height / 2
        bottom_node_top_location = bottom.get_center()
        bottom_node_top_location[1] += bottom.height / 2

        line = Line(top_node_bottom_location, bottom_node_top_location, color=WHITE)

        return line

    def _make_tree(self):
        """Construct the tree diagram"""
        tree_group = Group()
        max_depth = self.tree.max_depth
        # Make the root node
        nodes_map = {}
        root_node = self._make_node(
            node_index=0,
        )
        nodes_map[0] = root_node
        tree_group.add(root_node)
        # Save some information
        node_height = root_node.height
        node_width = root_node.width
        scale_factor = 1.0
        edge_map = {}
        # tree height
        tree_height = scale_factor * node_height * max_depth
        tree_width = scale_factor * 2**max_depth * node_width
        # traverse tree
        def recurse(node_index, depth, direction, parent_object, parent_node):
            # make the node object
            is_leaf = (
                self.tree.children_left[node_index]
                == self.tree.children_right[node_index]
            )
            node_object = self._make_node(node_index=node_index)
            nodes_map[node_index] = node_object
            node_height = node_object.height
            # set the node position
            direction_factor = -1 if direction == "left" else 1
            shift_right_amount = (
                0.9 * direction_factor * scale_factor * tree_width / (2**depth) / 2
            )
            if is_leaf:
                shift_down_amount = -1.0 * scale_factor * node_height
            else:
                shift_down_amount = -1.8 * scale_factor * node_height
            node_object.match_x(parent_object).match_y(parent_object).shift(
                [shift_right_amount, shift_down_amount, 0]
            )
            tree_group.add(node_object)
            # make a connection
            connection = self._make_connection(
                parent_object, node_object, is_leaf=is_leaf
            )
            edge_name = str(parent_node) + "," + str(node_index)
            edge_map[edge_name] = connection
            tree_group.add(connection)
            # recurse
            if not is_leaf:
                recurse(
                    self.tree.children_left[node_index],
                    depth + 1,
                    "left",
                    node_object,
                    node_index,
                )
                recurse(
                    self.tree.children_right[node_index],
                    depth + 1,
                    "right",
                    node_object,
                    node_index,
                )

        recurse(self.tree.children_left[0], 1, "left", root_node, 0)
        recurse(self.tree.children_right[0], 1, "right", root_node, 0)

        tree_group.scale(0.35)
        return tree_group, nodes_map, edge_map

    def create_level_order_expansion_decision_tree(self, tree):
        """Expands the decision tree in level order"""
        raise NotImplementedError()
    
    def create_bfs_expansion_decision_tree(self, tree):
        """Expands the tree using BFS"""
        animations = []
        split_node_animations = {} # Dictionary mapping split node to animation
        # Compute parent mapping
        parent_mapping = helpers.compute_node_to_parent_mapping(self.tree)
        # Create the root node as most common class
        placeholder_class_nodes = {}
        root_node_class_index = np.argmax(
            self.tree.value[0]
        )
        root_placeholder_node = LeafNode(
            class_index=root_node_class_index,
            class_colors=self.class_colors,
            class_image_paths=self.class_image_paths,
        )
        root_placeholder_node.move_to(self.nodes_map[0])
        placeholder_class_nodes[0] = root_placeholder_node
        root_create_animation = AnimationGroup(
            FadeIn(root_placeholder_node),
            lag_ratio=0.0
        )
        animations.append(root_create_animation)
        # Iterate through the nodes
        queue = [0]
        while len(queue) > 0:
            node_index = queue.pop(0)
            # Check if a node is a split node or not
            left_child_index = self.tree.children_left[node_index]
            right_child_index = self.tree.children_right[node_index]
            is_leaf_node = left_child_index == right_child_index
            if not is_leaf_node:
                # Remove the currently placeholder class node
                fade_out_animation = FadeOut(
                    placeholder_class_nodes[node_index]
                )
                animations.append(fade_out_animation)
                # Fade in the split node
                fade_in_animation = FadeIn(
                    self.nodes_map[node_index]
                )
                animations.append(fade_in_animation)
                # Split the node by creating the children and connecting them
                # to the parent
                # Handle left child
                assert left_child_index in self.nodes_map.keys()
                left_node = self.nodes_map[left_child_index]
                left_parent_edge = self.edge_map[f"{node_index},{left_child_index}"]
                # Get the children of the left node
                left_node_left_index = self.tree.children_left[left_child_index]
                left_node_right_index = self.tree.children_right[left_child_index]
                left_is_leaf = left_node_left_index == left_node_right_index
                if left_is_leaf:
                    # If a child is a leaf then just create it
                    left_animation = FadeIn(left_node)
                else:
                    # If the child is a split node find the dominant class and make a temp
                    left_node_class_index = np.argmax(
                        self.tree.value[left_child_index]
                    )
                    new_leaf_node = LeafNode(
                        class_index=left_node_class_index,
                        class_colors=self.class_colors,
                        class_image_paths=self.class_image_paths,
                    )
                    new_leaf_node.move_to(self.nodes_map[leaf_child_index])
                    placeholder_class_nodes[left_child_index] = new_leaf_node
                    left_animation = AnimationGroup(
                        FadeIn(new_leaf_node),
                        Create(left_parent_edge),
                        lag_ratio=0.0
                    )
                # Handle right child
                assert right_child_index in self.nodes_map.keys()
                right_node = self.nodes_map[right_child_index]
                right_parent_edge = self.edge_map[f"{node_index},{right_child_index}"]
                # Get the children of the left node
                right_node_left_index = self.tree.children_left[right_child_index]
                right_node_right_index = self.tree.children_right[right_child_index]
                right_is_leaf = right_node_left_index == right_node_right_index
                if right_is_leaf:
                    # If a child is a leaf then just create it
                    right_animation = FadeIn(right_node)
                else:
                    # If the child is a split node find the dominant class and make a temp
                    right_node_class_index = np.argmax(
                        self.tree.value[right_child_index]
                    )
                    new_leaf_node = LeafNode(
                        class_index=right_node_class_index,
                        class_colors=self.class_colors,
                        class_image_paths=self.class_image_paths,
                    )
                    placeholder_class_nodes[right_child_index] = new_leaf_node
                    right_animation = AnimationGroup(
                        FadeIn(new_leaf_node),
                        Create(right_parent_edge),
                        lag_ratio=0.0
                    )
                # Combine the animations
                split_animation = AnimationGroup(
                    left_animation,
                    right_animation,
                    lag_ratio=0.0,
                )
                animations.append(split_animation)
                # Add the split animation to the split node dict
                split_node_animations[node_index] = split_animation
            # Add the children to the queue
            if left_child_index != -1:
                queue.append(left_child_index)
            if right_child_index != -1:
                queue.append(right_child_index)

        return Succession(
            *animations, 
            lag_ratio=1.0
        ), split_node_animations

    def make_expand_tree_animation(self, node_expand_order):
        """
            Make an animation for expanding the decision tree

            Shows each split node as a leaf node initially, and
            then when it comes up shows it as a split node. The 
            reason for this is for purposes of animating each of the 
            splits in a decision surface.    
        """
        # Show the root node as a leaf node
        # Iterate through the nodes in the traversal order
        for node_index in node_expand_order[1:]:
            # Figure out if it is a leaf or not
            # If it is not a leaf then remove the placeholder leaf node
            #       then show the split node
            # If it is a leaf then just show the leaf node
            pass
        pass

    @override_animation(Create)
    def create_decision_tree(self, traversal_order="bfs"):
        """Makes a create animation for the decision tree"""
        # Comptue the node expand order
        if traversal_order == "level":
            node_expand_order = helpers.compute_level_order_traversal(self.tree)
        elif traversal_order == "bfs":
            node_expand_order = helpers.compute_bfs_traversal(self.tree)
        else:
            raise Exception(f"Uncrecognized traversal: {traversal_order}")
        # Make the animation
        expand_tree_animation = self.make_expand_tree_animation(node_expand_order)
        return expand_tree_animation

class DecisionTreeContainer():
    """Connects the DecisionTreeDiagram to the DecisionTreeEmbedding"""

    def __init__(self, sklearn_tree, points, classes):
        self.sklearn_tree = sklearn_tree
        self.points = points
        self.classes = classes

    def make_unfold_tree_animation(self):
        """Unfolds the tree through an in order traversal

        This animations unfolds the tree diagram as well as showing the splitting
        of a shaded region in the Decision Tree embedding.
        """
        # Draw points in the embedding
        # Start the tree splitting animation
        pass

"""
    Module for visualizing decision trees in Manim. 
    It parses a decision tree classifier from sklearn. 
"""
from manim import *
from manim_ml.one_to_one_sync import OneToOneSync

import numpy as np
from PIL import Image

def compute_node_depths(tree):
    """Computes the depths of nodes for level order traversal"""
    def depth(node_index, current_node_index=0):
        """Compute the height of a node"""
        if current_node_index == node_index:
            return 0
        elif tree.children_left[current_node_index] == tree.children_right[current_node_index]:
            return -1
        else:
            # Compute the height of each subtree
            l_depth = depth(node_index, tree.children_left[current_node_index])
            r_depth = depth(node_index, tree.children_right[current_node_index])
            # The index is only in one of them
            if l_depth != -1:
                return l_depth + 1
            elif r_depth != -1:
                return r_depth + 1
            else:
                return -1

    node_depths = [depth(index) for index in range(tree.node_count)]

    return node_depths

def compute_level_order_traversal(tree):
    """Computes level order traversal of a sklearn tree"""
    def depth(node_index, current_node_index=0):
        """Compute the height of a node"""
        if current_node_index == node_index:
            return 0
        elif tree.children_left[current_node_index] == tree.children_right[current_node_index]:
            return -1
        else:
            # Compute the height of each subtree
            l_depth = depth(node_index, tree.children_left[current_node_index])
            r_depth = depth(node_index, tree.children_right[current_node_index])
            # The index is only in one of them
            if l_depth != -1:
                return l_depth + 1
            elif r_depth != -1:
                return r_depth + 1
            else:
                return -1

    node_depths = [(index, depth(index)) for index in range(tree.node_count)]
    node_depths = sorted(node_depths, key=lambda x: x[1])
    sorted_inds = [node_depth[0] for node_depth in node_depths]

    return sorted_inds 

def compute_node_to_parent_mapping(tree):
    """Returns a hashmap mapping node indices to their parent indices"""
    node_to_parent = {0: -1} # Root has no parent
    num_nodes = tree.node_count
    for node_index in range(num_nodes):
        # Explore left children 
        left_child_node_index = tree.children_left[node_index]
        if left_child_node_index != -1:
            node_to_parent[left_child_node_index] = node_index
        # Explore right children
        right_child_node_index = tree.children_right[node_index]
        if right_child_node_index != -1:
            node_to_parent[right_child_node_index] = node_index
    
    return node_to_parent

class LeafNode(Group):
    """Leaf node in tree"""

    def __init__(self, class_index, display_type="image", class_image_paths=[],
            class_colors=[]):
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
            stroke_width=6
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
        decision_text = Text(
            node_text, 
            color=WHITE
        )
        # Draw the surrounding box
        bounding_box = SurroundingRectangle(
            decision_text, 
            buff=0.3, 
            color=WHITE
        )
        self.add(bounding_box)
        self.add(decision_text)

class DecisionTreeDiagram(Group):
    """Decision Tree Diagram Class for Manim"""

    def __init__(self, sklearn_tree, feature_names=None,
                class_names=None, class_images_paths=None,
                class_colors=[RED, GREEN, BLUE]):
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
        is_split_node = self.tree.children_left[node_index] != self.tree.children_right[node_index]
        if is_split_node:
            node_feature = self.tree.feature[node_index]
            node_threshold = self.tree.threshold[node_index]
            node = SplitNode(
                self.feature_names[node_feature], 
                node_threshold
            )
        else:
            # Get the most abundant class for the given leaf node
            # Make the leaf node object
            tree_class_index = np.argmax(self.tree.value[node_index])
            node = LeafNode(
                class_index=tree_class_index,
                class_colors=self.class_colors,
                class_image_paths=self.class_image_paths
            )
        return node

    def _make_connection(self, top, bottom, is_leaf=False):
        """Make a connection from top to bottom"""
        top_node_bottom_location = top.get_center()
        top_node_bottom_location[1] -= top.height / 2
        bottom_node_top_location = bottom.get_center()
        bottom_node_top_location[1] += bottom.height / 2

        line = Line(
            top_node_bottom_location, 
            bottom_node_top_location, 
            color=WHITE
        )

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
        tree_width = scale_factor * 2 ** max_depth * node_width
        # traverse tree
        def recurse(node_index, depth, direction, parent_object, parent_node):
            # make the node object
            is_leaf = self.tree.children_left[node_index] == self.tree.children_right[node_index]
            node_object = self._make_node(node_index=node_index)
            nodes_map[node_index] = node_object
            node_height = node_object.height
            # set the node position
            direction_factor = -1 if direction == "left" else 1
            shift_right_amount = 0.9 * direction_factor * scale_factor * tree_width / (2 ** depth) / 2
            if is_leaf:
                shift_down_amount = -1.0 * scale_factor * node_height
            else:
                shift_down_amount = -1.8 * scale_factor * node_height
            node_object \
                .match_x(parent_object) \
                .match_y(parent_object) \
                .shift([shift_right_amount, shift_down_amount, 0])
            tree_group.add(node_object)
            # make a connection
            connection = self._make_connection(parent_object, node_object, is_leaf=is_leaf)
            edge_name = str(parent_node)+","+str(node_index)
            edge_map[edge_name] = connection
            tree_group.add(connection)
            # recurse
            if not is_leaf:
                recurse(self.tree.children_left[node_index], depth + 1, "left", node_object, node_index)
                recurse(self.tree.children_right[node_index], depth + 1, "right", node_object, node_index)

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
        # Compute parent mapping
        parent_mapping = compute_node_to_parent_mapping(self.tree)
        # Create the root node
        animations.append(
            Create(self.nodes_map[0])
        )
        # Iterate through the nodes
        queue = [0]
        while len(queue) > 0:
            node_index = queue.pop(0)
            # Check if a node is a split node or not
            left_child = self.tree.children_left[node_index]
            right_child = self.tree.children_right[node_index]
            is_leaf_node = left_child == right_child
            if not is_leaf_node:
                # Split the node by creating the children and connecting them
                # to the parent
                # Get the nodes
                left_node = self.nodes_map[left_child]
                right_node = self.nodes_map[right_child]
                # Get the parent edges
                left_parent_edge = self.edge_map[f"{node_index},{left_child}"]
                right_parent_edge = self.edge_map[f"{node_index},{right_child}"]
                # Create the children
                split_animation = AnimationGroup(
                    FadeIn(left_node),
                    FadeIn(right_node),
                    Create(left_parent_edge),
                    Create(right_parent_edge),
                    lag_ratio=0.0
                )
                animations.append(
                    split_animation
                )
            # Add the children to the queue
            if left_child != -1:
                queue.append(left_child)
            if right_child != -1:
                queue.append(right_child)

        return AnimationGroup(
            *animations,
            lag_ratio=1.0
        )

    @override_animation(Create)
    def create_decision_tree(self, traversal_order="bfs"):
        """Makes a create animation for the decision tree"""
        if traversal_order == "level":
            return self.create_level_order_expansion_decision_tree(self.tree)
        elif traversal_order == "bfs":
            return self.create_bfs_expansion_decision_tree(self.tree)
        else:
            raise Exception(f"Uncrecognized traversal: {traversal_order}")

class DecisionTreeEmbedding:
    """Embedding for the decision tree"""

    pass

class DecisionTreeContainer(OneToOneSync):
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

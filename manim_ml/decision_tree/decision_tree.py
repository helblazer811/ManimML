"""
    Module for visualizing decision trees in Manim. 
    It parses a decision tree classifier from sklearn. 

    TODO return a map from nodes to split animation for BFS tree expansion
    TODO reimplement the decision 2D decision tree surface drawing. 
"""
from manim import *
from manim_ml.decision_tree.classification_areas import (
    compute_decision_areas,
    merge_overlapping_polygons,
)
import manim_ml.decision_tree.helpers as helpers
from manim_ml.one_to_one_sync import OneToOneSync

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
        # Compute parent mapping
        parent_mapping = helpers.compute_node_to_parent_mapping(self.tree)
        # Create the root node
        animations.append(Create(self.nodes_map[0]))
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
                    lag_ratio=0.0,
                )
                animations.append(split_animation)
            # Add the children to the queue
            if left_child != -1:
                queue.append(left_child)
            if right_child != -1:
                queue.append(right_child)

        return AnimationGroup(*animations, lag_ratio=1.0)

    @override_animation(Create)
    def create_decision_tree(self, traversal_order="bfs"):
        """Makes a create animation for the decision tree"""
        if traversal_order == "level":
            return self.create_level_order_expansion_decision_tree(self.tree)
        elif traversal_order == "bfs":
            return self.create_bfs_expansion_decision_tree(self.tree)
        else:
            raise Exception(f"Uncrecognized traversal: {traversal_order}")


class IrisDatasetPlot(VGroup):
    def __init__(self, iris):
        points = iris.data[:, 0:2]
        labels = iris.feature_names
        targets = iris.target
        # Make points
        self.point_group = self._make_point_group(points, targets)
        # Make axes
        self.axes_group = self._make_axes_group(points, labels)
        # Make legend
        self.legend_group = self._make_legend(
            [BLUE, ORANGE, GREEN], iris.target_names, self.axes_group
        )
        # Make title
        # title_text = "Iris Dataset Plot"
        # self.title = Text(title_text).match_y(self.axes_group).shift([0.5, self.axes_group.height / 2 + 0.5, 0])
        # Make all group
        self.all_group = Group(self.point_group, self.axes_group, self.legend_group)
        # scale the groups
        self.point_group.scale(1.6)
        self.point_group.match_x(self.axes_group)
        self.point_group.match_y(self.axes_group)
        self.point_group.shift([0.2, 0, 0])
        self.axes_group.scale(0.7)
        self.all_group.shift([0, 0.2, 0])

    @override_animation(Create)
    def create_animation(self):
        animation_group = AnimationGroup(
            # Perform the animations
            Create(self.point_group, run_time=2),
            Wait(0.5),
            Create(self.axes_group, run_time=2),
            # add title
            # Create(self.title),
            Create(self.legend_group),
        )
        return animation_group

    def _make_point_group(self, points, targets, class_colors=[BLUE, ORANGE, GREEN]):
        point_group = VGroup()
        for point_index, point in enumerate(points):
            # draw the dot
            current_target = targets[point_index]
            color = class_colors[current_target]
            dot = Dot(point=np.array([point[0], point[1], 0])).set_color(color)
            dot.scale(0.5)
            point_group.add(dot)
        return point_group

    def _make_legend(self, class_colors, feature_labels, axes):
        legend_group = VGroup()
        # Make Text
        setosa = Text("Setosa", color=BLUE)
        verisicolor = Text("Verisicolor", color=ORANGE)
        virginica = Text("Virginica", color=GREEN)
        labels = VGroup(setosa, verisicolor, virginica).arrange(
            direction=RIGHT, aligned_edge=LEFT, buff=2.0
        )
        labels.scale(0.5)
        legend_group.add(labels)
        # surrounding rectangle
        surrounding_rectangle = SurroundingRectangle(labels, color=WHITE)
        surrounding_rectangle.move_to(labels)
        legend_group.add(surrounding_rectangle)
        # shift the legend group
        legend_group.move_to(axes)
        legend_group.shift([0, -3.0, 0])
        legend_group.match_x(axes[0][0])

        return legend_group

    def _make_axes_group(self, points, labels, font="Source Han Sans", font_scale=0.75):
        axes_group = VGroup()
        # make the axes
        x_range = [
            np.amin(points, axis=0)[0] - 0.2,
            np.amax(points, axis=0)[0] - 0.2,
            0.5,
        ]
        y_range = [np.amin(points, axis=0)[1] - 0.2, np.amax(points, axis=0)[1], 0.5]
        axes = Axes(
            x_range=x_range,
            y_range=y_range,
            x_length=9,
            y_length=6.5,
            # axis_config={"number_scale_value":0.75, "include_numbers":True},
            tips=False,
        ).shift([0.5, 0.25, 0])
        axes_group.add(axes)
        # make axis labels
        # x_label
        x_label = (
            Text(labels[0], font=font)
            .match_y(axes.get_axes()[0])
            .shift([0.5, -0.75, 0])
            .scale(font_scale)
        )
        axes_group.add(x_label)
        # y_label
        y_label = (
            Text(labels[1], font=font)
            .match_x(axes.get_axes()[1])
            .shift([-0.75, 0, 0])
            .rotate(np.pi / 2)
            .scale(font_scale)
        )
        axes_group.add(y_label)

        return axes_group


class DecisionTreeSurface(VGroup):
    def __init__(self, tree_clf, data, axes, class_colors=[BLUE, ORANGE, GREEN]):
        # take the tree and construct the surface from it
        self.tree_clf = tree_clf
        self.data = data
        self.axes = axes
        self.class_colors = class_colors
        self.surface_rectangles = self.generate_surface_rectangles()

    def generate_surface_rectangles(self):
        # compute data bounds
        left = np.amin(self.data[:, 0]) - 0.2
        right = np.amax(self.data[:, 0]) - 0.2
        top = np.amax(self.data[:, 1])
        bottom = np.amin(self.data[:, 1]) - 0.2
        maxrange = [left, right, bottom, top]
        rectangles = compute_decision_areas(
            self.tree_clf, maxrange, x=0, y=1, n_features=2
        )
        # turn the rectangle objects into manim rectangles
        def convert_rectangle_to_polygon(rect):
            # get the points for the rectangle in the plot coordinate frame
            bottom_left = [rect[0], rect[3]]
            bottom_right = [rect[1], rect[3]]
            top_right = [rect[1], rect[2]]
            top_left = [rect[0], rect[2]]
            # convert those points into the entire manim coordinates
            bottom_left_coord = self.axes.coords_to_point(*bottom_left)
            bottom_right_coord = self.axes.coords_to_point(*bottom_right)
            top_right_coord = self.axes.coords_to_point(*top_right)
            top_left_coord = self.axes.coords_to_point(*top_left)
            points = [
                bottom_left_coord,
                bottom_right_coord,
                top_right_coord,
                top_left_coord,
            ]
            # construct a polygon object from those manim coordinates
            rectangle = Polygon(
                *points, color=color, fill_opacity=0.3, stroke_opacity=0.0
            )
            return rectangle

        manim_rectangles = []
        for rect in rectangles:
            color = self.class_colors[int(rect[4])]
            rectangle = convert_rectangle_to_polygon(rect)
            manim_rectangles.append(rectangle)

        manim_rectangles = merge_overlapping_polygons(
            manim_rectangles, colors=[BLUE, GREEN, ORANGE]
        )

        return manim_rectangles

    @override_animation(Create)
    def create_override(self):
        # play a reveal of all of the surface rectangles
        animations = []
        for rectangle in self.surface_rectangles:
            animations.append(Create(rectangle))
        animation_group = AnimationGroup(*animations)

        return animation_group

    @override_animation(Uncreate)
    def uncreate_override(self):
        # play a reveal of all of the surface rectangles
        animations = []
        for rectangle in self.surface_rectangles:
            animations.append(Uncreate(rectangle))
        animation_group = AnimationGroup(*animations)

        return animation_group


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

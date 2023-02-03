from manim import *
import numpy as np
from collections import deque
from sklearn.tree import _tree as ctree

class AABB:
    """Axis-aligned bounding box"""

    def __init__(self, n_features):
        self.limits = np.array([[-np.inf, np.inf]] * n_features)

    def split(self, f, v):
        left = AABB(self.limits.shape[0])
        right = AABB(self.limits.shape[0])
        left.limits = self.limits.copy()
        right.limits = self.limits.copy()
        left.limits[f, 1] = v
        right.limits[f, 0] = v

        return left, right

def tree_bounds(tree, n_features=None):
    """Compute final decision rule for each node in tree"""
    if n_features is None:
        n_features = np.max(tree.feature) + 1
    aabbs = [AABB(n_features) for _ in range(tree.node_count)]
    queue = deque([0])
    while queue:
        i = queue.pop()
        l = tree.children_left[i]
        r = tree.children_right[i]
        if l != ctree.TREE_LEAF:
            aabbs[l], aabbs[r] = aabbs[i].split(tree.feature[i], tree.threshold[i])
            queue.extend([l, r])
    return aabbs

def compute_decision_areas(
    tree_classifier, 
    maxrange, 
    x=0,
    y=1, 
    n_features=None
):
    """Extract decision areas.

    tree_classifier: Instance of a sklearn.tree.DecisionTreeClassifier
    maxrange: values to insert for [left, right, top, bottom] if the interval is open (+/-inf)
    x: index of the feature that goes on the x axis
    y: index of the feature that goes on the y axis
    n_features: override autodetection of number of features
    """
    tree = tree_classifier.tree_
    aabbs = tree_bounds(tree, n_features)
    maxrange = np.array(maxrange)
    rectangles = []
    for i in range(len(aabbs)):
        if tree.children_left[i] != ctree.TREE_LEAF:
            continue
        l = aabbs[i].limits
        r = [l[x, 0], l[x, 1], l[y, 0], l[y, 1], np.argmax(tree.value[i])]
        # clip out of bounds indices
        """
        if r[0] < maxrange[0]:
            r[0] = maxrange[0]
        if r[1] > maxrange[1]:
            r[1] = maxrange[1]
        if r[2] < maxrange[2]:
            r[2] = maxrange[2]
        if r[3] > maxrange[3]:
            r[3] = maxrange[3]
        print(r)
        """
        rectangles.append(r)
    rectangles = np.array(rectangles)
    rectangles[:, [0, 2]] = np.maximum(rectangles[:, [0, 2]], maxrange[0::2])
    rectangles[:, [1, 3]] = np.minimum(rectangles[:, [1, 3]], maxrange[1::2])
    return rectangles

def plot_areas(rectangles):
    for rect in rectangles:
        color = ["b", "r"][int(rect[4])]
        print(rect[0], rect[1], rect[2] - rect[0], rect[3] - rect[1])
        rp = Rectangle(
            [rect[0], rect[2]],
            rect[1] - rect[0],
            rect[3] - rect[2],
            color=color,
            alpha=0.3,
        )
        plt.gca().add_artist(rp)

def merge_overlapping_polygons(all_polygons, colors=[BLUE, GREEN, ORANGE]):
    # get all polygons of each color
    polygon_dict = {
        str(BLUE).lower(): [],
        str(GREEN).lower(): [],
        str(ORANGE).lower(): [],
    }
    for polygon in all_polygons:
        print(polygon_dict)
        polygon_dict[str(polygon.color).lower()].append(polygon)

    return_polygons = []
    for color in colors:
        color = str(color).lower()
        polygons = polygon_dict[color]
        points = set()
        for polygon in polygons:
            vertices = polygon.get_vertices().tolist()
            vertices = [tuple(vert) for vert in vertices]
            for pt in vertices:
                if pt in points:  # Shared vertice, remove it.
                    points.remove(pt)
                else:
                    points.add(pt)
        points = list(points)
        sort_x = sorted(points)
        sort_y = sorted(points, key=lambda x: x[1])

        edges_h = {}
        edges_v = {}

        i = 0
        while i < len(points):
            curr_y = sort_y[i][1]
            while i < len(points) and sort_y[i][1] == curr_y:
                edges_h[sort_y[i]] = sort_y[i + 1]
                edges_h[sort_y[i + 1]] = sort_y[i]
                i += 2
        i = 0
        while i < len(points):
            curr_x = sort_x[i][0]
            while i < len(points) and sort_x[i][0] == curr_x:
                edges_v[sort_x[i]] = sort_x[i + 1]
                edges_v[sort_x[i + 1]] = sort_x[i]
                i += 2

        # Get all the polygons.
        while edges_h:
            # We can start with any point.
            polygon = [(edges_h.popitem()[0], 0)]
            while True:
                curr, e = polygon[-1]
                if e == 0:
                    next_vertex = edges_v.pop(curr)
                    polygon.append((next_vertex, 1))
                else:
                    next_vertex = edges_h.pop(curr)
                    polygon.append((next_vertex, 0))
                if polygon[-1] == polygon[0]:
                    # Closed polygon
                    polygon.pop()
                    break
            # Remove implementation-markers from the polygon.
            poly = [point for point, _ in polygon]
            for vertex in poly:
                if vertex in edges_h:
                    edges_h.pop(vertex)
                if vertex in edges_v:
                    edges_v.pop(vertex)
            polygon = Polygon(*poly, color=color, fill_opacity=0.3, stroke_opacity=1.0)
            return_polygons.append(polygon)
    return return_polygons

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

    def make_split_to_animation_map(self):
        """
        Returns a dictionary mapping a given split
        node to an animation to be played
        """
        # Create an initial decision tree surface
        # Go through each split node
        # 1. Make a line split animation
        # 2. Create the relevant classification areas
        #    and transform the old ones to them
        pass

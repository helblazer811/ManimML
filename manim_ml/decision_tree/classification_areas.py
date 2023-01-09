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


def compute_decision_areas(tree_classifier, maxrange, x=0, y=1, n_features=None):
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

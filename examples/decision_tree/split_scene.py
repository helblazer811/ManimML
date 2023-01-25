from sklearn import datasets
from decision_tree_surface import *
from manim import *
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import entropy
import math
from PIL import Image

iris = datasets.load_iris()
font = "Source Han Sans"
font_scale = 0.75

images = [
    Image.open("iris_dataset/SetosaFlower.jpeg"),
    Image.open("iris_dataset/VeriscolorFlower.jpeg"),
    Image.open("iris_dataset/VirginicaFlower.jpeg"),
]


def entropy(class_labels, base=2):
    # compute the class counts
    unique, counts = np.unique(class_labels, return_counts=True)
    dictionary = dict(zip(unique, counts))
    total = 0.0
    num_samples = len(class_labels)
    for class_index in range(0, 3):
        if not class_index in dictionary:
            continue
        prob = dictionary[class_index] / num_samples
        total += prob * math.log(prob, base)
    # higher set
    return -total


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
    def __init__(self):
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

    def _make_axes_group(self, points, labels):
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
        rectangles = decision_areas(self.tree_clf, maxrange, x=0, y=1, n_features=2)
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


class DecisionTree:
    """
    Draw a single tree node
    """

    def _make_node(
        self,
        feature,
        threshold,
        values,
        is_leaf=False,
        depth=0,
        leaf_colors=[BLUE, ORANGE, GREEN],
    ):
        if not is_leaf:
            node_text = f"{feature}\n    <= {threshold:.3f} cm"
            # draw decision text
            decision_text = Text(node_text, color=WHITE)
            # draw a box
            bounding_box = SurroundingRectangle(decision_text, buff=0.3, color=WHITE)
            node = VGroup()
            node.add(bounding_box)
            node.add(decision_text)
            # return the node
        else:
            # plot the appropriate image
            class_index = np.argmax(values)
            # get image
            pil_image = images[class_index]
            leaf_group = Group()
            node = ImageMobject(pil_image)
            node.scale(1.5)
            rectangle = Rectangle(
                width=node.width + 0.05,
                height=node.height + 0.05,
                color=leaf_colors[class_index],
                stroke_width=6,
            )
            rectangle.move_to(node.get_center())
            rectangle.shift([-0.02, 0.02, 0])
            leaf_group.add(rectangle)
            leaf_group.add(node)
            node = leaf_group

        return node

    def _make_connection(self, top, bottom, is_leaf=False):
        top_node_bottom_location = top.get_center()
        top_node_bottom_location[1] -= top.height / 2
        bottom_node_top_location = bottom.get_center()
        bottom_node_top_location[1] += bottom.height / 2
        line = Line(top_node_bottom_location, bottom_node_top_location, color=WHITE)
        return line

    def _make_tree(self, tree, feature_names=["Sepal Length", "Sepal Width"]):
        tree_group = Group()
        max_depth = tree.max_depth
        # make the base node
        feature_name = feature_names[tree.feature[0]]
        threshold = tree.threshold[0]
        values = tree.value[0]
        nodes_map = {}
        root_node = self._make_node(feature_name, threshold, values, depth=0)
        nodes_map[0] = root_node
        tree_group.add(root_node)
        # save some information
        node_height = root_node.height
        node_width = root_node.width
        scale_factor = 1.0
        edge_map = {}
        # tree height
        tree_height = scale_factor * node_height * max_depth
        tree_width = scale_factor * 2**max_depth * node_width
        # traverse tree
        def recurse(node, depth, direction, parent_object, parent_node):
            # make sure it is a valid node
            # make the node object
            is_leaf = tree.children_left[node] == tree.children_right[node]
            feature_name = feature_names[tree.feature[node]]
            threshold = tree.threshold[node]
            values = tree.value[node]
            node_object = self._make_node(
                feature_name, threshold, values, depth=depth, is_leaf=is_leaf
            )
            nodes_map[node] = node_object
            node_height = node_object.height
            # set the node position
            direction_factor = -1 if direction == "left" else 1
            shift_right_amount = (
                0.8 * direction_factor * scale_factor * tree_width / (2**depth) / 2
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
            edge_name = str(parent_node) + "," + str(node)
            edge_map[edge_name] = connection
            tree_group.add(connection)
            # recurse
            if not is_leaf:
                recurse(tree.children_left[node], depth + 1, "left", node_object, node)
                recurse(
                    tree.children_right[node], depth + 1, "right", node_object, node
                )

        recurse(tree.children_left[0], 1, "left", root_node, 0)
        recurse(tree.children_right[0], 1, "right", root_node, 0)

        tree_group.scale(0.35)
        return tree_group, nodes_map, edge_map

    def color_example_path(
        self, tree_group, nodes_map, tree, edge_map, example, color=YELLOW, thickness=2
    ):
        # get decision path
        decision_path = tree.decision_path(example)[0]
        path_indices = decision_path.indices
        # highlight edges
        for node_index in range(0, len(path_indices) - 1):
            current_val = path_indices[node_index]
            next_val = path_indices[node_index + 1]
            edge_str = str(current_val) + "," + str(next_val)
            edge = edge_map[edge_str]
            animation_two = AnimationGroup(
                nodes_map[current_val].animate.set_color(color)
            )
            self.play(animation_two, run_time=0.5)
            animation_one = AnimationGroup(
                edge.animate.set_color(color),
                # edge.animate.set_stroke_width(4),
            )
            self.play(animation_one, run_time=0.5)
        # surround the bottom image
        last_path_index = path_indices[-1]
        last_path_rectangle = nodes_map[last_path_index][0]
        self.play(last_path_rectangle.animate.set_color(color))

    def create_sklearn_tree(self, max_tree_depth=1):
        # learn the decision tree
        iris = load_iris()
        tree = learn_iris_decision_tree(iris, max_depth=max_tree_depth)
        feature_names = iris.feature_names[0:2]
        return tree.tree_

    def make_tree(self, max_tree_depth=2):
        sklearn_tree = self.create_sklearn_tree()
        # make the tree
        tree_group, nodes_map, edge_map = self._make_tree(
            sklearn_tree.tree_, feature_names
        )
        tree_group.shift([0, 5.5, 0])
        return tree_group
        # self.add(tree_group)
        # self.play(SpinInFromNothing(tree_group), run_time=3)
        # self.color_example_path(tree_group, nodes_map, tree, edge_map, iris.data[None, 0, 0:2])


class DecisionTreeSplitScene(Scene):
    def make_decision_tree_classifier(self, max_depth=4):
        decision_tree = DecisionTreeClassifier(
            random_state=1, max_depth=max_depth, max_leaf_nodes=8
        )
        decision_tree = decision_tree.fit(iris.data[:, :2], iris.target)
        # output the decisioin tree in some format
        return decision_tree

    def make_split_animation(self, data, classes, data_labels, main_axes):

        """
        def make_entropy_animation_and_plot(dim=0, num_entropy_values=50):
            # calculate the entropy values
            axes_group = VGroup()
            # make axes
            range_vals = [np.amin(data, axis=0)[dim], np.amax(data, axis=0)[dim]]
            axes = Axes(x_range=range_vals,
                        y_range=[0, 1.0],
                        x_length=9,
                        y_length=4,
            #            axis_config={"number_scale_value":0.75, "include_numbers":True},
                        tips=False,
                    )
            axes_group.add(axes)
            # make axis labels
            # x_label
            x_label = Text(data_labels[dim], font=font) \
                            .match_y(axes.get_axes()[0]) \
                            .shift([0.5, -0.75, 0]) \
                            .scale(font_scale*1.2)
            axes_group.add(x_label)
            # y_label
            y_label = Text("Information Gain", font=font) \
                .match_x(axes.get_axes()[1]) \
                .shift([-0.75, 0, 0]) \
                .rotate(np.pi / 2) \
                .scale(font_scale * 1.2)

            axes_group.add(y_label)
            # line animation
            information_gains = []
            def entropy_function(split_value):
                # lower entropy
                lower_set = np.nonzero(data[:, dim] <= split_value)[0]
                lower_set = classes[lower_set]
                lower_entropy = entropy(lower_set)
                # higher entropy
                higher_set = np.nonzero(data[:, dim] > split_value)[0]
                higher_set = classes[higher_set]
                higher_entropy = entropy(higher_set)
                # calculate entropies
                all_entropy = entropy(classes, base=2)
                lower_entropy = entropy(lower_set, base=2)
                higher_entropy = entropy(higher_set, base=2)
                mean_entropy = (lower_entropy + higher_entropy) / 2
                # calculate information gain
                lower_prob = len(lower_set) / len(data[:, dim])
                higher_prob = len(higher_set) / len(data[:, dim])
                info_gain = all_entropy - (lower_prob * lower_entropy + higher_prob * higher_entropy)
                information_gains.append((split_value, info_gain))
                return info_gain

            data_range = np.amin(data[:, dim]), np.amax(data[:, dim])

            entropy_graph = axes.get_graph(
                entropy_function, 
            #    color=RED, 
            #    x_range=data_range
            )
            axes_group.add(entropy_graph)
            axes_group.shift([4.0, 2, 0])
            axes_group.scale(0.5)
            dot_animation = Dot(color=WHITE)
            axes_group.add(dot_animation)
            # make animations
            animation_group = AnimationGroup(
                Create(axes_group, run_time=2),
                Wait(3),
                MoveAlongPath(dot_animation, entropy_graph, run_time=20, rate_func=rate_functions.ease_in_out_quad),
                Wait(2)
            )

            return axes_group, animation_group, information_gains
        """

        def make_split_line_animation(dim=0):
            # make a line along one of the dims and move it up and down
            origin_coord = [
                np.amin(data, axis=0)[0] - 0.2,
                np.amin(data, axis=0)[1] - 0.2,
            ]
            origin_point = main_axes.coords_to_point(*origin_coord)
            top_left_coord = [origin_coord[0], np.amax(data, axis=0)[1]]
            bottom_right_coord = [np.amax(data, axis=0)[0] - 0.2, origin_coord[1]]
            if dim == 0:
                other_coord = top_left_coord
                moving_line_coord = bottom_right_coord
            else:
                other_coord = bottom_right_coord
                moving_line_coord = top_left_coord
            other_point = main_axes.coords_to_point(*other_coord)
            moving_line_point = main_axes.coords_to_point(*moving_line_coord)
            moving_line = Line(origin_point, other_point, color=RED)
            movement_line = Line(origin_point, moving_line_point)
            if dim == 0:
                movement_line.shift([0, moving_line.height / 2, 0])
            else:
                movement_line.shift([moving_line.width / 2, 0, 0])
            # move the moving line along the movement line
            animation = MoveAlongPath(
                moving_line,
                movement_line,
                run_time=20,
                rate_func=rate_functions.ease_in_out_quad,
            )
            return animation, moving_line

        # plot the line in white then make it invisible
        # make an animation along the line
        # make a
        # axes_one_group, top_animation_group, info_gains = make_entropy_animation_and_plot(dim=0)
        line_movement, first_moving_line = make_split_line_animation(dim=0)
        # axes_two_group, bottom_animation_group, _ = make_entropy_animation_and_plot(dim=1)
        second_line_movement, second_moving_line = make_split_line_animation(dim=1)
        # axes_two_group.shift([0, -3, 0])
        animation_group_one = AnimationGroup(
            #    top_animation_group,
            line_movement,
        )

        animation_group_two = AnimationGroup(
            #    bottom_animation_group,
            second_line_movement,
        )
        """
        both_axes_group = VGroup(
            axes_one_group,
            axes_two_group
        )
        """

        return (
            animation_group_one,
            animation_group_two,
            first_moving_line,
            second_moving_line,
            None,
            None,
        )
        #    both_axes_group, \
        #    info_gains

    def construct(self):
        # make the points
        iris_dataset_plot = IrisDatasetPlot()
        iris_dataset_plot.all_group.scale(1.0)
        iris_dataset_plot.all_group.shift([-3, 0.2, 0])
        # make the entropy line graph
        # entropy_line_graph = self.draw_entropy_line_graph()
        # arrange the plots
        # do animations
        self.play(Create(iris_dataset_plot))
        # make the decision tree classifier
        decision_tree_classifier = self.make_decision_tree_classifier()
        decision_tree_surface = DecisionTreeSurface(
            decision_tree_classifier, iris.data, iris_dataset_plot.axes_group[0]
        )
        self.play(Create(decision_tree_surface))
        self.wait(3)
        self.play(Uncreate(decision_tree_surface))
        main_axes = iris_dataset_plot.axes_group[0]
        (
            split_animation_one,
            split_animation_two,
            first_moving_line,
            second_moving_line,
            both_axes_group,
            info_gains,
        ) = self.make_split_animation(
            iris.data[:, 0:2], iris.target, iris.feature_names, main_axes
        )
        self.play(split_animation_one)
        self.wait(0.1)
        self.play(Uncreate(first_moving_line))
        self.wait(3)
        self.play(split_animation_two)
        self.wait(0.1)
        self.play(Uncreate(second_moving_line))
        self.wait(0.1)
        # highlight the maximum on top
        # sort by second key
        """
        highest_info_gain = sorted(info_gains, key=lambda x: x[1])[-1]
        highest_info_gain_point = both_axes_group[0][0].coords_to_point(*highest_info_gain)
        highlighted_peak = Dot(highest_info_gain_point, color=YELLOW)
        # get location of highest info gain point
        highest_info_gain_point_in_iris_graph = iris_dataset_plot.axes_group[0].coords_to_point(*[highest_info_gain[0], 0])
        first_moving_line.start[0] = highest_info_gain_point_in_iris_graph[0]
        first_moving_line.end[0] = highest_info_gain_point_in_iris_graph[0]
        self.play(Create(highlighted_peak))
        self.play(Create(first_moving_line))
        text = Text("Highest Information Gain")
        text.scale(0.4)
        text.move_to(highlighted_peak)
        text.shift([0, 0.5, 0])
        self.play(Create(text))
        """
        self.wait(1)
        # draw the basic tree
        decision_tree_classifier = self.make_decision_tree_classifier(max_depth=1)
        decision_tree_surface = DecisionTreeSurface(
            decision_tree_classifier, iris.data, iris_dataset_plot.axes_group[0]
        )
        decision_tree_graph, _, _ = DecisionTree()._make_tree(
            decision_tree_classifier.tree_
        )
        decision_tree_graph.match_y(iris_dataset_plot.axes_group)
        decision_tree_graph.shift([4, 0, 0])
        self.play(Create(decision_tree_surface))
        uncreate_animation = AnimationGroup(
            #    Uncreate(both_axes_group),
            #    Uncreate(highlighted_peak),
            Uncreate(second_moving_line),
            #    Unwrite(text)
        )
        self.play(uncreate_animation)
        self.wait(0.5)
        self.play(FadeIn(decision_tree_graph))
        # self.play(FadeIn(highlighted_peak))
        self.wait(5)

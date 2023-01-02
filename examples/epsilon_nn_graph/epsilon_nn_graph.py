"""
    Example where I draw an epsilon nearest neighbor graph animation
"""
from cProfile import label
from manim import *
from sklearn.datasets import make_moons
from sklearn.cluster import SpectralClustering
import numpy as np

# Make the specific scene
config.pixel_height = 1200
config.pixel_width = 1200
config.frame_height = 12.0
config.frame_width = 12.0


def make_moon_points(num_samples=100, noise=0.1, random_seed=1):
    """Make two half moon point shapes"""
    # Make sure the points are normalized
    X, y = make_moons(n_samples=num_samples, noise=noise, random_state=random_seed)
    X -= np.mean(X, axis=0)
    X /= np.std(X, axis=0)
    X[:, 1] += 0.3
    # X[:, 0] /= 2 # squeeze width

    return X


def make_epsilon_balls(epsilon_value, points, axes, ball_color=RED, opacity=0.0):
    """Draws epsilon balls"""
    balls = []
    for point in points:
        ball = Circle(epsilon_value, color=ball_color, fill_opacity=opacity)
        global_location = axes.coords_to_point(*point)
        ball.move_to(global_location)
        balls.append(ball)

    return VGroup(*balls)


def make_epsilon_graph(epsilon_value, dots, points, edge_color=ORANGE):
    """Makes an epsilon nearest neighbor graph for the given dots"""
    # First compute the adjacency matrix from the epsilon value and the points
    num_dots = len(dots)
    adjacency_matrix = np.zeros((num_dots, num_dots))
    # Note: just doing lower triangular matrix
    for i in range(num_dots):
        for j in range(i):
            dist = np.linalg.norm(dots[i].get_center() - dots[j].get_center())
            is_connected = 1 if dist < epsilon_value else 0
            adjacency_matrix[i, j] = is_connected
    # Draw a graph based on the adjacency matrix
    edges = []
    for i in range(num_dots):
        for j in range(i):
            is_connected = adjacency_matrix[i, j]
            if is_connected:
                # Draw a connection between the corresponding dots
                dot_a = dots[i]
                dot_b = dots[j]
                edge = Line(
                    dot_a.get_center(),
                    dot_b.get_center(),
                    color=edge_color,
                    stroke_width=3,
                )
                edges.append(edge)

    return VGroup(*edges), adjacency_matrix


def perform_spectral_clustering(adjacency_matrix):
    """Performs spectral clustering given adjacency matrix"""
    clustering = SpectralClustering(
        n_clusters=2, affinity="precomputed", random_state=0
    ).fit(adjacency_matrix)
    labels = clustering.labels_

    return labels


def make_color_change_animation(labels, dots, colors=[ORANGE, GREEN]):
    """Makes a color change animation"""
    anims = []

    for index in range(len(labels)):
        color = colors[labels[index]]
        dot = dots[index]
        anims.append(dot.animate.set_color(color))

    return AnimationGroup(*anims, lag_ratio=0.0)


class EpsilonNearestNeighborScene(Scene):
    def construct(
        self,
        num_points=200,
        dot_radius=0.1,
        dot_color=BLUE,
        ball_color=WHITE,
        noise=0.1,
        ball_opacity=0.0,
        random_seed=2,
    ):
        # Make moon shape points
        # Note: dot is the drawing object and point is the math concept
        moon_points = make_moon_points(
            num_samples=num_points, noise=noise, random_seed=random_seed
        )
        # Make an axes
        axes = Axes(
            x_range=[-6, 6, 1],
            y_range=[-6, 6, 1],
            x_length=12,
            y_length=12,
            tips=False,
            axis_config={"stroke_color": "#000000"},
        )
        axes.scale(2.2)
        self.add(axes)
        # Draw points
        dots = []
        for point in moon_points:
            axes_location = axes.coords_to_point(*point)
            dot = Dot(axes_location, color=dot_color, radius=dot_radius, z_index=1)
            dots.append(dot)

        dots = VGroup(*dots)
        self.play(Create(dots))
        # Draw epsilon bar with initial value
        epsilon_bar = NumberLine(
            [0, 2], length=8, stroke_width=2, include_ticks=False, include_numbers=False
        )
        epsilon_bar.shift(4.5 * DOWN)
        self.play(Create(epsilon_bar))
        current_epsilon = ValueTracker(0.3)
        epsilon_point = epsilon_bar.number_to_point(current_epsilon.get_value())
        epsilon_dot = Dot(epsilon_point)
        self.add(epsilon_dot)
        label_location = epsilon_bar.number_to_point(1.0)
        label_location -= DOWN * 0.1
        label_text = MathTex("\epsilon").scale(1.5)
        # label_text = Text("Epsilon")
        label_text.move_to(epsilon_bar.get_center())
        label_text.shift(DOWN * 0.5)
        self.add(label_text)
        # Make an updater for the dot
        def dot_updater(epsilon_dot):
            # Get location on epsilon_bar
            point_loc = epsilon_bar.number_to_point(current_epsilon.get_value())
            epsilon_dot.move_to(point_loc)

        epsilon_dot.add_updater(dot_updater)
        # Make the epsilon balls
        epsilon_balls = make_epsilon_balls(
            current_epsilon.get_value(),
            moon_points,
            axes,
            ball_color=ball_color,
            opacity=ball_opacity,
        )
        # Set up updater for radius of balls
        def epsilon_balls_updater(epsilon_balls):
            for ball in epsilon_balls:
                ball.set_width(current_epsilon.get_value())

        # Turn epsilon up and down
        epsilon_balls.add_updater(epsilon_balls_updater)
        # Fade in the initial balls
        self.play(FadeIn(epsilon_balls), lag_ratio=0.0)
        # Iterate through different values of epsilon
        for value in [1.5, 0.5, 0.9]:
            self.play(current_epsilon.animate.set_value(value), run_time=2.5)
        # Perform clustering
        epsilon_value = 0.9
        # Show connecting graph
        epsilon_graph, adjacency_matrix = make_epsilon_graph(
            current_epsilon.get_value(), dots, moon_points, edge_color=WHITE
        )
        self.play(FadeOut(epsilon_balls))
        self.play(FadeIn(epsilon_graph))
        # Fade out balls
        self.play(Wait(1.5))
        # Perform clustering
        labels = perform_spectral_clustering(adjacency_matrix)
        # Change the colors of the dots
        color_change_animation = make_color_change_animation(labels, dots)
        self.play(color_change_animation)
        # Fade out graph edges
        self.play(FadeOut(epsilon_graph))
        self.play(Wait(5.0))

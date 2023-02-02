from manim import *
from manim_ml.decision_tree.decision_tree import (
    DecisionTreeDiagram,
    DecisionTreeSurface,
    IrisDatasetPlot,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
import sklearn
import matplotlib.pyplot as plt


def learn_iris_decision_tree(iris):
    decision_tree = DecisionTreeClassifier(
        random_state=1, max_depth=3, max_leaf_nodes=6
    )
    decision_tree = decision_tree.fit(iris.data[:, :2], iris.target)
    # output the decisioin tree in some format
    return decision_tree


def make_sklearn_tree(dataset, max_tree_depth=3):
    tree = learn_iris_decision_tree(dataset)
    feature_names = dataset.feature_names[0:2]
    return tree, tree.tree_


class DecisionTreeScene(Scene):
    def construct(self):
        """Makes a decision tree object"""
        iris_dataset = datasets.load_iris()
        clf, sklearn_tree = make_sklearn_tree(iris_dataset)
        # sklearn.tree.plot_tree(clf, node_ids=True)
        # plt.show()
        decision_tree = DecisionTreeDiagram(
            sklearn_tree,
            class_images_paths=[
                "images/iris_dataset/SetosaFlower.jpeg",
                "images/iris_dataset/VeriscolorFlower.jpeg",
                "images/iris_dataset/VirginicaFlower.jpeg",
            ],
            class_names=["Setosa", "Veriscolor", "Virginica"],
            feature_names=["Sepal Length", "Sepal Width"],
        )
        decision_tree.move_to(ORIGIN)
        create_decision_tree = Create(decision_tree, traversal_order="bfs")
        self.play(create_decision_tree)
        # self.play(create_decision_tree)


class SurfacePlot(Scene):
    def construct(self):
        iris_dataset = datasets.load_iris()
        iris_dataset_plot = IrisDatasetPlot(iris_dataset)
        iris_dataset_plot.all_group.scale(1.0)
        iris_dataset_plot.all_group.shift([-3, 0.2, 0])
        self.play(Create(iris_dataset_plot))
        # make the decision tree classifier
        decision_tree_classifier, sklearn_tree = make_sklearn_tree(iris_dataset)
        decision_tree_surface = DecisionTreeSurface(
            decision_tree_classifier, iris_dataset.data, iris_dataset_plot.axes_group[0]
        )
        self.play(Create(decision_tree_surface))
        self.wait(1)

from manim_ml.neural_network.activation_functions.relu import ReLUFunction

name_to_activation_function_map = {"ReLU": ReLUFunction}


def get_activation_function_by_name(name):
    return name_to_activation_function_map[name]

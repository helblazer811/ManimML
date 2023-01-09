def compute_node_depths(tree):
    """Computes the depths of nodes for level order traversal"""

    def depth(node_index, current_node_index=0):
        """Compute the height of a node"""
        if current_node_index == node_index:
            return 0
        elif (
            tree.children_left[current_node_index]
            == tree.children_right[current_node_index]
        ):
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
        elif (
            tree.children_left[current_node_index]
            == tree.children_right[current_node_index]
        ):
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
    node_to_parent = {0: -1}  # Root has no parent
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

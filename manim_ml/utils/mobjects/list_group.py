from manim import *


class ListGroup(Mobject):
    """Indexable Group with traditional list operations"""

    def __init__(self, *layers):
        super().__init__()
        self.items = [*layers]

    def __getitem__(self, indices):
        """Traditional list indexing"""
        return self.items[indices]

    def insert(self, index, item):
        """Inserts item at index"""
        self.items.insert(index, item)
        self.submobjects = self.items

    def remove_at_index(self, index):
        """Removes item at index"""
        if index > len(self.items):
            raise Exception(f"ListGroup index out of range: {index}")
        item = self.items[index]
        del self.items[index]
        self.submobjects = self.items

        return item

    def remove_at_indices(self, indices):
        """Removes items at indices"""
        items = []
        for index in indices:
            item = self.remove_at_index(index)
            items.append(item)

        return items

    def remove(self, item):
        """Removes first instance of item"""
        self.items.remove(item)
        self.submobjects = self.items

        return item

    def get(self, index):
        """Gets item at index"""
        return self.items[index]

    def add(self, item):
        """Adds to end"""
        self.items.append(item)
        self.submobjects = self.items

    def replace(self, index, item):
        """Replaces item at index"""
        self.items[index] = item
        self.submobjects = self.items

    def index_of(self, item):
        """Returns index of item if it exists"""
        for index, obj in enumerate(self.items):
            if item is obj:
                return index
        return -1

    def __len__(self):
        """Length of items"""
        return len(self.items)

    def set_z_index(self, z_index_value, family=True):
        """Sets z index of all values in ListGroup"""
        for item in self.items:
            item.set_z_index(z_index_value, family=True)

    def __iter__(self):
        self.current_index = -1
        return self

    def __next__(self):  # Python 2: def next(self)
        self.current_index += 1
        if self.current_index < len(self.items):
            return self.items[self.current_index]
        raise StopIteration

class Stack:
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        return self.items.pop()

    def update(self, item):
        self.items[len(self.items) - 1] = item
        return self.items

    def peek(self):
        return self.items[len(self.items) - 1]

    def size(self):
        return len(self.items)
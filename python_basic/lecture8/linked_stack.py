class Node(object):
    def __init__(self, value=None, pointer=None):
        self.value = value
        self.pointer = pointer

class Stack(object):
    def __init__(self):
        self.head = None

    def isEmpty(self):
        return not bool(self.head)

    def push(self, item):
        self.head = Node(item, self.head)

    def size(self):
        node = self.head
        count = 0
        while node:
            count +=1
            node = node.pointer
        return count

    def pop(self):
        if self.head:
            node = self.head
            self.head = node.pointer
            return node.value
        else:
            print('Stack is empty.')

    def peek(self):
        if self.head:
            return self.head.value
        else:
            print('Stack is empty.')

    def __repr__(self):
        items = []
        node = self.head
        while node:
            items.append(node.value)
            node = node.pointer
        items.reverse()
        return '{}'.format(items)

if __name__ == '__main__':
    stack = Stack()
    print(stack.isEmpty())
    stack.push(23)
    stack.push(4)
    stack.push(8)
    print("Size: ", stack.size())
    print(stack)
    print("Peek: ", stack.peek())
    print("Pop!  ", stack.pop())
    print(stack)

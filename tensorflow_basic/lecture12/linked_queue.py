class Node(object):
    def __init__(self, value=None, pointer=None):
        self.value = value
        self.pointer = None

class Queue(object):
    def __init__(self):
        self.head = None
        self.tail = None

    def isEmpty(self):
        return not bool(self.head)

    def enqueue(self, value):
        node = Node(value)
        if not self.head:
            self.head = node
            self.tail = node
        else:
            if self.tail:
                self.tail.pointer = node
            self.tail = node

    def size(self):
        node = self.head
        count = 0
        while node:
                count += 1
                node = node.pointer
        return count

    def peek(self):
        return self.head.value

    def __repr__(self):
        items = []
        node = self.head
        while node:
            items.append(node.value)
            node = node.pointer
        items.reverse()
        return '{}'.format(items)

    def dequeue(self):
        if self.head:
            value = self.head.value
            self.head = self.head.pointer
            return value
        else:
            print('Queue is empty')

if __name__ == '__main__':
    queue = Queue()
    print(queue.isEmpty())
    queue.enqueue(23)
    queue.enqueue(4)
    queue.enqueue(8)
    print("Size: ", queue.size())
    print(queue)
    print("Peek: ", queue.peek())
    print("Dequeue!  ", queue.dequeue())
    print(queue)


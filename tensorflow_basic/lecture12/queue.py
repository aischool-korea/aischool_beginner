class Queue(object):
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return not bool(self.items)

    def enqueue(self, value):
        self.items.insert(0, value)

    def size(self):
        return len(self.items)

    def __repr__(self):
        return '{}'.format(self.items)

    def peek(self):
        if self.items:
            return self.items[-1]
        else:
            print('Queue is empty.')

    def dequeue(self):
        value = self.items.pop()
        if value is not None:
            return value
        else:
            return 'Queue is empty'

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
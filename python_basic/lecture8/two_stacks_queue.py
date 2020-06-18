class Queue(object):

    def __init__(self):
        self.in_stack = []
        self.out_stack = []

    def isEmpty(self):
        return not (bool(self.in_stack) or bool(self.out_stack))

    def _transfer(self):
        while self.in_stack:
            self.out_stack.append(self.in_stack.pop())

    def enqueue(self, item):
        return self.in_stack.append(item)

    def size(self):
        return len(self.in_stack) + len(self.out_stack)

    def peek(self):
        if not self.out_stack:
            self._transfer()
        if self.out_stack:
            return self.out_stack[-1]
        else:
            return "Queue empty!"

    def __repr__(self):
        if not self.out_stack:
            self._transfer()
        if self.out_stack:
            return '{}'.format(self.out_stack)
        else:
            return "Queue is empty"

    def dequeue(self):
        if not self.out_stack:
            self._transfer()
        if self.out_stack:
            return self.out_stack.pop()
        else:
            return "Queue is empty"

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
class Queue:
    def __init__(self):
        self.items = []

    def enqueue(self, item):
        self.items.append(item)

    def dequeue(self):
        if not self.is_empty():
            return self.items.pop(0)
        else:
            print("Queue is empty")
            return None

    def peek(self):
        if not self.is_empty():
            return self.items[0]
        else:
            print("Queue is empty")
            return None

    def is_empty(self):
        return len(self.items) == 0

    def size(self):
        return len(self.items)


# Example usage:
q = Queue()

# Enqueue elements
q.enqueue(10)
q.enqueue(20)
q.enqueue(30)

# Dequeue elements
print("Dequeued element:", q.dequeue())
print("Dequeued element:", q.dequeue())

# Peek at the front element
print("Front element:", q.peek())

# Check if the queue is empty
print("Is queue empty?", q.is_empty())

# Get the size of the queue
print("Queue size:", q.size())


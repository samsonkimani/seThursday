class Stack:
  def __init__(self, max_size):
    self.max_size = max_size
    self.arr = [None] * max_size
    self.top = -1

  def is_empty(self):
    return self.top == -1

  def is_full(self):
    return self.top == self.max_size - 1

  def push(self, data):
    if self.is_full():
      raise Exception("Stack Overflow")
    self.top += 1
    self.arr[self.top] = data

  def pop(self):
    if self.is_empty():
      raise Exception("Stack Underflow")
    data = self.arr[self.top]
    self.top -= 1
    return data

  def peek(self):
    if self.is_empty():
      return None
    return self.arr[self.top]

# Example usage
stack = Stack(10)
stack.push(1)
stack.push(2)
print(stack.pop())  # Output: 2
print(stack.peek())  # Output: 1

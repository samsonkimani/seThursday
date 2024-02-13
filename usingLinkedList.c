#include <stdio.h>
#include <stdlib.h>

typedef struct Node {
    int data;
    struct Node *next;
} Node;

typedef struct {
    Node *top;
} Stack;

// Function to initialize an empty stack
void initialize(Stack *stack) {
    stack->top = NULL;
}

// Function to check if the stack is empty
int isEmpty(Stack *stack) {
    return stack->top == NULL;
}

// Function to push an element onto the stack
void push(Stack *stack, int value) {
    Node *newNode = (Node*) malloc(sizeof(Node));
    if (newNode == NULL) {
        printf("Memory allocation failed\n");
        return;
    }
    newNode->data = value;
    newNode->next = stack->top;
    stack->top = newNode;
}

// Function to pop an element from the stack
int pop(Stack *stack) {
    if (isEmpty(stack)) {
        printf("Stack underflow\n");
        return -1;
    }
    Node *temp = stack->top;
    int data = temp->data;
    stack->top = stack->top->next;
    free(temp);
    return data;
}

// Function to return the top element of the stack without removing it
int peek(Stack *stack) {
    if (isEmpty(stack)) {
        printf("Stack is empty\n");
        return -1;
    }
    return stack->top->data;
}

int main() {
    Stack stack;
    initialize(&stack);

    // Pushing elements onto the stack
    push(&stack, 10);
    push(&stack, 20);
    push(&stack, 30);

    // Peeking at the top element
    printf("Top element: %d\n", peek(&stack));

    // Popping elements from the stack
    printf("Popped element: %d\n", pop(&stack));
    printf("Popped element: %d\n", pop(&stack));
    printf("Popped element: %d\n", pop(&stack));

    // Checking if the stack is empty
    if (isEmpty(&stack)) {
        printf("Stack is empty\n");
    } else {
        printf("Stack is not empty\n");
    }

    return 0;
}


---
title: "How can two linked lists be summed?"
date: "2025-01-30"
id: "how-can-two-linked-lists-be-summed"
---
The crux of summing two linked lists, where each node represents a digit in a number, and the lists are structured in reverse order (least significant digit first), lies in performing iterative addition while handling carry-over values. This is not a simple element-wise addition like arrays; it necessitates a traversal of both lists concurrently and the construction of a new list to store the result. My work on embedded systems, where memory efficiency and pointer manipulation are paramount, has frequently required this type of algorithmic thinking.

To illustrate, let's assume each node contains a single integer digit (0-9) and the lists represent numbers with their digits in reverse order. For example, the list `1 -> 2 -> 3` would represent the number 321, and `4 -> 5 -> 6` would be 654. Summing these lists should produce the list `5 -> 7 -> 9`, representing 975. The challenge arises when the digit sum exceeds nine, necessitating a carry-over to the next higher place value.

The algorithm I typically employ involves these steps: I initialize a `carry` variable to 0 and create a dummy head node for the resultant linked list. This dummy node avoids special case handling when adding the first digit. Then, I iterate through both input lists using while loops that continue as long as either list has nodes remaining or there's a carry value. During each iteration, I sum the digits from the corresponding nodes (if they exist) and the current `carry`. I extract the new digit by taking the result modulo 10 and create a new node containing that digit. I append this new node to the result list and update the `carry` by integer dividing the result by 10. The crucial point is to handle the case where one list is shorter than the other, or when the final carry is not zero. Once the loops finish, I return the head of the resultant list which is after the dummy node.

Here are three code examples demonstrating different scenarios, using Python syntax for clarity:

**Example 1: Basic Addition**

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def add_lists(l1, l2):
    dummy_head = ListNode(0)
    current = dummy_head
    carry = 0

    while l1 or l2 or carry:
        sum_val = carry
        if l1:
            sum_val += l1.val
            l1 = l1.next
        if l2:
            sum_val += l2.val
            l2 = l2.next

        carry = sum_val // 10
        current.next = ListNode(sum_val % 10)
        current = current.next

    return dummy_head.next


# Example Usage:
l1 = ListNode(1, ListNode(2, ListNode(3)))  # Represents 321
l2 = ListNode(4, ListNode(5, ListNode(6)))  # Represents 654
result = add_lists(l1, l2)
# Result List: 5 -> 7 -> 9 (Represents 975)

def print_list(head):
    while head:
        print(head.val, end=" -> ")
        head = head.next
    print("None")
print_list(result)
```

This code example demonstrates the core algorithm. `add_lists` function encapsulates the iterative addition with `carry` handling. It covers the scenario where both lists have the same number of digits, thus showcasing the typical case. The print list function is just a utility to show the content of the list after the addition.

**Example 2: Handling Unequal Length Lists**

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def add_lists(l1, l2):
    dummy_head = ListNode(0)
    current = dummy_head
    carry = 0

    while l1 or l2 or carry:
        sum_val = carry
        if l1:
            sum_val += l1.val
            l1 = l1.next
        if l2:
            sum_val += l2.val
            l2 = l2.next

        carry = sum_val // 10
        current.next = ListNode(sum_val % 10)
        current = current.next

    return dummy_head.next

# Example Usage:
l1 = ListNode(9, ListNode(9, ListNode(9)))    # Represents 999
l2 = ListNode(1)                             # Represents 1
result = add_lists(l1, l2)
# Result List: 0 -> 0 -> 0 -> 1 (Represents 1000)

def print_list(head):
    while head:
        print(head.val, end=" -> ")
        head = head.next
    print("None")
print_list(result)
```
This example highlights the algorithm's ability to handle lists of different lengths. Here, `l1` represents 999 and `l2` is 1. The core while loop naturally handles the shorter list by treating the missing nodes as zero without special case considerations. The result should be 1000.

**Example 3: Handling Final Carry**

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def add_lists(l1, l2):
    dummy_head = ListNode(0)
    current = dummy_head
    carry = 0

    while l1 or l2 or carry:
        sum_val = carry
        if l1:
            sum_val += l1.val
            l1 = l1.next
        if l2:
            sum_val += l2.val
            l2 = l2.next

        carry = sum_val // 10
        current.next = ListNode(sum_val % 10)
        current = current.next

    return dummy_head.next

# Example Usage:
l1 = ListNode(5, ListNode(9))  # Represents 95
l2 = ListNode(5)  # Represents 5
result = add_lists(l1, l2)
# Result List: 0 -> 0 -> 1 (Represents 100)


def print_list(head):
    while head:
        print(head.val, end=" -> ")
        head = head.next
    print("None")
print_list(result)
```

This example specifically demonstrates handling a leftover `carry`. When adding 95 and 5, the result should be 100. The critical element here is the `while l1 or l2 or carry` condition. This condition ensures that if there is a non-zero `carry` after processing both lists, it will add that carry value to the result list even when both `l1` and `l2` become empty.

In terms of efficiency, this iterative algorithm has a time complexity of O(max(m, n)), where 'm' and 'n' are the lengths of the two input lists. This linear complexity arises from having to iterate at most the length of the longer list. The space complexity is O(max(m,n)), as in the worst case, the resulting list might have max(m,n) + 1 nodes if there is a final carry. I've found this approach to be robust and effective even in resource-constrained embedded systems environments.

For further exploration and deeper understanding, I recommend studying fundamental data structure texts focusing on linked lists and algorithm analysis. Specific resources that I've found helpful include textbooks on data structures and algorithms that cover iterative list manipulation, and algorithm design principles. In addition to academic texts, exploring resources dedicated to competitive programming problems is beneficial, as such exercises frequently involve linked list manipulation. Practice with these types of exercises enhances practical skill and reinforces understanding of these fundamental concepts. The key is to visualize the list structure, and the steps involved in iterative traversal and node creation.

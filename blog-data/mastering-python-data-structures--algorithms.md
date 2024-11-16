---
title: "Mastering Python Data Structures & Algorithms"
date: "2024-11-16"
id: "mastering-python-data-structures--algorithms"
---

dude so you won't BELIEVE this video i just watched it's like a crash course in how to totally level up your python game specifically with data structures and algorithms but like in the most chill way ever  it's not all stuffy lectures and dry theory it's got this crazy awesome dude explaining things and he's got this really goofy sense of humor so it keeps things super interesting  the whole point was to show how different data structures impact performance like when you're dealing with tons of data because that's where the real magic happens in programming right?

okay so the setup is pretty simple the guy he’s basically this super enthusiastic professor type but way less intimidating  he starts by laying down the groundwork  he's like "yo let's talk about efficiency because nobody wants their code to run slower than a snail on valium" and then BAM he dives into arrays  i mean everyone knows arrays right?  like the most basic way to store data it's all super linear you just have a numbered list of stuff.   he actually showed this really funny visual of like a train going down the tracks to explain how accessing elements in an array is so straightforward because you know exactly where each element is by its index just like the train cars you know exactly the order.  he also made a point of how accessing an element is O(1) which is like super fast constant time regardless of how big the array is  

then things get REALLY interesting he started talking about linked lists  oh man linked lists are a trip dude.  imagine instead of a train you've got these little train cars scattered all over the place each one having a little note saying where the next car is  it’s not just one continuous track it's like a hop-scotch of data so you have to follow the little notes to find the next car which is why searching for specific item is slower O(n) meaning it's proportional to the size of your data.  he even showed a really cool animation of a little cartoon bunny hopping between the cars which was oddly mesmerizing   

one of the KEY moments was when he compared searching through an array versus a linked list for a specific item. with the array it's like a direct shot. you know exactly where to go. but with the linked list?  it's a wild goose chase sometimes even worse than dating apps.  you gotta start at the beginning and hop along until you find what you're looking for. this is where the whole big O notation thing comes in it's a way to talk about how the time it takes to run your code scales with the size of your input data this is really essential for optimization because you want code that doesn't take forever as data increases  think about searching a phonebook with a million names versus only 10 names  array still fast O(1) for access. but linked list is proportionally slower O(n)


another major takeaway was about hash tables  these things are INSANE  it's like having a super-efficient library catalog  you don't have to search through every book one by one. you just look up the title (your key) and bam! you get the location (the value) almost instantly. he made it super relatable by comparing it to looking up words in a dictionary using a hash function  i mean finding a word is much faster than going through each letter and each page.  he said that on average the search time is O(1) making it ridiculously fast and this was further hammered home with a graphic of super fast searches that made me want to use hash tables in every project forever.



okay so then the fun part about trees particularly binary search trees  these are like organizational ninjas  they are structured so that searching inserting and deleting is way faster than in a plain linked list.  he used the analogy of finding a specific card in a deck of sorted cards using binary search  each comparison cuts the search space in half.  like you start in the middle if it's too high you eliminate half the deck and move to the lower half if too low you eliminate the upper half. pretty slick, right? this makes the search average O(log n) much faster than the O(n) of a linked list or the O(n) of unsorted arrays or O(n^2) of some naive sorting algorithms. this is a game changer when your data gets huge. the visual representation was a cool animated tree growing and shrinking as elements were added and removed  super satisfying to watch.


he then explained how to implement a simple binary search tree in python this was probably the most helpful part for me because i was actually trying to wrap my head around the implementation details


```python
class Node:
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None

class BinarySearchTree:
    def __init__(self):
        self.root = None

    def insert(self, data):
        if self.root is None:
            self.root = Node(data)
        else:
            self._insert_recursive(self.root, data)

    def _insert_recursive(self, node, data):
        if data < node.data:
            if node.left is None:
                node.left = Node(data)
            else:
                self._insert_recursive(node.left, data)
        else:
            if node.right is None:
                node.right = Node(data)
            else:
                self._insert_recursive(node.right, data)

    def search(self, data):
        return self._search_recursive(self.root, data)

    def _search_recursive(self, node, data):
        if node is None or node.data == data:
            return node
        if data < node.data:
            return self._search_recursive(node.left, data)
        return self._search_recursive(node.right, data)


# Example usage
bst = BinarySearchTree()
bst.insert(8)
bst.insert(3)
bst.insert(10)
bst.insert(1)
bst.insert(6)
bst.insert(14)
print(bst.search(6).data)  # Output: 6
print(bst.search(7))       # Output: None

```

this code basically shows how to create a binary search tree insert new nodes and search for existing nodes  it’s recursive which is pretty neat  the `insert` method adds new nodes to the tree while maintaining the binary search tree property ensuring that the left subtree contains smaller values and the right subtree contains larger values. the `search` method efficiently finds a node with a given value by traversing the tree, effectively using binary search to achieve logarithmic time complexity.  i'll definitely be using this when i need to implement a fast search function


he also showed some python code for a hash table which was also super useful


```python
class HashTable:
    def __init__(self, capacity):
        self.capacity = capacity
        self.table = [[] for _ in range(capacity)]

    def __setitem__(self, key, value):
        index = self._hash(key) % self.capacity
        for i, (k, v) in enumerate(self.table[index]):
            if k == key:
                self.table[index][i] = (key, value)
                return
        self.table[index].append((key, value))

    def __getitem__(self, key):
        index = self._hash(key) % self.capacity
        for k, v in self.table[index]:
            if k == key:
                return v
        raise KeyError(key)

    def _hash(self, key):
        return hash(key)

#example
ht = HashTable(5)
ht["apple"] = 1
ht["banana"] = 2
ht["cherry"] = 3
print(ht["banana"]) # Output 2
```

this code implements a simple hash table using separate chaining to handle collisions.  the `__setitem__` method inserts key-value pairs  `__getitem__` retrieves values and the `_hash` function computes the hash of the key which determines the index.  it’s pretty basic but it shows the core concepts.   separate chaining handles collisions that could occur when two different keys happen to hash to the same index within the hash table’s array. when a collision happens, instead of overwriting an existing entry, it creates a linked list (or a chain) at that index to store all colliding key-value pairs.



and finally there was this awesome section on graphs  man graphs are everywhere  social networks maps  all that good stuff  he explained different graph representations like adjacency matrices and adjacency lists  he even had this really funny visual of a bunch of people connected by rubber bands representing a social network and how different algorithms are like different ways of moving through that network.  he did mention  dijkstra’s algorithm for finding the shortest path which is a big deal for things like GPS navigation. although he didn't get into the code for that one, it was still useful to get an overview of its use cases.



so the resolution?  the video totally drove home the importance of understanding data structures and algorithms  choosing the right one for the job can mean the difference between code that runs smoothly and code that crawls at a snail's pace.  it also emphasized that learning this stuff doesn't have to be some boring theoretical exercise it can actually be really engaging and fun.  plus i now have a few python snippets i can use in my own projects which is always a win.  so yeah  definitely check it out if you're trying to seriously level up your coding skills  you won’t regret it.

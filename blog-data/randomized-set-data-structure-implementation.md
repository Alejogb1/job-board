---
title: "randomized set data structure implementation?"
date: "2024-12-13"
id: "randomized-set-data-structure-implementation"
---

Okay so you want to talk about randomized set implementations huh Been there done that Let me tell you it's not as straightforward as some might think you'd think it's a simple wrapper around a hashset or something but nah not really There are nuances

My first encounter with this was way back in like probably 2014 or 2015 I was working on this real-time data processing pipeline and we needed a way to deduplicate incoming events without a fixed order of insertion or deletion basically we had a firehose of events and needed to keep track of which ones we had already seen to avoid reprocessing them and the order we processed them did not matter But just a plain set didn't quite cut it because we occasionally had to remove entries randomly to simulate expiring entries without a strict fifo or lifo queue It was a real headache at first

So a pure hashset is excellent for O(1) average lookup insertion and deletion no doubt about that However remove at random from a hashset is a real pain because you can't just say remove an index 5 for example as it is not indexed right You can make a list out of it but then O(n) remove time if you are not at the end so that is a no go

What I figured out and used was a bit of a hybrid approach Its kind of like having the best of both worlds hashset and a dynamic array or list

So here's how it usually works

1 We have a dictionary or hashmap or whatever floats your boat we'll call it the value index mapping this map stores each element and its index in an array that we maintain alongside
2 Then we also have an array or list we'll call this the array or values array This array holds the actual elements itself
3 When you want to insert an element you first check if the element already exists in the value index mapping if yes it is already in the set and do nothing if not you append the element to the end of the array and also add it to the value index mapping with its new array index
4 To remove at random you first generate a random index between 0 and size of the array minus one and then what is done is pretty nifty to make it all O(1) so no looping
5 Get the value in that array at the random index to be removed We then get value of the array at the last index in array this is the one that will take the place of the one we removing and the value at the random index removed will be just overwritten by the last index value and then the value we getting from the last array position value needs to have it's index updated in the value index mapping to be the new position in array it has taken and finally the last index in array is now removed from array using remove last or pop and then remove mapping for the last position
6 For contains simply look it up in the value index mapping
It sounds like a lot but its pretty simple in code

Here's some Python code to illustrate the approach

```python
import random

class RandomizedSet:
    def __init__(self):
        self.value_index_mapping = {}
        self.values_array = []

    def insert(self, val: int) -> bool:
        if val in self.value_index_mapping:
            return False
        self.value_index_mapping[val] = len(self.values_array)
        self.values_array.append(val)
        return True

    def remove(self) -> bool:
        if not self.values_array:
           return False
        random_index = random.randint(0, len(self.values_array) - 1)
        last_index = len(self.values_array) - 1
        if random_index != last_index:
           val_to_remove = self.values_array[random_index]
           last_element = self.values_array[last_index]
           self.values_array[random_index] = last_element
           self.value_index_mapping[last_element] = random_index
        
        removed_value = self.values_array.pop()
        del self.value_index_mapping[removed_value]
        
        return True
    
    def contains(self, val: int) -> bool:
        return val in self.value_index_mapping
```
You can test it by doing some insertions and then some removals and the the contains and all methods will work as expected The time complexity of all methods are average O(1) which is what we want

I remember when i first implemented this i made a stupid mistake where i was updating the mapping incorrectly and it was inserting but not updating mapping at all when removing so it was still returning that it contains but i already removed it I was like what in the world and then i got the debugger out and it turned out it was like one line that made the whole thing not work and after that the random removal worked flawlessly I almost threw my computer out the window when I saw that it was a one-line fix and that line i was missing was the line to update the mapping table I can't tell you how many times that kind of thing has happened It's almost comical at this point You spend hours debugging something and then its like it just needs a single line to be changed or added

Alright here is another example this time using Java it is the same logic but different programming language

```java
import java.util.*;

class RandomizedSet {
    private Map<Integer, Integer> valueIndexMap;
    private List<Integer> valuesList;
    private Random random;

    public RandomizedSet() {
        valueIndexMap = new HashMap<>();
        valuesList = new ArrayList<>();
        random = new Random();
    }

    public boolean insert(int val) {
        if (valueIndexMap.containsKey(val)) {
            return false;
        }
        valueIndexMap.put(val, valuesList.size());
        valuesList.add(val);
        return true;
    }

    public boolean remove() {
        if(valuesList.isEmpty()){
           return false;
        }
       int randomIndex = random.nextInt(valuesList.size());
       int lastIndex = valuesList.size() -1;

        if(randomIndex != lastIndex){
            int valToRemove = valuesList.get(randomIndex);
            int lastElement = valuesList.get(lastIndex);
            valuesList.set(randomIndex, lastElement);
            valueIndexMap.put(lastElement, randomIndex);
        }

        int removedVal = valuesList.remove(valuesList.size()-1);
        valueIndexMap.remove(removedVal);
       
        return true;
    }


    public boolean contains(int val) {
        return valueIndexMap.containsKey(val);
    }
}
```
And to top it all off heres a version in C++

```cpp
#include <iostream>
#include <unordered_map>
#include <vector>
#include <random>
#include <algorithm>


class RandomizedSet {
public:
    RandomizedSet() {}

    bool insert(int val) {
        if (valueIndexMap.count(val)) {
            return false;
        }
        valueIndexMap[val] = valuesList.size();
        valuesList.push_back(val);
        return true;
    }

    bool remove() {
       if (valuesList.empty()){
          return false;
       }

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> distrib(0, valuesList.size() - 1);

        int randomIndex = distrib(gen);
        int lastIndex = valuesList.size() - 1;

         if (randomIndex != lastIndex){
              int valueToRemove = valuesList[randomIndex];
              int lastElement = valuesList[lastIndex];
              valuesList[randomIndex] = lastElement;
              valueIndexMap[lastElement] = randomIndex;
          }


        int removedValue = valuesList.back();
        valuesList.pop_back();
        valueIndexMap.erase(removedValue);
       

        return true;

    }

    bool contains(int val) {
        return valueIndexMap.count(val);
    }

private:
    std::unordered_map<int, int> valueIndexMap;
    std::vector<int> valuesList;
};

```

This implementation in C++ is very similar to the other ones it has the same logic but is using the C++ standard library and you can see the time complexities are O(1) average and it satisfies what we set out to do

Now about resources if you want to dive deeper into the algorithmic aspects of this sort of stuff you might want to check out "Introduction to Algorithms" by Cormen et al Its the bible of algorithms for a reason another good book if you're more into data structures is "Algorithms" by Robert Sedgewick and Kevin Wayne that one has a more practical implementations

As you can see its not rocket science this concept is super useful in a lot of different scenarios and you can always optimize it further depending on the trade-offs you are willing to make but the core idea remains the same you maintain a separate indexing map and array for efficient look up and random access so yeah thats the deal hope it makes sense and if you need any more help just shout

---
title: "How to Visualize Sorting Algorithms"
date: "2024-11-16"
id: "how-to-visualize-sorting-algorithms"
---

okay so you wanna hear about this video i watched it was like a total mind-melt about sorting algorithms right  the whole point was to show how different ways of sorting data work and why some are way faster than others  it wasn't just some dry lecture either oh no it was all done with these super cute little animations of these blocks bouncing around and changing colors it was like watching a really geeky but strangely hypnotic dance-off between algorithms

first off the setup was genius it started with this total mess a bunch of random numbered blocks all jumbled up it was like looking into a toddler's toy box after a hurricane  and then BAM the video started showing how different algorithms would sort these blocks into perfect order  that visual representation was a game changer made it so much easier to grasp the concepts

one of the first algorithms they showed was bubble sort  remember that one from your intro to programming class  it’s so simple it’s almost embarrassing  basically it compares adjacent elements and swaps them if they are in the wrong order  it keeps doing this until the whole list is sorted  they showed this with the blocks literally bubbling up to their correct positions which was kinda hilarious but also really helpful visually  i mean  you could almost *hear* the blocks saying “psst hey buddy you’re out of order”

here’s some python code to illustrate this mess  it's basic but gets the point across:


```python
def bubble_sort(list_):
    n = len(list_)
    for i in range(n-1):
        for j in range(n-i-1):
            if list_[j] > list_[j+1]:
                list_[j], list_[j+1] = list_[j+1], list_[j] #this is the swap – so elegant
    return list_

my_list = [64, 34, 25, 12, 22, 11, 90]
sorted_list = bubble_sort(my_list)
print("Sorted array:", sorted_list)
```

so yeah bubble sort it’s cute it's easy to understand but also super slow especially with a lot of data  the video totally emphasized that point  one part showed bubble sort taking forever to sort like a thousand blocks  it was actually kinda funny to watch it chug along  they even had a little timer running in the corner making it extra dramatic. they used a really cute little sound effect too each swap had a tiny "boing" sound which was surprisingly effective  helped you keep track visually and auditorily  it was like watching a slow motion train wreck  you knew it would eventually get there but you just wanted to scream "hurry up already"

then they jumped into merge sort which was a total opposite a much more sophisticated algorithm that uses a "divide and conquer" approach  it recursively divides the list into smaller sublists until each sublist contains only one element then it repeatedly merges the sublists to produce new sorted sublists until there is only one sorted list remaining. this was shown in a really clever way the video literally split the blocks in half and then showed them being merged back together beautifully.  it was way more efficient  like night and day compared to bubble sort  that part of the video really hammered home the difference between O(n^2) and O(n log n) time complexity  i know it sounds like nerd speak but watching it visually really made it click.


here's some python for merge sort. it’s a bit more involved but bear with me:

```python
def merge_sort(list_):
    if len(list_) > 1:
        mid = len(list_)//2
        left_half = list_[:mid]
        right_half = list_[mid:]

        merge_sort(left_half)
        merge_sort(right_half)

        i = j = k = 0

        while i < len(left_half) and j < len(right_half):
            if left_half[i] < right_half[j]:
                list_[k] = left_half[i]
                i += 1
            else:
                list_[k] = right_half[j]
                j += 1
            k += 1

        while i < len(left_half):
            list_[k] = left_half[i]
            i += 1
            k += 1

        while j < len(right_half):
            list_[k] = right_half[j]
            j += 1
            k += 1
    return list_

my_list = [64, 34, 25, 12, 22, 11, 90]
sorted_list = merge_sort(my_list)
print("Sorted array:", sorted_list)
```

see the difference?  merge sort is way more elegant and efficient  the video did a great job of explaining why the recursive approach makes it so much faster and scalable.  another really cool part they included was a comparison of different sorting algorithms  like quick sort and heap sort  they all got their own little block-sorting show  which was great because it helped show how each one had its own strengths and weaknesses  depending on the size of the data and how it was already organized  it wasn't just about finding the fastest algorithm  it was about understanding the trade-offs


they also touched on the concept of best case worst case and average case scenarios  this is super important stuff for any programmer  it's not just about how fast an algorithm *can* be but also how it performs under different conditions  for example bubble sort has a best case scenario where the list is already sorted which is pretty fast  but its worst case (a reverse sorted list)  is catastrophically slow  they showed these different scenarios visually with different initial arrangements of the blocks which i thought was pretty slick


and finally they showed this thing called quicksort  and let me tell you that was the star of the show  it's a bit more complicated than merge sort but it's usually way faster  it's another divide and conquer algorithm  but it partitions the array around a pivot element  which is chosen randomly or strategically. this whole process is recursive  it keeps doing this until everything is in the right spot.  the visual representation was insane you saw how the pivot block would magically move to its correct position and then the rest would get sorted around it. it was mesmerizing


quick sort python code  this one’s a bit more advanced but don't freak out:


```python
import random

def quick_sort(list_):
    if len(list_) < 2:
        return list_
    else:
        pivot = list_[0] # we're choosing the first element as our pivot for simplicity
        less = [i for i in list_[1:] if i <= pivot]
        greater = [i for i in list_[1:] if i > pivot]
        return quick_sort(less) + [pivot] + quick_sort(greater)


my_list = [64, 34, 25, 12, 22, 11, 90]
sorted_list = quick_sort(my_list)
print("Sorted array:", sorted_list)

```

the resolution of the whole video was basically this  there's no one-size-fits-all solution for sorting  different algorithms have different strengths and weaknesses you need to understand the trade-offs between speed simplicity and scalability  and visualizing these algorithms with those bouncing blocks  was an absolute game-changer it helped clarify those complex concepts  that was why the video was so impactful it made learning about algorithms actually fun which is saying something  it  wasn’t just memorizing formulas it was about genuinely understanding how these things worked underneath the hood and appreciating their elegance  and honestly that’s what made it so memorable  i mean who knew sorting algorithms could be so entertaining right

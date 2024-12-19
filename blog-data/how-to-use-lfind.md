---
title: "how to use lfind?"
date: "2024-12-13"
id: "how-to-use-lfind"
---

Okay so you wanna know about lfind right alright I’ve been there believe me lfind can be a bit of a pain if you're not careful

I remember way back when I was still a junior dev fresh out of university I had this huge project dealing with tons of raw data It was before everyone was obsessed with JSON and we were using these weird custom binary formats and I had to implement some fast search algorithms well not fast like super duper fast but fast enough that it wouldn’t take forever to find some data so the issue came down to basically looking through a massive array of stuff not knowing the actual type or size in advance because that was the genius idea of my older colleague oh man I ended up having to scour through some old C documentation and even some dusty textbooks to figure out how to effectively use lfind that’s when I really started understanding pointer arithmetic too a rite of passage for us low level guys

So let's get down to the gritty details of what lfind is and how it works lfind is this nifty function from the POSIX standard it’s essentially a linear search function It’s designed for situations where you need to find an element within an array of elements but you don’t have a specific type in advance you're dealing with void pointers and a size per element so you have flexibility but with that comes some responsibility of course it's not like Python's "in" operator or Javascript's "includes" its much more low-level

The signature of `lfind` is like this:

```c
void *lfind(const void *key, const void *base, size_t *nelp, size_t width,
            int (*compar)(const void *, const void *));
```

Alright let’s break that down:

*   `const void *key` This is a pointer to the value you’re searching for
*   `const void *base` This is a pointer to the base of the array you’re searching in
*   `size_t *nelp` This is a pointer to a size_t variable which represents the number of elements in your array the trick is that if the element is not found the function will return null and increase the value pointed by `nelp` by 1 at the end this means that after the execution of this function the number of elements in your data structure will have increased because the element was added as a last element
*   `size_t width` This is the size in bytes of each element in the array
*   `int (*compar)(const void *, const void *)` This is a function pointer to a comparison function This function takes two pointers as arguments and should return an integer less than zero if the first element is less than the second zero if they are equal and greater than zero if the first element is greater than the second

It is very important to write this comparison function correctly this is where most of the mistakes happen

So basically what this does is it linearly searches through your array starting from the `base` looking for an element that matches your `key` according to the comparison function specified by `compar` if it finds the element it will return a pointer to that element in the array otherwise it will return NULL and increase `nelp` as mentioned earlier

Now for some code example let’s say you have a bunch of integers and you want to find a specific number and if it is not found the array is increased with the new number you also need to handle the memory allocation for the new element remember you are in the C land now:

```c
#include <stdio.h>
#include <stdlib.h>
#include <search.h>

int compare_ints(const void *a, const void *b) {
    return (*(int *)a - *(int *)b);
}

int main() {
    int numbers[] = {10, 20, 30, 40, 50};
    size_t num_elements = sizeof(numbers) / sizeof(numbers[0]);
    int key = 30;
    int *found;

    found = lfind(&key, numbers, &num_elements, sizeof(int), compare_ints);

    if (found != NULL) {
        printf("Found the element: %d\n", *found);
    } else {
       
        printf("Element not found adding it to the array\n");
         numbers[num_elements++] = key;
        found = &numbers[num_elements - 1];
        printf("The new element was allocated with value: %d\n", *found);
    }

    //printing all the elements
    for(int i =0; i< num_elements; i++) {
        printf("%d ", numbers[i]);
    }
    printf("\n");

    return 0;
}
```

In this example `compare_ints` is the comparison function that compares two integers and `lfind` searches for `key` inside the `numbers` array if the element is found it will return a pointer to the element otherwise it will return `NULL` in this case the new element will be added to the array

Now lets make it more interesting suppose you have a array of structs lets say like user structs you have fields `id` and `name` and you want to find a user with a given id this is pretty common

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <search.h>

typedef struct {
    int id;
    char name[50];
} User;


int compare_users(const void *a, const void *b) {
    return ((User *)a)->id - ((User *)b)->id;
}

int main() {
    User users[] = {
        {1, "Alice"},
        {2, "Bob"},
        {3, "Charlie"},
        {4, "David"}
    };
    size_t num_users = sizeof(users) / sizeof(users[0]);
    User key_user = {3, ""}; //looking for user with id 3 the name does not matter for comparison
    User *found_user;

    found_user = lfind(&key_user, users, &num_users, sizeof(User), compare_users);
    if (found_user != NULL) {
        printf("Found user: ID %d Name: %s\n", found_user->id, found_user->name);
    } else {
        printf("User not found so I am going to allocate it with the new user\n");
        User newUser;
        newUser.id = key_user.id;
        strcpy(newUser.name,"New User");

        users[num_users++] = newUser;
        found_user = &users[num_users -1];
        printf("New user allocated: ID %d Name: %s\n", found_user->id, found_user->name);
    }
    //printing the new user
    for(int i =0; i< num_users; i++) {
        printf("ID %d Name: %s \n", users[i].id, users[i].name);
    }
   

    return 0;
}
```

In this example `compare_users` is used to compare users based on the `id` field and `lfind` will search for the user and return the pointer to the found user. if the user is not found the new user will be appended to the end of the array

Let’s do one more example just to be thorough lets say you are dealing with array of strings:

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <search.h>

int compare_strings(const void *a, const void *b) {
    return strcmp(*(const char **)a, *(const char **)b);
}

int main() {
    char *strings[] = {"apple", "banana", "cherry", "date"};
    size_t num_strings = sizeof(strings) / sizeof(strings[0]);
    char *key = "banana";
    char **found;

    found = lfind(&key, strings, &num_strings, sizeof(char*), compare_strings);

    if (found != NULL) {
        printf("Found the string: %s\n", *found);
    } else {
        printf("String not found so I am adding it to the end\n");
        strings[num_strings++] = key;
       found = &strings[num_strings -1];
         printf("New string added %s \n", *found);
    }
     //printing all the elements
    for(int i =0; i< num_strings; i++) {
         printf("%s ", strings[i]);
     }
     printf("\n");

    return 0;
}
```

In this example `compare_strings` uses `strcmp` to compare two C strings and `lfind` will search for the string inside the `strings` array

A little tip here because `lfind` returns a `void*` pointer you always have to cast it to the right pointer type so that you can use it correctly and of course pay attention to memory because if you are adding elements to the array it is your responsibility to allocate that memory also the `lfind` function can add complexity to your code due to its low-level nature for large datasets or repeated searches something more efficient like a hash table or a tree data structure will be more optimal but I am pretty sure you were already aware of that

There is no magical solution and sometimes a simple solution using a linear search can be very useful and easy to implement I was dealing with a project where it had to be done in a quick and dirty manner for a proof of concept prototype so that solution was suitable for that context The same applies here the best solution depends on the problem you are dealing with

The big O complexity of `lfind` is O(n) because it is a linear search meaning in the worst case scenario it has to check every single element in the array to find the element this is important to know when dealing with performance issues especially when your data grows

In terms of learning more about these low-level stuff I would recommend some classic like "Computer Systems A Programmers Perspective" by Randal E Bryant and David R O'Hallaron for understanding more about how memory and pointers works it's a gold standard book you should have it in your bookshelf I have used this book in many projects and helped me a lot.

Also, “The C Programming Language” by Brian Kernighan and Dennis Ritchie also known as K&R is a good book too it is very detailed about the C language itself and a good reference to understand how everything works in C that's why I am old enough that I know `lfind` was used a lot before we had fancy collections in modern languages like python or javascript we had to deal with these low level functions and it's a good thing for every developer to be familiar with this kind of stuff so that we understand how the machines work

Oh I nearly forgot once I was debugging a piece of code that used `lfind` and I spent like 2 hours trying to figure out why it was always returning `NULL` turns out I forgot to include the `string.h` header for `strcmp` the code compiled but obviously the comparison wasn't working correctly it was one of those silly mistakes you would not believe but well they happen to the best of us

Anyway that's pretty much what I can tell you about `lfind` it's not super complex but it's important to understand how it works especially if you’re dealing with raw memory and low level programming and of course always double check your comparison function and remember to cast the return pointer to the right type I hope this long explanation was helpful to you happy coding

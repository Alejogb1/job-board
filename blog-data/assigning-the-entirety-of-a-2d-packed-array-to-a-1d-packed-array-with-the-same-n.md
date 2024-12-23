---
title: "assigning the entirety of a 2d packed array to a 1d packed array with the same n?"
date: "2024-12-13"
id: "assigning-the-entirety-of-a-2d-packed-array-to-a-1d-packed-array-with-the-same-n"
---

 I've seen this one before a classic actually so you've got a 2D packed array and you want to flatten it into a 1D packed array same number of elements makes sense right I've been there I remember back in the day when I was first playing around with image processing we'd do this all the time before libraries handled it for us so yeah no worries I've got you

So the core problem is how to map the indices from a 2D structure to a 1D structure in memory Packed arrays as you know are just contiguous blocks so the address calculations are relatively straightforward We just need to walk through the rows and columns of the 2D array and place the values sequentially in the 1D array

Let's break it down the naive approach the one I probably used back in the day before learning better I just used nested loops to iterate through the 2D array and assign values to the corresponding location in the 1D array This worked but it's not exactly the most elegant or performant especially when you start dealing with large arrays I remember debugging a very slow matrix multiplication once and this was partially to blame I swear

Here's a basic C implementation showing that

```c
void flatten_2d_to_1d_naive(int** src, int* dest, int rows, int cols) {
    int k = 0;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            dest[k++] = src[i][j];
        }
    }
}
```

So you can see here the nested loops walk through the `src` 2D array `src[i][j]` and fill them in order of row in the 1D destination array `dest[k]`. The index `k` is incremented each time. Pretty basic stuff right? But while it works and can be easy to read it is not optimized for cache locality and memory access patterns for large data sets That's why I ended up doing it in a more sophisticated way

A more efficient way especially if you're working with a language that supports it is to calculate the index directly This avoids the inner loop and can be faster since it reduces the instruction count by eliminating the extra loop check the increment and decrement operations associated with it

Let's look at that same function using pointer arithmetic

```c
void flatten_2d_to_1d_ptr(int** src, int* dest, int rows, int cols) {
    int* p_dest = dest;
    for (int i = 0; i < rows; i++) {
        for(int j=0; j < cols; j++){
           *p_dest++ = src[i][j];
        }
    }
}
```

See? We're just moving a pointer which is often faster and compiler can make better optimization since it knows exactly what we're doing. The idea is that when we move along each row of the 2d array by incrementing the column index `j`, in the destination 1D array that will correspond to incrementing the pointer at the same rate and when moving from one row to another we can continue using the same pointer. It does require we know where the destination pointer is and the address calculation to fill all the slots in the 1D array. That being said this is still slower than copying data directly via memory copy operations. But it helps highlight the basic mechanism we use to make this mapping

There are ways to be even faster for instance using standard library functions like `memcpy` in C. However you need to cast your 2d array as a pointer array to be able to do that so let's show that as a final example.

```c
#include <string.h>
#include <stdlib.h>

void flatten_2d_to_1d_memcpy(int** src, int* dest, int rows, int cols) {
    size_t row_size = sizeof(int) * cols;
    for (int i = 0; i < rows; i++) {
        memcpy(dest + (i * cols), src[i], row_size);
    }
}

int main()
{
    int rows = 3;
    int cols = 4;

    int** src_arr = (int**)malloc(rows * sizeof(int*));

    for(int i = 0; i < rows; ++i){
      src_arr[i] = (int*)malloc(cols * sizeof(int));
        for(int j = 0; j < cols; j++)
            src_arr[i][j] = (i + 1)*10 + (j + 1);
    }

    int* dest_arr = (int*)malloc(rows * cols * sizeof(int));

   flatten_2d_to_1d_memcpy(src_arr,dest_arr,rows,cols);

     for (int i = 0; i < rows * cols; i++) {
       printf("%d ",dest_arr[i]);
     }
    printf("\n");
    //Clean Memory
      for (int i = 0; i < rows; i++) {
        free(src_arr[i]);
    }
    free(src_arr);
    free(dest_arr);

  return 0;

}
```

Here's where it gets interesting The `memcpy` function is optimized for copying blocks of memory its often hardware accelerated and faster than the explicit loop-based approach It copies a row of your source matrix to the destination address calculated using the row and column size we pass. There's less overhead with this approach.

Now a word of warning don't just blindly use `memcpy` without thinking about data layout if your array is not stored in a contiguous way in memory or your architecture uses other memory order this won't work

Also you need to check memory allocation as a best practice but that was not the main point of this explanation. I also included a `main` function for easy copy paste if you just want to test.

Speaking of memory there are potential pitfalls you might encounter If you are using dynamic memory allocation remember to free the memory properly to avoid leaks You would not be the first programmer to leak memory if you do. It is kind of a right of passage in programming actually

Also watch out for bounds checking especially if you're mixing languages or using low-level libraries its easy to write past the end of arrays. Been there done that. And of course overflow issues are the classical way to introduce vulnerabilities you really have to be careful with this.

Now here is the one joke I was supposed to make. Why was the array always invited to the party? Because it knew how to keep everything in order and everyone wanted to see it get flattened on the floor. Ok ok sorry I'll get back to it.

Performance optimization depends on the use case If you're working with small arrays the difference between these methods will be negligible but if your arrays are massive it will be considerable. There's a whole area of work dedicated to optimize cache access and making use of your hardware architecture so I suggest you learn from that.

I should probably give you some resources So for understanding the low-level details of memory layout and pointer arithmetic I recommend checking out "Computer Systems A Programmer's Perspective" by Randal Bryant and David O'Hallaron great book I learned a lot from it. It will give you insights on how data is actually arranged in memory and the mechanics involved. There are some chapters discussing memory alignment and access patterns which you should review.

For a more in-depth look at memory optimization techniques I suggest checking the book "Modern Operating Systems" by Andrew S. Tanenbaum. While primarily about OS concepts it includes a significant discussion on memory management including virtual memory caching strategies and how it impacts performance. These two books are great for starting your journey into that rabbit hole.

If you are using languages like Python or other higher level languages it will have its own way of doing things but these basic principles about mapping indices still apply. You just don't need to implement them by hand. Understanding these basic concepts will help you understand how your high-level languages are doing it under the hood though. In Python for example Numpy does a great job of handling these details but its core implementations rely on libraries similar to the C code we saw.

And that's about it for now Let me know if you have any more questions or if something is not clear always happy to help. Good luck with your code and don't be afraid to dive into the details of how these things work it's a really satisfying journey.

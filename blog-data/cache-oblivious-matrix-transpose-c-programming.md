---
title: "cache oblivious matrix transpose c programming?"
date: "2024-12-13"
id: "cache-oblivious-matrix-transpose-c-programming"
---

 so cache oblivious matrix transpose in C right I've been down that rabbit hole before let me tell you it's a classic problem that makes you appreciate good memory access patterns it's not just about transposing a matrix it's about doing it efficiently without knowing the specifics of the cache which is the whole cache oblivious part

I remember back in the day when I was still working on my undergrad thesis a professor threw this at me as a challenge "transpose this matrix so that it sings to your CPU" he said with a cryptic grin I was like " sure dude let's see what we can do" naive college me I tried all sorts of things the standard row-major swaps the column-major approaches and yeah they all worked in the sense they transposed the matrix but when you scaled up with bigger matrices it was like watching a turtle race a jet I mean cache misses were everywhere performance was dreadful

First off what does cache oblivious even mean it means our algorithm shouldn't rely on specific cache sizes or line sizes or even associativity we write code that works well regardless of those hardware details if you're like most folks starting you're used to thinking in terms of locality of reference if data you access is in the cache good performance if not well you got a slow trip to RAM cache oblivious algorithms are about maximizing that locality without explicitly tuning for it

The trick to cache oblivious matrix transpose is usually divide and conquer its recursively breaking down the matrix into smaller submatrices and then doing the transpose on the submatrices we keep doing this until the submatrices fit entirely into the cache

Here's how a naive recursive approach might look in C

```c
void transpose_recursive_naive(int *A, int *B, int n, int row_start, int col_start) {
    if (n == 1) {
        B[col_start] = A[row_start];
        return;
    }

    int half = n / 2;
    transpose_recursive_naive(A, B, half, row_start, col_start);
    transpose_recursive_naive(A, B, half, row_start, col_start + half);
    transpose_recursive_naive(A, B, half, row_start + half, col_start);
    transpose_recursive_naive(A, B, half, row_start + half, col_start + half);
}

void matrix_transpose_naive(int *A, int *B, int n) {
    transpose_recursive_naive(A, B, n, 0, 0);
}
```

Now this seems elegant right very compact and simple but here's the problem this code while recursive has a huge flaw it doesn't have the property of being cache oblivious because it isn't doing an in place transpose it's making use of another matrix B which doesn't contribute at all to the optimization goals of cache oblivious algorithms. We are constantly writing to a new location instead of exploiting the cache and the spatial locality of data. So as you can expect this doesn't perform particularly well due to the constant writes to matrix B

Let me show you the approach we used back in my thesis years a good way to tackle this problem is with a block recursive approach we break the matrix into blocks that are roughly the size of your cache or less then we recursively transpose those blocks you'll see what i mean

```c
void transpose_block_recursive(int *A, int n, int row_start, int col_start) {
  if (n <= 32) { // Base case for block size
     for(int i = row_start; i < row_start + n; i++){
        for(int j = col_start; j < col_start + n; j++){
          if (i < j) {
             int temp = A[i * n + j];
             A[i * n + j] = A[j * n + i];
             A[j * n + i] = temp;
           }
        }
     }
      return;
  }

  int half = n / 2;
  transpose_block_recursive(A, half, row_start, col_start);
  transpose_block_recursive(A, half, row_start, col_start + half);
  transpose_block_recursive(A, half, row_start + half, col_start);
  transpose_block_recursive(A, half, row_start + half, col_start + half);
  for(int i = row_start; i < row_start + half; i++){
        for(int j = col_start + half; j < col_start + n; j++){
             int temp = A[i * n + j];
             A[i * n + j] = A[j * n + i];
             A[j * n + i] = temp;
          }
      }
}

void matrix_transpose_block(int *A, int n) {
  transpose_block_recursive(A, n, 0, 0);
}
```

Now see the magic here we're transposing in place which means we reuse the same memory locations we're dividing and conquering until a block fits into the cache hopefully if you choose a nice base case say 32 or 64 which is about right for most L1 cache sizes we do a trivial transpose for that small submatrix inside the block recursive function which helps us avoid excess function calls which would impact performance this approach maximizes cache hits and minimizes cache misses when accessing our matrix A.

The key is that block size 32 is a tunable parameter you can test it out and adjust to your specific processor architecture or machine hardware

And if you were thinking "hey wouldn't it be nice if we could avoid recursive calls" sure you are right in the general case because we are using function calls there would be a cost in that process if you have a really big matrix, well it can happen we can unroll that recursion using iterations

```c
void matrix_transpose_iterative(int *A, int n) {
  int blockSize = 32; // Base case block size, can be tuned

  for (int rowBlock = 0; rowBlock < n; rowBlock += blockSize) {
    for (int colBlock = 0; colBlock < n; colBlock += blockSize) {
        for(int i = rowBlock; i < (rowBlock + blockSize > n ? n : rowBlock + blockSize) ; i++){
          for (int j = colBlock; j < (colBlock + blockSize > n ? n: colBlock+blockSize); j++) {
           if(i < j){
             int temp = A[i * n + j];
             A[i * n + j] = A[j * n + i];
             A[j * n + i] = temp;
           }
          }
        }
    }
  }
}
```

In that last code we're basically mimicking the behavior of recursive calls with nested loops we're just breaking the matrix into blocks and transposing each block and here we also avoided recursive calls and we iterate through the blocks and perform small transpose operation on each one. This iterative approach would be way better at handling large matrices and can also have a performance benefit over recursive functions for large datasets.

 now if you want to really get under the hood you should totally check out "Cache-Oblivious Algorithms" by Harald Prokop from the MIT Laboratory for Computer Science it's a great resource for theoretical foundations then dive into "Introduction to Algorithms" by Cormen et al it has really good coverage on divide and conquer concepts and they do touch cache oblivious strategies as well.

And look the thing is not all code is perfect some might have some bugs in them like the time I tried to implement this and reversed the row and col indexes for like two days straight before figuring out I’m an idiot and switched them back. So always test always verify your code before claiming it solves the world's problems.

In the end you should be experimenting with different block sizes and really measure the performance of your transpose algorithms on your specific hardware that's how you learn best. Good luck and happy coding

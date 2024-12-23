---
title: "systemverilog foreach syntax for looping through lower dimension of multidimensi?"
date: "2024-12-13"
id: "systemverilog-foreach-syntax-for-looping-through-lower-dimension-of-multidimensi"
---

so you’re asking about `foreach` loops in SystemVerilog specifically how they work with multidimensional arrays and how to target the lower dimensions right I get it been there done that let’s dive in I've spent way too many late nights debugging these things

Look SystemVerilog `foreach` is pretty powerful for array traversal but it can trip you up if you’re not careful with multidimensional arrays The key thing to remember is how it handles the indices by default it iterates over the entire array structure that includes all dimensions

So if you declare a two dimensional array like say `int my_array [3][4]` it's a 3x4 array `foreach (my_array[i][j])` would indeed give you access to each element with `i` going from 0 to 2 and `j` going from 0 to 3 perfectly as you would expect

But what if you want to loop through all the 1D slices I mean not all the individual elements but each one dimensional lower dimension slice right that’s when things can get a little tricky

 here's the most direct answer and I'll illustrate with a few code snippets that I myself have actually used for similar cases in my previous projects So imagine that you wanted to sum all the values on your array row by row here is the most direct approach I usually take

```systemverilog
module foreach_example;

  int my_array [3][4] = '{
    '{1,2,3,4},
    '{5,6,7,8},
    '{9,10,11,12}
  };

  int sum;

  initial begin

    // Loop through rows (1st dimension slices)
    foreach(my_array[i]) begin
      sum = 0; // Reset sum for each row
      $display("Row index %0d", i);
      foreach(my_array[i][j]) begin
        sum += my_array[i][j];
        $display("  Element [%0d][%0d] = %0d", i, j, my_array[i][j]);
      end
      $display("Row %0d Sum = %0d", i, sum);
    end


    $display("---------------------");
    //Alternative more compact version
    foreach(my_array[i])
        $display("Row %0d values are %p", i, my_array[i]);
    end

endmodule
```

In that example the outer loop `foreach(my_array[i])` iterates through the 'rows' that is through the first dimension This is because when you write `my_array[i]` you’re directly accessing a 1D slice of the array the elements are my_array[0] my_array[1] and my_array[2] each of which is a 1 dimensional integer array of 4 elements

Then the inner loop `foreach(my_array[i][j])` accesses the individual elements within each row using a second index `j` and sum its values to a variable named `sum`. This is the standard double loop you will see everywere

Note that I added a second example that actually prints the entire row at once instead of iterating through all elements one by one this can be done by adding a %p format on the $display statement that allows the tool to print the row array all at once

Now let’s say for a moment you actually needed to sum all the values of the columns instead of the rows here is how you would do it

```systemverilog
module foreach_column_example;

  int my_array [3][4] = '{
    '{1,2,3,4},
    '{5,6,7,8},
    '{9,10,11,12}
  };

  int sum;

  initial begin
    for(int j = 0; j < 4 ; j++) begin
      sum = 0; // Reset sum for each column
      $display("Column index %0d", j);
      foreach(my_array[i]) begin
          sum += my_array[i][j];
          $display("  Element [%0d][%0d] = %0d", i, j, my_array[i][j]);
      end
      $display("Column %0d Sum = %0d", j, sum);
    end

  end

endmodule
```

I added a non `foreach` for loop for the first dimension in order to acess the columns of the multidimensional array the `foreach` now is iterating inside the first dimension but with `j` index fixed on a given column this has the same effect as iterating through the columns

A common mistake that beginners make or at least that I used to make when I was starting with SystemVerilog is trying something like `foreach(my_array[][j])` to access the column or `foreach(my_array[i][])` to acess the rows without any index specified or trying to swap the indexes and it won't work that way as you might have figured by now SystemVerilog's `foreach` syntax expects the index to be explicitly named when accessing a slice or an entire dimension and it also needs the explicit mention of the dimension itself

Now I’m gonna tell you about something I did in the past that made me learn this the hard way and is related to my previous company it was an image processing filter for a very high speed image sensor So I had a 2D array representing a frame that came from a camera and each element of the array was a pixel And I needed to apply a filter that would operate on each row separately right so I thought it would be simple I had done similar things in C++ but the language semantics are quite different

So I started using a single `foreach` like I saw in some early examples I found in the web and I was accessing the pixel data the wrong way and the image was getting completely corrupted the filter wasn’t being applied correctly the pixel data was being manipulated with other adjacent pixel's data which caused all sorts of weird effects on the screen at that moment I started realizing that the multidimensional arrays were more complex than I initially thought

I was scratching my head like a monkey trying to open a coconut that night I remember I was so frustrated I almost threw the whole computer out of the window at the end I had a chat with my coworker that knew way more SystemVerilog than me and he mentioned this little trick using `foreach` to access the 1 dimensional slices of the multidimensional array

After that night I had a big break through and I started using the double foreach as mentioned earlier and everything started to make sense and the filter worked exactly as expected that was one of the moments I’m really proud of because I solved a big bug in a very small amount of time once I understood what I was doing wrong

 one last example let’s suppose you have a 3D array and you want to go through each 2D slice which is quite a complex operation this code is more for the sake of showing how the syntax is extended to higher dimensions it is very similar to the first one I showed you earlier but it is using one extra level of nesting

```systemverilog
module foreach_3d_example;

  int my_3d_array [2][3][4];

  initial begin
    // initialize 3d array
    int count = 1;
    foreach(my_3d_array[i][j][k])
      my_3d_array[i][j][k] = count++;


    // Loop through 2D slices
    foreach (my_3d_array[i]) begin
       $display("Slice Index i = %0d", i);
       foreach (my_3d_array[i][j]) begin
        $display("    Row Index j = %0d  Values = %p", j, my_3d_array[i][j]);
       end
    end
  end
endmodule
```

Here the outer `foreach (my_3d_array[i])` gives you access to each 2D slice Then the inner one `foreach (my_3d_array[i][j])` loops through each row on the 2D slice at index `i` the innermost access `my_3d_array[i][j][k]` gives you access to each element inside that row

So remember it is key to think of these `foreach` statements as accessing the multidimensional array slice by slice not the whole array at once It’s like having several drawers inside a cabinet and you are opening the cabinet and then the drawers one by one

Now here's a dad joke to lighten the mood why was the computer cold because it left its windows open get it haha anyway enough of that

If you are looking for more resources about SystemVerilog arrays I would suggest that you take a look at the "SystemVerilog for Verification" book by Chris Spear and also the "IEEE Standard for SystemVerilog 1800-2017" is the perfect place to double check this syntax on the original document from the people who wrote the standard itself

These resources go into far more detail than I can cover here but I hope this helps you navigate the somewhat mysterious world of `foreach` loops and multidimensional arrays It's really about practice and understanding the indexing behavior and how exactly is that SystemVerilog's foreach actually works internally

Let me know if you have any more questions I’m always here to help a fellow programmer

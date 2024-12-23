---
title: "$fgets in verilog file reading?"
date: "2024-12-13"
id: "fgets-in-verilog-file-reading"
---

 so file reading in Verilog using `$fgets` yeah I've been there done that got the t-shirt and probably a few headaches along the way let's break it down because it can get a little wonky if you're not careful. This isn't like Python where file handling is practically a bedtime story. Verilog is hardware description language remember so we’re dealing with time steps and logic simulation not really designed for complex file I/O.

First off `$fgets` this is a system task and it's pretty much your main tool for pulling data out of a file within a Verilog simulation. Think of it like a read function but very very basic. Its syntax is like this:

```verilog
integer file_handle;
string line_buffer;
$fgets(line_buffer, file_handle);
```

You gotta declare `file_handle` as an integer that's how you reference the open file and you got `line_buffer` that's a string which is where the line read from the file is placed. The `$fgets` task will return 0 if it encounters end of the file otherwise a nonzero value.

My first real go at using it was in a project where I needed to load a huge amount of test data for some custom DSP core I had designed back when i was in college. I had this CSV with thousands of data points and wasn't about to manually create input vectors for each and every single one. It was tedious i can tell you that much. I initially tried to use `fread` but quickly found out that it’s not really great for parsing text based file specially when your text has commas in it.

Now some key points to keep in mind. The file needs to be open before you try to read from it. That's done using `$fopen` and when you are done remember to close it using `$fclose`. Failing to do so you may face funny behavior from your simulator. You might think your code is fine but simulator might be doing weird things like freezing up or just misreading your file.

So a typical workflow goes like this open the file read line by line process the data close the file. You will typically see this setup with the help of `while` loop like in the example below:

```verilog
module file_reading_example;
  integer file_handle;
  string line_buffer;
  reg [7:0] data_array [0:1023]; // Example array to store data
  integer index;

  initial begin
    file_handle = $fopen("data.txt", "r");
    if (file_handle == 0) begin
      $display("Error: Could not open file!");
      $finish;
    end

    index = 0;
    while (!$feof(file_handle) && index < 1024) begin
      $fgets(line_buffer, file_handle);
      if ($sscanf(line_buffer, "%h", data_array[index]) == 1)
        index = index + 1;
      else
          $display("Warning: Invalid line format");
    end
    $fclose(file_handle);
	$display("Data loaded successfully");

    //Display data for verification
	for (integer i = 0; i < index; i++)
		$display("Data array index %0d is %h", i, data_array[i]);

    $finish;
  end

endmodule
```

In the snippet above the code opens a file named `data.txt` in read mode `r`. Then it loops until the end of file `$feof(file_handle)` is reached or until it has stored 1024 data values into an array `data_array` and in the process use `$sscanf` to parse each line into a hex number. If the line isn't a number it gives a warning. This is pretty common to see a basic approach to read file in Verilog and the code will print the values read after the loop to the simulator console.

Now lets talk about `$sscanf`. You see it used in the example above. This is basically like a `scanf` for strings. It's super handy for breaking down a line into different variables. In this code I am only using the `%h` format specifier which reads in hex values but you could do a lot more with this if your file contains different types of data. You can read integer and strings using `%d` and `%s` respectively. It's good to test your `$sscanf` by using $display inside the loop to verify the format of your input file. When I did that DSP project sometimes I had commas and sometimes I just had spaces it was a mess. So I had to go back and fix my data file many times.

So with these details out of the way and now if you are wondering how to handle large data files well honestly Verilog isn’t really designed for that kind of thing. It’s a hardware description language so it's mainly used to simulate hardware. When simulating large amounts of data reading files can be very very slow because your simulator has to perform file operations in addition to running the simulation. For huge files and high-performance simulation it's better to explore other tools. Sometimes using testbenches that generate the data or utilizing a pre-processor to create a memory initialization file or a very large array can be better. But for small to medium size datasets and quick prototyping `$fgets` works .

One gotcha I ran into early on was that `$fgets` doesn't handle newlines the way you think it does. If you are on Linux or Mac it's a line feed `\n` character. Windows is a carriage return and line feed `\r\n`. `$fgets` will just store the newlines into the buffer it is not going to eat it up. So sometimes when parsing or printing the data this becomes an issue and needs to be explicitly handled. When I was doing a cross platform project the different new lines format caused some headaches when the data files were generated on one platform and simulated in another. The simulator in my computer was having a terrible time and was complaining. So I had to deal with that manually by either writing a small script to process the data files or adding a couple of lines of code in my Verilog module.

Here is an example of how to remove those carriage returns `\r`. You need to do this manually if you intend to parse your file when your file comes from a Windows environment and are doing the simulation on Linux for instance:

```verilog
module file_reading_example_no_cr;
  integer file_handle;
  string line_buffer;
  string clean_buffer;
  reg [7:0] data_array [0:1023];
  integer index;

  initial begin
    file_handle = $fopen("data_windows.txt", "r");
    if (file_handle == 0) begin
      $display("Error: Could not open file!");
      $finish;
    end

    index = 0;
    while (!$feof(file_handle) && index < 1024) begin
      $fgets(line_buffer, file_handle);

      //Remove carriage return if present
      clean_buffer = "";
      for (integer i = 0; i < line_buffer.len(); i++)
          if (line_buffer[i] != "\r")
             clean_buffer = {clean_buffer, line_buffer[i]};


      if ($sscanf(clean_buffer, "%h", data_array[index]) == 1)
        index = index + 1;
      else
        $display("Warning: Invalid line format");
    end
    $fclose(file_handle);
    $display("Data loaded successfully");

    //Display data for verification
    for (integer i = 0; i < index; i++)
        $display("Data array index %0d is %h", i, data_array[i]);

    $finish;
  end

endmodule
```

In this snippet the code iterates over the line read using the `$fgets` function and adds character one by one to the `clean_buffer` string without the carriage return character. Notice the use of `.len()` to get the length of the string. This version of the example will work whether your file contains carriage returns and or line feeds.

A good place to start if you want more detail would be to go over your Verilog simulator's manual or you can search and read papers on verification and simulation techniques. Specifically, look for sections that talk about file I/O and text processing. There are a few good textbooks on Verilog as well check out the one by Samir Palnitkar its good for basic understanding of the language. Now go break some code... or maybe just read from some files and that was a joke just for you I know its not funny I'm a hardware engineer my jokes aren't great.

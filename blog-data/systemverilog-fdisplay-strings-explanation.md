---
title: "systemverilog fdisplay strings explanation?"
date: "2024-12-13"
id: "systemverilog-fdisplay-strings-explanation"
---

 so you're asking about `$fdisplay` in SystemVerilog specifically when it's used with strings right got it. Been there done that messed it up a few times too. So basically `$fdisplay` is your go-to for writing formatted output to a file it's not just about strings but let's focus on that aspect since that's what you're asking. It's a system task a built-in thing in the language and it's super handy for debugging and logging info about your hardware designs.

Now the thing with strings and `$fdisplay` is this it uses format specifiers kind of like `printf` in C or similar languages. When you pass a string argument without any format specifier it'll just print the string as is. But things get interesting when you want to do more than just that lets say you want to display string variable that comes with the module I mean you have to or you will never get to debug in a reasonable timeframe if it involves strings right?

Think about this scenario I had this huge verification environment right back in my first job and I was tracing packets going through a complex network-on-chip. I used `$fdisplay` a lot but initially I was just dumb and printing raw data and then I noticed it was a nightmare to make sense of anything. So I started using strings to label everything in my log files.

Let's get to the code shall we? Here is an example if you want to just print a static string:

```systemverilog
module testbench;

  initial begin
    integer file_handle;
    file_handle = $fopen("my_log.txt", "w");
    if (file_handle) begin
        $fdisplay(file_handle, "This is a static string");
        $fclose(file_handle);
    end
  end

endmodule
```

See that `This is a static string` bit well that is what you will see in `my_log.txt` after running it no tricks no surprises. The `w` option means write which overwrites any existing file if you want to append use `a`. Remember that file handling is essential in systemverilog you must close the files after you are done writing because if you dont you will waste a lot of time troubleshooting bugs of this nature and I speak from experience.

Now lets talk dynamic strings with variables it gets a little more interesting. Lets say you have a string variable that you want to print. You will probably think the same as I thought when I started that you could just simply do this:

```systemverilog
module testbench;

  string my_string;

  initial begin
    integer file_handle;
    my_string = "Hello dynamic world";
    file_handle = $fopen("my_log.txt", "w");
    if (file_handle) begin
       $fdisplay(file_handle, my_string);
       $fclose(file_handle);
    end
  end

endmodule
```

And if you run this you will see in `my_log.txt` the string "Hello dynamic world". Nothing earth shattering right? But things get spicy as I said when we try to mix string variables with other data types in the same print statement.

And this is where most people stumble. You can't just mash together different types of variables in a `$fdisplay` statement without format specifiers. You will need those. Think of format specifiers as placeholders for the data type you're printing. For example `%s` is used for string or `%h` is for hexadecimal `%d` for decimal and so on. Lets show that in practice:

```systemverilog
module testbench;

  string packet_type;
  int packet_id;

  initial begin
    integer file_handle;
    packet_type = "Data";
    packet_id = 42;
    file_handle = $fopen("my_log.txt", "w");
    if (file_handle) begin
      $fdisplay(file_handle, "Packet type: %s, ID: %d", packet_type, packet_id);
      $fclose(file_handle);
    end
  end

endmodule
```

Notice the `"Packet type: %s, ID: %d"` part? The `%s` gets replaced by the string in `packet_type` and `%d` by the decimal value in `packet_id`. If you don't use the format specifiers it will interpret that as a literal string if you pass a variable and will output garbage. I messed up on this more than once thinking it would somehow magically understand what I wanted. Well it doesn't it is not a mind reader!

Also remember to check you have the correct amount of arguments and format specifiers. If your string has two `%s` you need two strings to substitute and you can't print an integer when using a format specifier meant for string. If not you will not only get warnings but also inconsistent behavior in your log files. Trust me I spent half a day trying to find that one bug. It turned out the `%s` was supposed to be `%d` and the variable was not a string but an integer. My debugging skills went up 10 points that day. Now some people might complain that they need to use `%s` to print a string what if they wanted a literal `%s` ? Well just use `%%` it should be common enough across programming languages right?

And a final tip when dealing with long strings I mean those that you construct with concatenation using curly braces like this `{string1, string2}` you are going to want to be extra careful. You will need to allocate memory dynamically with `$sformatf` and then pass it to `$fdisplay`. I don't have a code example for that right now but you should look into it if you deal with string concatenation. This will avoid some common pitfalls I have seen people do.

Now some people think SystemVerilog is hard but if you know the basic tricks it's not that different than other languages. It's just a little less forgiving especially when it comes to strings but hey that’s part of the fun right? I find myself doing some very weird bugs with printing and format specifiers all the time. Last time I tried to print 1 string with the wrong specifier it printed 0x00000000 and I spent way to much time debugging that issue. It is funny how those simple mistakes can eat up all of your time.

Now for resources you want the IEEE 1800-2017 standard document for SystemVerilog. Yes it is a hefty document but it's where the truth lies. Don't bother with blog posts too much you will waste a lot of time since most of the times they are outdated or just wrong. There's also a good book called "SystemVerilog for Verification" by Chris Spear and Greg Tumbush which is a classic in the field. Don't be shy about reading the official documentation it will pay off in the long run.

So yeah that's basically my take on `$fdisplay` and strings. It's powerful if you use it right but it can bite you in the rear end if you are not careful. Learn it know it and use it. You will thank me later. I have to go back to fix my RTL now.

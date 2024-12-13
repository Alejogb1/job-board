---
title: "frida call function using arguments?"
date: "2024-12-13"
id: "frida-call-function-using-arguments"
---

Okay so you wanna call a function using Frida with arguments huh I've been there dude tons of times This whole process can be a bit of a headache especially when you're dealing with weird data types or some quirky native code But don’t fret I’ve spilled enough coffee over this stuff to help you out

Let's break it down First off the general idea is pretty straightforward You use Frida to hook the function you wanna call then you execute that function with your own arguments Now this seems simple but let me tell you some implementation can be a real pain in the rear end I’ve spent a week debugging a single argument passing issue in some legacy Java library Once I did a deep dive with Java reflection to figure out that the parameter was a proxy object and I had to create it from scratch on the fly in JS just to pass it to the hooked function And that’s not the worst I’ve seen it gets way worse when it comes to c++

So first things first we need to get our function hooked up I usually start with this simple approach:

```javascript
  //Assuming you have an already attached frida session in the 'session' var
  var module = Process.getModuleByName("your_module.so");
  var functionAddress = module.getExportByName("your_function_name").address;

  Interceptor.attach(functionAddress, {
    onEnter: function (args) {
      console.log("Function called with arguments:", args);
      //Here you'd put your actual logic to call the function
    }
  });
```

This snippet is basic it hooks `your_function_name` located in `your_module.so`. The `onEnter` part logs the arguments it received Now this is a great starting point to see what’s happening when the function is called it will print all the arguments that were used on that invocation.

Now for the real deal calling the function with your own args we need to manipulate the `args` array or better yet we need to call the native function by using it directly That's where things get a bit more interesting I generally use `NativeFunction` for this and you will too. Here's an example that should get the job done for simple int and string cases.

```javascript
  //Assume we want to call a function int your_function_name(int a, const char* b)
  var module = Process.getModuleByName("your_module.so");
  var functionAddress = module.getExportByName("your_function_name").address;

  var your_function = new NativeFunction(functionAddress, 'int', ['int', 'pointer']);

  var my_int_arg = 123;
  var my_str_arg = Memory.allocUtf8String("Hello Frida!");

  var result = your_function(my_int_arg, my_str_arg);
  console.log("Function result:", result);

  // Remember to free the string you allocated once its not needed anymore.
  // It is very common to miss this and end up with a very bad memory leak.
  Memory.free(my_str_arg);
```

Okay this snippet here is more involved We're using `NativeFunction` which takes the function's address the return type (int) and argument types as well. The most common type are `int` `uint` `pointer` `float` `double` and so on Be sure to check out the Frida documentation for more specific types if you are dealing with non common c data types.

We create our arguments `my_int_arg` is an integer while for string `my_str_arg` we use `Memory.allocUtf8String` because we need to pass a pointer to the C function We then call the function like a normal Javascript method passing the arguments and printing the result.

Notice the `Memory.free(my_str_arg);` this is super important you will cause a memory leak if you don’t call that when you allocate memory with Frida So always keep track of your allocated memory and free it when no longer needed this is not a game folks

If you are dealing with structures or complex object things gets complicated very quickly and this is where things can get tricky.

Let's say for example that you need to call a function that receives a C structure as input you will need to allocate that structure manually and then populate it with data before passing it as argument.

```javascript
 // Assume struct MyStruct { int field1; float field2; };
 // and  int your_function_name(MyStruct* my_struct)
  var module = Process.getModuleByName("your_module.so");
  var functionAddress = module.getExportByName("your_function_name").address;

  var my_struct_size = 8; //Size of int + float
  var my_struct_ptr = Memory.alloc(my_struct_size);

  // Set the struct fields
  my_struct_ptr.writeS32(42); // field1 is int starting at offset 0
  my_struct_ptr.add(4).writeFloat(3.1415); // field2 is float starting at offset 4

  var your_function = new NativeFunction(functionAddress, 'int', ['pointer']);

  var result = your_function(my_struct_ptr);
  console.log("Function result:", result);

  // We do not need to free `my_struct_ptr` because it is memory allocated
  // by Memory.alloc, you should free only when you use memory.allocUtf8String

```

Alright this is where some deep knowledge of C data structures starts to help We first create a new allocated memory block `my_struct_ptr` based on the size of our struct in bytes. Then we manually write to that memory location using `writeS32` for the int and `writeFloat` for the float we increment the pointer position using the `add` method to ensure the next write operations are done on the right memory position. After that we call the hooked function with the address of our created struct and the usual log the return of the function.

Important note: In this specific example we didn't need to free the struct memory as it was allocated by `Memory.alloc`. Remember that only `Memory.allocUtf8String` memory needs to be manually deallocated.

Now here's the thing you will sometimes face situations where you are dealing with classes and objects. In those cases you should look into how are those classes represented in memory and how you should handle them in Frida this usually implies that you will need to call the constructor of the classes and then call the methods directly. This can be a real rabbit hole if you have no idea what you are doing but with patience and time it can be done. (Don’t worry it happens even to the best of us).

There is another tricky part some functions may have a variable arguments list called also variadic arguments. In those cases you'll need to use some low-level techniques for instance using the `call` method instead of NativeFunction where you will pass all arguments in the correct format using registers this can be extremely complex and is usually not needed. However if you have to do it you can check the Frida documentation there are good examples of how to achieve this using `Registers`.

Regarding resources beyond the Frida docs themselves I would recommend getting your hands dirty with "Programming from the Ground Up" by Jonathan Bartlett if you want to understand how low level stuff such as memory and assembly works and "Modern C++ Design" by Andrei Alexandrescu for dealing with modern C++ concepts which you may find in modern libraries. Knowing more about low level stuff can definitely get you out of trouble more often than not

Also consider reading papers on the specific technologies you are trying to hook for instance if you are hooking a JNI library check the JNI official documentation and also the official Android documentation it is a huge time saver to know the specifics of what you are trying to hook.

Another crucial aspect is understanding the calling conventions for the architecture you're targeting This will help you understand how arguments are passed how the stack is handled and how return values are processed This is fundamental to avoid weird crashes and issues.

Also never underestimate the power of print statements `console.log()` it is your best friend and savior when dealing with a frustrating issue try to add as much log output as you can. Also use `hexdump` to see the contents of memory regions it will usually show you if a memory position has the data you expected or not.

Finally sometimes you will face the real problem that you cannot solve with regular Frida functions. In those cases you may have to create a small native library that you can compile in C or C++ that helps you with the specific problem you are trying to solve. For example accessing the contents of a weird memory region or doing some operations that are hard to do in plain javascript.

To wrap it up remember this is a marathon not a sprint so be patient take a deep breath and don’t be afraid to experiment and break things and you will definitely conquer any challenge you face I've seen it all from hooking kernel functions to reversing closed source embedded systems believe me nothing can surprise me anymore

I hope this helps and remember if you have further questions don’t hesitate to ask I'm here to share the little that I’ve learned over time with all of you good luck.

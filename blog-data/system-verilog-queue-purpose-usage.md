---
title: "system verilog queue purpose usage?"
date: "2024-12-13"
id: "system-verilog-queue-purpose-usage"
---

Alright so you're asking about SystemVerilog queues right Yeah I've messed around with those things plenty of times Been there done that got the t-shirt I guess I can spill some details and maybe help you understand how to use them when to use them why to use them and well the quirks that sometimes come with them

So basically a SystemVerilog queue is a dynamic data structure it's like a flexible array that automatically grows or shrinks as you add or remove elements It's not a static size like a normal array which I always find useful I mean who wants to manually allocate memory size for dynamic structures anyway not me

It implements the FIFO first-in-first-out structure that means the first element you put in is the first one you get out This is usually what you expect for data streams or processing chains kind of like a pipe of digital data I often use queues when handling packets in testbenches they make life way easier than constantly shifting data arrays

There are key things to know you can declare a queue using the `$` symbol before the square brackets for example an integer queue would be something like `int queue_name[$]` Notice the dollar sign means "dynamic" or unknown size at declaration time So you don't have to commit to an initial size you just need to declare the type that will be stored in the queue that's an important concept So it can be a queue of integers reals structs objects anything

Now lets talk about the basic operations you can perform on them

-   `push_back` or `push_back(element)` adds a new element to the end of the queue it's like adding a car to the end of a train I use it a lot for adding incoming data

-   `push_front(element)` adds a new element to the beginning of the queue this is like pushing a train car at the start it's less common but sometimes I need to prepend something

-   `pop_front()` removes and returns the first element of the queue which is always the oldest one This is often used in consuming the incoming data

-   `pop_back()` removes and returns the last element of the queue which is the most recent one This is used less often but still handy for some specific use cases

-   `size()` returns the current number of elements stored in the queue that's useful for checking emptiness or fullness before you perform any operation

-   `insert(index, element)` inserts an element at a specific index this is less common but sometimes necessary

-   `delete(index)` removes element from a specific index again less common

-   `queue_name.randomize()` will randomize all the elements of a queue or if you randomize a queue of objects will randomize all of its fields

Now let’s get into some code examples to clear things up a bit

```systemverilog
module queue_example;

  int my_queue[$];
  int i;

  initial begin

    // Push elements to the queue
    my_queue.push_back(10);
    my_queue.push_back(20);
    my_queue.push_back(30);

    // Check the current size
    $display("Queue size %0d",my_queue.size()); // Output: Queue size 3

    // Pop an element from the front
    i = my_queue.pop_front();
    $display("Popped element from the front %0d",i); // Output: Popped element from the front 10

    // Check again the size
    $display("Queue size %0d",my_queue.size()); // Output: Queue size 2

     //Push another element to the back
    my_queue.push_back(40);
    // Pop an element from the back
     i = my_queue.pop_back();
    $display("Popped element from the back %0d",i);  // Output: Popped element from the back 40

    //Check the last size
    $display("Queue size %0d",my_queue.size()); // Output: Queue size 2


    // Check what is left
     for(int j=0; j < my_queue.size() ; j++)
         $display("Left over elements %0d",my_queue[j]); // Output: Left over elements 20 and 30

  end

endmodule
```

Here we declare a queue called `my_queue` of type integer we add some elements we check the size remove elements and check again I think the comments are self-explanatory

One thing you gotta be careful about is accessing elements out of the queue bounds like you do in a static array you must not use indices out of range If your queue has five elements you cannot use the 7th element this will result in an error

Now for the common use cases in my experience queues are great for implementing FIFOs in data processing blocks for example if you have a testbench for a data processing IP you often use queues to store the data coming in and the data being processed that way you can keep track of the data easily You can also use them as buffers for data that comes in bursts You can also use them for data transfer between processes like in concurrent testbenches For instance when sending packets in different threads using queues helps in synchronized communication

Let’s say I’m receiving data and need to process it using a task I can use queues to implement the data transfer between them see this code

```systemverilog
module queue_task_example;

  int data_queue[$];

  task producer;
    input int data_val;
    data_queue.push_back(data_val);
    $display("Produced: %0d", data_val);
  endtask

  task consumer;
    output int processed_val;
    if(data_queue.size() > 0)
        processed_val = data_queue.pop_front();
    else
        processed_val = -1;
    $display("Consumed: %0d", processed_val);
  endtask

  initial begin
    fork
        begin
            //Producer thread
            repeat(5)
              begin
                  producer($random % 100);
                  #10;
              end
        end
        begin
            //Consumer thread
           repeat(5)
               begin
                  int val;
                  consumer(val);
                  #15;
                end
        end
    join
  end

endmodule
```

In this example we create a producer and consumer task that operate concurrently The producer adds data to the queue and the consumer pulls it from the queue this demonstrates a simple producer consumer pattern a very common one in data processing

And one last one that might be useful

```systemverilog
module queue_struct_example;

  typedef struct packed {
    int id;
    bit [7:0] data;
  } packet_t;

  packet_t packet_queue[$];

  initial begin
    packet_t pkt;

    pkt.id = 1;
    pkt.data = 8'hAA;
    packet_queue.push_back(pkt);

    pkt.id = 2;
    pkt.data = 8'hBB;
    packet_queue.push_back(pkt);


    for(int i = 0; i < packet_queue.size(); i++) begin
      $display("Packet ID: %0d Data: %h", packet_queue[i].id, packet_queue[i].data);

      end

  end
endmodule
```

This code example shows that you can use queues with structs or any other data type you define. As long as you are consistent in the type you use when declaring the queue and when manipulating the queue there shouldn't be an issue

As for resources that may help well I'm not a big fan of links if you want to really know your things you need good books and papers I suggest searching for the "SystemVerilog IEEE 1800 standard" it's the bible of this language then you can read papers or research about specific advanced usage of SystemVerilog like for advanced verification techniques You can look at papers about assertion based verification or constraint random verification that often use queues extensively There are books also about these topics as well

Ok so I think that's all I can remember for now about my experience with queues in SystemVerilog they are pretty useful tools really I cannot stress that enough Once you understand the fundamentals I bet you will use them all the time When I first used them I thought "hey this is kinda like a list in python but with better performance" but then I found out that its better to be treated as a pipe of data like I said earlier I think I spent a week just messing around with queues on my first project and it helped a lot
And as a final comment I hope I answered your question and don't worry we are all beginners at some point and it's always helpful to share knowledge just try not to get lost in the queue of questions you'll find out there get it queue of questions sorry i just couldnt help it

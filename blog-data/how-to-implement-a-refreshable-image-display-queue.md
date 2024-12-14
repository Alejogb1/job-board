---
title: "How to implement a Refreshable Image Display Queue?"
date: "2024-12-14"
id: "how-to-implement-a-refreshable-image-display-queue"
---

so, you're looking to build a refreshable image display queue, huh? i've definitely been down that road before, many times actually. it's one of those things that sounds simple enough on the surface, but can get pretty hairy once you start diving into the details. let me share my experience and how i usually tackle it, keeping it techy and down to earth.

the core problem, as i see it, is managing a sequence of images where the display needs to update at intervals, but you also want the flexibility to add, remove, or prioritize images in that queue. and, crucially, to do all this efficiently without bogging down your app or device.

my first real encounter with this was back in the early days, i was working on this embedded system for a digital photo frame (yeah, those were a thing!). it was ridiculously underpowered – think kilobytes of ram not megabytes and a processor that coughed a little bit each time it had to decode a jpeg. we had this slideshow feature and initially, i just loaded all the images into an array, and then cycled through them. it was a mess, things got slow when the user tried to copy lots of photos at once to the frame, the frame was not always the fastest device. i knew i needed a queue-based approach, but something a bit more sophisticated than just a basic fifo.

so, where do we start? first things first, a structure to hold our images. i'd generally avoid loading the full image into the queue structure itself, for performance and memory usage reasons, especially if you are dealing with high-resolution stuff. instead, i prefer to store pointers or file paths to the image data itself.

here is a struct in c++ to get an idea:

```cpp
struct image_queue_item {
  std::string image_path;
  int priority;
  bool is_loaded;
  // we might add other metadata, like the image creation date
};
```

next up, the queue itself. i tend to use a `std::deque` in c++ for this – it provides efficient insertion and removal from both ends. the priority attribute in our struct lets us sort or re-order our queue as needed. i sometimes use a priority queue (`std::priority_queue`) when i need to keep the queue always sorted by priority but that has its drawbacks when trying to do manipulations in the queue as it is always sorted. it makes more sense to maintain and reorder a `std::deque` and sort it when we need to refresh the display or push a new image with higher priority.

here's how you might represent the queue in c++:

```cpp
#include <deque>
#include <string>
#include <algorithm>

class image_queue {
public:
  void add_image(const std::string& path, int priority);
  void remove_image(const std::string& path);
  image_queue_item get_next_image();
  void sort_queue();

private:
  std::deque<image_queue_item> queue;
};

void image_queue::add_image(const std::string& path, int priority) {
    queue.push_back({path, priority, false});
}
void image_queue::remove_image(const std::string& path){
    auto it = std::remove_if(queue.begin(), queue.end(),
                            [&path](const image_queue_item& item){return item.image_path == path;});
    queue.erase(it, queue.end());
}
image_queue_item image_queue::get_next_image() {
    if (queue.empty()) {
      return {};
    }
    image_queue_item next = queue.front();
    queue.pop_front();
    return next;
  }

void image_queue::sort_queue() {
    std::sort(queue.begin(), queue.end(),
              [](const image_queue_item& a, const image_queue_item& b) {
                  return a.priority > b.priority; // higher priority first
              });
}
```

with this base, you can define methods to manipulate your queue. `add_image` inserts a new image with its priority, and `remove_image` removes a given image by its path. `get_next_image` returns and removes the image at the front of the queue. and then a simple sort function `sort_queue`.

now for the refreshing part. this is where things get interesting. the approach i take here depends heavily on the display environment. on that old photo frame, i had a very tight refresh loop. i would essentially call get next image, load the image if not loaded, display it and repeat, adding a small delay, that was a while loop doing that.

on a more modern platform, you probably have better options such as using timers or a refresh event trigger. the main goal is to avoid doing too much work on the main thread and to offload as much as you can of the rendering and loading. this can be done by using threads or tasks.

so, for instance:

```cpp
#include <iostream>
#include <chrono>
#include <thread>

//using the previous class:
image_queue my_queue;

void display_image(const image_queue_item& item){
    if(!item.is_loaded){
        //simulate loading image from disk/network
        std::cout << "loading image from path " << item.image_path << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    std::cout << "displaying image " << item.image_path << std::endl;
}

void display_loop() {
    while (true) {
        my_queue.sort_queue();
        image_queue_item current_image = my_queue.get_next_image();
        if(!current_image.image_path.empty()){
          display_image(current_image);
          //here we add it back to the queue to display it again after the others.
          my_queue.add_image(current_image.image_path, current_image.priority);
        }
        else{
            std::cout << "Queue empty" << std::endl;
        }

      std::this_thread::sleep_for(std::chrono::seconds(2));
    }
}

int main() {
    my_queue.add_image("/images/img1.jpg", 1);
    my_queue.add_image("/images/img2.png", 3);
    my_queue.add_image("/images/img3.bmp", 2);
    std::thread displayThread(display_loop);
    std::this_thread::sleep_for(std::chrono::seconds(10));
    my_queue.add_image("/images/img4.jpeg", 4); //this one will be shown before the first ones

  displayThread.join();
  return 0;
}

```

this is a simplified example, of course, but it illustrates a basic refresh loop using threads. the actual details will vary. in this example we display each image, after loading it if not loaded and re-add the image to the queue, after all the other images. after that, we sort the queue and display the next image. note that this thread is running in the background, while the main thread can add images.

a crucial detail to keep in mind is loading images. it's easy to make the mistake of loading images in the refresh loop, but that's a recipe for sluggish performance. my typical approach is to use worker threads to load images in the background.

now, when dealing with images, especially big ones, memory management becomes important. you want to avoid memory leaks or unnecessary allocation. i usually keep a cache of loaded images in a separate structure and recycle the memory to avoid re-allocating it each time.

when the image is displayed, the `is_loaded` flag in our `image_queue_item` is set. once the image is removed from the queue and displayed again the logic in `display_image` will load the image if `is_loaded` flag is false, you may even add a flag that can set that images are to be always loaded, this will depend on your use case.

resources that helped me understand this better? well, besides the usual c++ documentation, the book "effective c++" by scott meyers is always a great read for understanding how to best use the language, "modern c++ design" by andrei alexandrescu can get you deep into the details of design patters and techniques that are very useful when dealing with these kind of problems. and if you are into memory management "understanding the linux kernel" by daniel pierre bovet is a very interesting read for how low-level memory management works.

i'm not going to lie, implementing these things can be a bit of a journey. it's never as simple as it looks on paper. i've lost a lot of time when i started with this stuff but after all these years i can navigate this sort of issue fairly quickly.

oh and you know what they say about multi-threading right? the problem is, it never happens when you want it to. but i guess that's the fun part, is not it? hope this helps. let me know if you have any other questions, i'm happy to share more.

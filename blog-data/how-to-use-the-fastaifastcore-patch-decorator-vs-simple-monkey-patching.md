---
title: "How to use the fastai.fastcore patch decorator vs simple monkey-patching?"
date: "2024-12-15"
id: "how-to-use-the-fastaifastcore-patch-decorator-vs-simple-monkey-patching"
---

alright, so you're asking about the difference between using `fastcore`'s `@patch` decorator and just doing a good old monkey patch in python, eh? i’ve been there, trust me. i’ve got a few scars from the early days of messing with dynamically changing behavior in python classes. this isn't exactly a new problem but `fastcore` provides a pretty slick way to manage it.

basically, monkey patching, in its raw form, is just directly modifying an object’s or class's behavior at runtime. you're reaching into the guts of a module or class and tweaking things. it’s powerful, yes, but also, as my old mentor used to say, "with great power comes the potential to cause great headaches." that’s because if not done with care, monkey patching can quickly make your code really difficult to trace and reason about. things get modified from the outside, and it's not always immediately apparent *where* that change is happening. this leads to weird bugs. oh, the weird bugs.

on the other hand, the `@patch` decorator from `fastcore` is a structured approach to the same thing. it’s still monkey patching underneath the hood but wrapped in a nice, predictable package. instead of just swapping out a method in some random place, `@patch` lets you add a method to a class as if it were declared inside the class itself. it feels cleaner, more integrated, and it's easier to see what's going on.

let's look at some concrete examples. a few years back, i was working on a complex image processing pipeline (this was before all the fancy ai stuff hit the scene, yes, i’m *that* old school) and i needed to add a specific grayscale conversion method to a class in some older library i was using. without `fastcore`, we were doing this (and believe me, it wasn't pretty)

```python
import numpy as np

class ImageProcessor:
    def __init__(self, image):
        self.image = image

    def display(self):
        print("image displayed")

def grayscale_conversion(self):
    # simplified grayscale conversion logic here
    # just for example purposes
    gray_image = (0.299 * self.image[:,:,0] + 0.587 * self.image[:,:,1] + 0.114 * self.image[:,:,2]).astype(np.uint8)
    self.image = gray_image
    return self

# traditional monkey patching
ImageProcessor.grayscale = grayscale_conversion


#testing
image = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)
processor = ImageProcessor(image)
processor.grayscale()
print(processor.image.shape) # (100, 100)
```

see that? we had to define `grayscale_conversion` function outside the `imageprocessor` class and then essentially just glue it onto the class. not terrible in a single instance, but imagine this happening dozens of times across your codebase. and god forbid you have to debug it later. finding all the scattered patches was like going on a treasure hunt, except the treasure was always some obscure bug.

it gets much cleaner with `fastcore`. here's the same thing but using `@patch`:

```python
import numpy as np
from fastcore.patch import patch

class ImageProcessor:
    def __init__(self, image):
        self.image = image

    def display(self):
        print("image displayed")

@patch
def grayscale(self: ImageProcessor):
    gray_image = (0.299 * self.image[:,:,0] + 0.587 * self.image[:,:,1] + 0.114 * self.image[:,:,2]).astype(np.uint8)
    self.image = gray_image
    return self

#testing
image = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)
processor = ImageProcessor(image)
processor.grayscale()
print(processor.image.shape) # (100, 100)
```

notice how the `@patch` decorator is attached to our `grayscale` function? this tells `fastcore` that this function should act as a method on the `imageprocessor` class. it's much more readable, because the intent is clearly stated at the definition of the function itself. `fastcore` also handles type hinting nicely. see `self: ImageProcessor`? it allows it to do type checks, and make sure you're applying to the class you expect to apply.

now, let's talk about another scenario. what if you need to patch a class method that takes some additional parameters? well, classic monkey-patching, of course, handles it. the `@patch` decorator in `fastcore` handles this too but it’s just a bit more explicit. let's say, for instance, that i need to add a `resize` function to my image processor. i've learned a thing or two so i will do it with fastcore. take a look.

```python
import numpy as np
from fastcore.patch import patch
from skimage.transform import resize

class ImageProcessor:
    def __init__(self, image):
        self.image = image

    def display(self):
        print("image displayed")

@patch
def grayscale(self: ImageProcessor):
    gray_image = (0.299 * self.image[:,:,0] + 0.587 * self.image[:,:,1] + 0.114 * self.image[:,:,2]).astype(np.uint8)
    self.image = gray_image
    return self

@patch
def resize_image(self: ImageProcessor, new_size):
  self.image = resize(self.image, new_size, anti_aliasing=True)
  return self

# testing
image = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)
processor = ImageProcessor(image)
processor.grayscale()
print(processor.image.shape) # (100, 100)
processor.resize_image((50, 50))
print(processor.image.shape) # (50, 50)
```

again, you'll notice `@patch` just sits there, making things much more organized, and the arguments `new_size` are passed just as you would expect. it makes the code much more understandable.

another benefit that might not be immediately obvious is that `fastcore`’s patch decorator plays much nicer with tooling. code editors and linters understand what you’re doing when you use `@patch`. when you simply swap out a method on the fly, you lose the type checking and autocomplete benefits that come with modern IDEs. you lose those intellisense features, which, over the long run, makes you faster and less error-prone.

now, i know what some of you might say – "i don’t want to add another library dependency just for patching a class." and to that, i say, fair enough. if you have a very simple, contained scenario where a single monkey patch is all that is needed, sure, you can go for it. i’ve been there. but i’ve learned it’s a false economy to avoid small, well-designed libraries. the time you save in debugging and maintenance usually surpasses the overhead of managing a small extra dependency. and `fastcore` is a library which is used in many other libraries. so you might be already using it without knowing, or you might be using some library which uses it. it's like that saying, "don't reinvent the wheel, unless you plan to learn a lot about wheels". (yes, i know, that was lame, i’m old, give me a break).

in short, using `fastcore`'s `@patch` is more about writing maintainable, readable code than just saving a few lines. it's about structure, about making your intent explicit, and about making your future self not curse your past self for some weird hack. when things get hairy, as they always do in software, you'll be glad you used a structured method instead of plain monkey-patching.

for resources to go deeper, i would recommend looking into:

*   "fluent python" by luciano ramalho, there's a pretty awesome explanation on monkey patching and dynamic attributes in python in one of its chapters. it is a bit more advanced so, be ready.
*   the official `fastcore` documentation: it's concise and well-written. just read the part about `@patch` it should cover all that's needed.

i hope that makes it a bit clearer. feel free to ask if anything is not that clear.

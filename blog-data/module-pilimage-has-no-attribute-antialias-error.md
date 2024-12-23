---
title: "module 'pil.image' has no attribute 'antialias' error?"
date: "2024-12-13"
id: "module-pilimage-has-no-attribute-antialias-error"
---

 so you're getting that `module 'PIL.Image' has no attribute 'antialias'` right Yeah I've been there trust me I've wrestled with that particular beast more than once and it's usually down to a PIL version mismatch or just plain old misunderstanding how things have evolved in the Pillow land

First things first let's clarify a bit PIL the Python Imaging Library is the ancestor and Pillow is the maintained actively forked version Pillow is what you want to be using and the `antialias` thing is precisely why You probably have an old PIL version lurking somewhere maybe you inherited some code or haven't updated in ages and it's biting you now

The `antialias` attribute was part of a legacy resizing method in older PIL versions This method was deprecated then fully removed from Pillow since it had performance and other issues The go-to resizing method now involves using the `Image.Resampling` enum constants and they offer a much more fine-grained control

Let's talk about your problem specifically Your code probably has something along the lines of this

```python
from PIL import Image

image = Image.open("some_image.jpg")
resized_image = image.resize((new_width, new_height), Image.ANTIALIAS) # This is the bad guy
```

This is the code that throws the error since `Image.ANTIALIAS` isn't a thing anymore and the pillow is screaming at us for that You need to replace `Image.ANTIALIAS` with one of the available resampling filters from the `Image.Resampling` enum like `Image.Resampling.LANCZOS` or `Image.Resampling.BILINEAR` or `Image.Resampling.NEAREST` the latter is faster but less accurate

Here's the fixed code that will likely solve your issue

```python
from PIL import Image

image = Image.open("some_image.jpg")
resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS) # This is the good guy
```

The `Image.Resampling.LANCZOS` is a good general purpose resampling filter but depending on your use case you might want to experiment with other filters Check Pillow’s documentation you know its actually pretty decent.

Now I've been through the hell of debugging similar issues when working on some image processing pipeline a few years ago I was converting like thousands of scanned documents to a web friendly format and I had this old code that was using `Image.ANTIALIAS` everything was working fine on my dev environment because i was working with the most up-to-date versions of Pillow and Python but when we pushed to production we got this crazy error that took me hours to figure out because we were running the process in a docker container with an old version of PIL that nobody was aware of. I remember my colleague was pulling his hair out i told him that we need to check our package versions he said ‘no they are all fine!’ i replied ‘ok lets take another look!’ It was such a funny situation at that moment because he was so confident but he just missed the small versioning detail. Turns out the docker image used an older PIL version which didn't have the `Image.ANTIALIAS` and since then I've always added strict version control to our dependencies. This taught me the hard way about version mismatches

So yeah that's the main solution to the `module 'PIL.Image' has no attribute 'antialias'` problem Make sure you’re using a recent version of Pillow and use the `Image.Resampling` constants Also when you work in any project i urge you to always use virtual environments and specify your dependencies.

Just to clarify your error is very straightforward the old `Image.ANTIALIAS` attribute is deprecated and then removed You're using either an outdated Pillow version or you just need to change the syntax to the new standard You can find more info about this in the official Pillow documentation about resampling if you want to really dive deep into it check out “Digital Image Processing” by Rafael C. Gonzalez and Richard E. Woods it’s a bible in the image processing community

Now just to give another example if you want to resize an image to a thumbnail and you want a fast version use `Image.Resampling.NEAREST` it would look something like this

```python
from PIL import Image

image = Image.open("another_image.png")
image.thumbnail((128, 128), Image.Resampling.NEAREST) # fast and small resizing for thumbnails

image.save("thumbnail.png")
```
Remember to check Pillow’s documentation to get updated on the latest info about any changes or new additions the library had. As i said before the `Image.ANTIALIAS` is long gone and the Resampling constants is the way to go now and its pretty straightforward

So yeah it's all about Pillow version and using the right method. Check your imports check your packages and replace the old deprecated syntax with the new and you should be good to go and always remember virtual environments is your friend. If you still have problems just paste the traceback and all relevant details.

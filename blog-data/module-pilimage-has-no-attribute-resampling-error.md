---
title: "module 'pil.image' has no attribute 'resampling' error?"
date: "2024-12-13"
id: "module-pilimage-has-no-attribute-resampling-error"
---

Okay so you're hitting the dreaded "module 'PIL.Image' has no attribute 'resampling'" error right Been there wrestled with that beast myself back in the day feels like a lifetime ago but the scars remain you know how it is

First off let's get this straight this isn't your fault per se Python imaging library PIL or Pillow as we call it nowadays is a fickle mistress Sometimes the error message throws you off and you start thinking you're doing something wildly wrong but usually it's a small detail hidden in the package itself

So you're trying to use the `resampling` attribute within the `PIL.Image` module and it's just not there I know that pain It's like reaching for your favorite wrench and finding a spanner I think in my experience I faced this around the time Pillow 8 came out and I had a lot of legacy code that didn't use named constants for filters I was working on a personal project back then a crude image resizing app it wasn't pretty but it was mine

Let me give you the lowdown on why you're probably seeing this and how to fix it

The problem is that the `resampling` attribute was introduced in Pillow version 9.0.0 before that versions you didn't use it directly you instead used the older named constants or integers that PIL used internally in functions like `resize` or `thumbnail` It's a classic case of API change that can bite you if you don't track package upgrades Now this change was made for better consistency and you have more readable code but when you're stuck with older versions well it's a real drag

So if you're seeing this error it most likely means you're using an older version of Pillow earlier than 9.0.0 I mean I don't know your exact setup it's like trying to debug a remote server without ssh but in most of the cases that's what happened

Here's the first piece of the puzzle check your Pillow version You can usually do this from your terminal:

```bash
python -m pip show Pillow
```

That should give you the version number. If it's anything below 9.0.0 there's your culprit and the quickest fix? Upgrade Pillow to a more recent version I mean come on don't use outdated packages in your code unless you are working with old servers of course and even then there are some upgrades that you can do

You can use pip to do that just run:

```bash
python -m pip install --upgrade Pillow
```

After that try running your script again If you're still stuck which can happen sometimes if you are doing all sorts of shenanigans with your python environment or you're in some docker image then there is more that we have to check.

Now assuming the upgrade worked which it should here is how you would use `resampling` attribute with Pillow 9.0.0 or later:

```python
from PIL import Image

# example image path
img_path = "my_image.jpg" # Replace with your image path

try:
    img = Image.open(img_path)
    # Resize with LANCZOS resampling
    resized_img = img.resize((img.width//2, img.height//2), resampling=Image.Resampling.LANCZOS)
    resized_img.save("resized_image_lanczos.jpg")
     # Resize with BILINEAR resampling
    resized_img_bilinear = img.resize((img.width // 2, img.height // 2), resampling=Image.Resampling.BILINEAR)
    resized_img_bilinear.save("resized_image_bilinear.jpg")
    # Resize with NEAREST resampling
    resized_img_nearest = img.resize((img.width // 2, img.height // 2), resampling=Image.Resampling.NEAREST)
    resized_img_nearest.save("resized_image_nearest.jpg")

except FileNotFoundError:
    print(f"Error: Image file not found at {img_path}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
```

Notice how we use `Image.Resampling.LANCZOS` `Image.Resampling.BILINEAR` and `Image.Resampling.NEAREST` that's the new way It's part of an enum that Pillow now exposes for better code structure It's a good move to be honest much more readable than dealing with integers like `Image.LANCZOS` or whatever.

But what if you're stuck on an older version for some reason some legacy project or ancient server I've been there trust me You might need to stick to the old way for now and still be able to use it like it's meant to

This is how you would do it with Pillow versions before 9.0.0:

```python
from PIL import Image

# Example image path
img_path = "my_image.jpg" # Replace with your image path

try:
    img = Image.open(img_path)
    # Resize with LANCZOS resampling (old way)
    resized_img_old_lanczos = img.resize((img.width//2, img.height//2), Image.LANCZOS)
    resized_img_old_lanczos.save("resized_image_old_lanczos.jpg")

    # Resize with BILINEAR resampling (old way)
    resized_img_old_bilinear = img.resize((img.width//2, img.height//2), Image.BILINEAR)
    resized_img_old_bilinear.save("resized_image_old_bilinear.jpg")

    # Resize with NEAREST resampling (old way)
    resized_img_old_nearest = img.resize((img.width//2, img.height//2), Image.NEAREST)
    resized_img_old_nearest.save("resized_image_old_nearest.jpg")

except FileNotFoundError:
    print(f"Error: Image file not found at {img_path}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
```

See? `Image.LANCZOS` `Image.BILINEAR` `Image.NEAREST` These are the old constants that were used instead of the `resampling` enum. And there is a lot of other constants that exist just a quick look in the documentation will tell you more about the possible constants. It's all there it's not rocket science. But if we want to be honest it wasn't the easiest thing to use and that's the reason why they changed it I think. Also they added more resampling methods in the new implementation.

Remember this about those constants the ones before the version 9.0.0 these were just integers that mapped to particular algorithms for resampling so for example `Image.LANCZOS` was something like `4` on the previous versions. But do not try to remember it now just make the jump and upgrade your Pillow to the version 9.0.0 and use the enum.

I am not going to bore you with all the possible combinations of parameters but always refer to the official documentation I know you are a coder you can read technical documentation if not well there is no hope for you. There is always the possibility that some dependencies or packages can interfere in the usage of Pillow so if it doesn't work I'm sure you can search it on the internet if the upgrade works but you still have some other errors.

One thing I'd say avoid using very old versions of Python packages unless you really need it it's just a bad idea in the long run and you will probably have more issues not just with the `resampling` attribute but with other things too. This is just a general rule of thumb when you start making more complicated apps and packages start to pile up. One wrong package version can mess up everything I am sure you have experienced it.

Oh one more thing if you're working with scientific image processing you might want to check out some papers on image resampling algorithms like the ones they used in the package. If you want to really understand what these resampling techniques do under the hood I recommend checking "Digital Image Processing" by Rafael C. Gonzalez and Richard E. Woods or "Computer Vision: Algorithms and Applications" by Richard Szeliski These books are classics in the field and will give you the technical understanding of each algorithm used.

And the joke that you asked for oh here we go why was the image sad? Because it was feeling pixelated get it heh heh anyway moving on

So to recap check your Pillow version upgrade if needed or use the older constants and that should sort it out. Always check the documentation especially when they make big changes like this it's there for a reason and it's useful. And that's pretty much it for this issue if you have more problems with this just let me know but hopefully this should help you to sort out the dreaded `resampling` error.

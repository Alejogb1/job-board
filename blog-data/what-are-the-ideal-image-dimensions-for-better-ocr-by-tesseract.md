---
title: "What are the Ideal image dimensions for better OCR by Tesseract?"
date: "2024-12-14"
id: "what-are-the-ideal-image-dimensions-for-better-ocr-by-tesseract"
---

alright, let's talk about image dimensions for tesseract, it's a topic i've spent way too many hours on, believe me. i've seen it all, from barely-there text to massive scans that choked my poor little machine.

first off, there’s no single magic number. it’s more of a ‘it depends’ situation, but with some good guidelines that i've learned through my own painful experiences. what i've found is that you’re aiming for a sweet spot: not too small that the characters are unrecognizable, and not so big that tesseract wastes time processing a bunch of unnecessary pixels, or worse, introduces its own errors trying to make sense of the extra data.

back in the days of my early experiments, i had a real doozy of a project, converting a huge collection of old library cards. those things were tiny, scanned at low resolution, and the text was often faded. i started using default settings, and tesseract produced gibberish, to say the least, it felt like i was trying to read an ancient alien language. i then started to experiment by resizing the images and that is where the journey began.

so, what are these guidelines? well, first, tesseract needs some space to work with. if a character is only a few pixels tall, it's basically a guessing game for the algorithm. i generally recommend that the average character height in your image should be around 20 to 30 pixels. this works quite well for a good number of standard fonts. if your text has a mix of sizes, aim for the average character height. but if, for example, you have text that is always very small you could try a lower number but never lower than 10.

another important aspect is the dpi. tesseract loves 300 dpi images. anything lower, and the image can lose vital details. anything much higher can be overkill, adding processing time without a significant gain in accuracy. sometimes i might go for 400 dpi, but only if the source material is particularly challenging. i've even tried 600 dpi once, mostly out of sheer desperation, and the results were marginally better, but the processing time increase was substantial.

but hey, just increasing dpi and resizing is not all there is to it, you see, aspect ratio plays a role too. it is very easy to get carried away with image manipulation. i had a situation where i tried to make a long and skinny text bigger by only stretching its height. i thought it would work but it ended up completely messing the letterforms. so, you need to try to keep your aspect ratio correct when resizing.

now, let’s get into code examples, because that’s where things get real. using python and the pillow library, here's a simple way to resize an image:

```python
from PIL import Image

def resize_image(image_path, target_height, target_dpi):
    """resizes image to a target height preserving aspect ratio and sets the dpi"""
    img = Image.open(image_path)
    original_width, original_height = img.size

    height_ratio = target_height / original_height
    new_width = int(original_width * height_ratio)
    resized_image = img.resize((new_width, target_height))
    resized_image.info['dpi'] = (target_dpi, target_dpi)
    return resized_image

if __name__ == '__main__':
  image_path = "input.png"
  resized_image = resize_image(image_path, 600, 300)
  resized_image.save("output.png")

```

this script will load an image from `input.png`, resize it so its height becomes 600 while maintaining the original proportions and, set the dpi, and save the result to `output.png`. of course, you should adapt the parameters to your needs and if the image height is already good just resize the width.

here's another example of how to determine a proper size of an image that you want to process. let’s say your average character height is about 15px in the original scanned image. we need to scale it up so the text can be read better by tesseract.

```python
from PIL import Image

def determine_target_height(image_path, avg_char_height_px, target_char_height_px):
    """calculates the scaling factor and returns target height based on character heights"""
    img = Image.open(image_path)
    original_height = img.size[1]
    height_ratio = target_char_height_px / avg_char_height_px
    new_height = int(original_height * height_ratio)
    return new_height

if __name__ == '__main__':
  image_path = "input.png"
  target_height = determine_target_height(image_path, 15, 30)
  print(f"target height: {target_height}")

```

this will get you a proper height that you can use with the previous function. we are calculating here the scaling factor between current character height and the target one to get the new image height, you would need to resize the image using the previously mentioned code.

also, another thing to consider is that sometimes, resizing is not enough. if your images have noise, for example, that can really confuse tesseract. things like gaussian noise or speckles can impact the results drastically. i tried to read some printed text that was digitized using old and cheap equipment. that thing introduced so much noise that i thought tesseract was drunk when i saw the output. using some image preprocessing can really help clean up the image before throwing it to tesseract. here is some code:

```python
from PIL import Image, ImageFilter

def preprocess_image(image_path):
    """applies a median filter to reduce noise"""
    img = Image.open(image_path).convert('L') # convert to grayscale
    preprocessed_img = img.filter(ImageFilter.MedianFilter(size=3))
    return preprocessed_img

if __name__ == '__main__':
    image_path = "noisy.png"
    preprocessed_image = preprocess_image(image_path)
    preprocessed_image.save("preprocessed.png")
```

this little helper function converts the image to grayscale and applies a median filter to reduce the noise. median filters are good because they tend to preserve edges while smoothing out the noise. you should adapt the size of the filter to your specific needs. also, you may need more sophisticated preprocessing techniques depending on your images. if you need to deal with skewed images there are many rotation and perspective correction techniques that you might need to use. it all depends on the quality of the scans and what are you trying to do.

regarding resources if you are doing this kind of job, i strongly recommend the book "digital image processing" by gonzalez and woods. it's a complete bible on this topic and you will understand the principles behind what you are doing. also, for the tesseract side, check the tesseract documentation; it contains a lot of useful info, also reading the original papers where tesseract was first introduced can help you understand the underling mechanisms.

and a little tip, always test a small batch of images before processing everything. there's no point in applying the same settings to all your images if they don't work well for the first ones. i used to waste hours when i started, because i was impatient. if i could go back in time, i’d tell my past self to be more methodical.

also, one time, i accidentally wrote a script that resized my images to zero height... the server stopped processing... i then realized that zero height images do not really process, who would have thought?!

anyway, that is what i’ve learned over the years regarding image dimensions and tesseract. it is not a simple cut and paste problem, but with a bit of experimentation and understanding, you can definitely make tesseract work for you. good luck and happy coding!

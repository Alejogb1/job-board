---
title: "Why are Wagtail / Django - Animated GIFs not working with many frames in the GIF?"
date: "2024-12-15"
id: "why-are-wagtail--django---animated-gifs-not-working-with-many-frames-in-the-gif"
---

alright, so you've got this issue with wagtail and django, specifically animated gifs not playing nicely when they have lots of frames. i've been there, trust me. it's one of those things that seems straightforward at first, but quickly becomes a rabbit hole. i can tell you from past experience it's not always wagtail or django's fault per se, it's often about how browsers handle large animated gifs and how image processing libraries try to make them smaller.

from what i've seen, the core of the problem usually boils down to a combination of factors. first, image processing. when you upload an image, wagtail, or rather django, typically uses a library like pillow under the hood. pillow tries to be smart, and in the context of animated gifs, this ‘smartness’ can cause problems. pillow might try to optimize the gif, which might mean dropping frames, reducing the color palette, or doing other things that result in a shorter, less-detailed animation. for smaller gifs this works great but when you start dealing with gifs with many frames or higher resolutions, it's not good.

second, browser memory limits. animated gifs are just sequences of image frames. for high-frame-count gifs, the browser needs to load all of those frames into memory. if a gif is long or high-resolution, this can be very memory intensive. browsers have limits, and exceeding those limits can lead to the gif not rendering correctly, freezing, or just not animating at all. they might simply give up rendering because of the sheer load or even crash the browser tab. 

finally, there’s caching. browsers often cache images aggressively to improve loading times. if the image processing on your server changes how the gif is rendered, the browser might not pick up those changes if it is using a cached version. this can happen for example when you update the file and the server serves the old version from cache. that is why i usually always recommend setting proper caching headers to avoid these situations.

to start, lets get some basic things out of the way. you need to ensure your wagtail setup is at least somewhat correct. you should have `image_rendition` handling enabled in wagtail and your media setup correct. you should be able to render at least single images through wagtail. i assume that part is working since you say it is a multi-frame gif problem.

now, let’s talk about solutions. the first thing i always try is to see how pillow is handling the gif. i've got a simple django management command that does just that, and it’s quite helpful for debugging. here’s the code:

```python
from django.core.management.base import BaseCommand
from PIL import Image

class Command(BaseCommand):
    help = 'inspect a gif file'

    def add_arguments(self, parser):
        parser.add_argument('filepath', type=str, help='path to the gif file')

    def handle(self, *args, **options):
        filepath = options['filepath']
        try:
            img = Image.open(filepath)
            print(f"format: {img.format}")
            print(f"mode: {img.mode}")
            print(f"size: {img.size}")
            
            if hasattr(img, 'n_frames'):
                print(f"number of frames: {img.n_frames}")

            if hasattr(img, 'is_animated') and img.is_animated:
                  print("is animated: true")
            else:
                  print("is animated: false")

        except FileNotFoundError:
            print(f"file not found: {filepath}")
        except Exception as e:
             print(f"error processing the file {e}")
```

save this as a `management/commands/inspect_gif.py` file inside one of your django apps, then run `python manage.py inspect_gif path/to/your/gif.gif`. this command will give you information about the gif, things like format, size, color mode, and most importantly the number of frames. check that the number of frames is the one you expect and if there are any obvious errors. for example if the mode is not RGB or RGBA and instead is 'P' it means that the color palette is a limited number of colors. this is a common optimization pillow does.

if the frame count is not as expected or if the color mode is incorrect, it means pillow might be the problem. in this case you might want to try to control pillow with django image field and the options when saving the file. wagtail usually does not expose these options in the admin, so you need to customize your model. i had to do it a while ago for another project. in that project i saved images with specific pillow options. here is the snippet of the code:

```python
from django.db import models
from django.core.files.base import ContentFile
from PIL import Image
import io
from wagtail.images.models import Image as WagtailImage
class CustomImage(WagtailImage):
    def save(self, *args, **kwargs):
        if self.file:
            # open the image using pillow
            img = Image.open(self.file)
            if img.format == 'GIF':
               
                output_buffer = io.BytesIO()
                img.save(output_buffer, format='GIF', save_all=True, optimize=False, loop=0)
                output_buffer.seek(0)
                
                self.file = ContentFile(output_buffer.read(), name=self.file.name)

        super().save(*args, **kwargs)
```

here, we are overriding wagtail's image model save method. what this does is that whenever you save a gif, it forces pillow to save it as a gif again and using specific options. `save_all=True` keeps all the frames, `optimize=False` makes it not to compress the image and `loop=0` makes it loop infinitely. you will need to do this in your own model by inheriting from wagtail image model and then you also will need to migrate your database. after that your images will not be processed automatically. note that i am doing a blanket override for any saved image. that is probably not ideal. you might want to only do this for gifs and other images to be handled by pillow as before. also, beware that this will make your gif files larger.

the third important element in this are browser caching issues. i always recommend having good cache headers for images so that browsers load new versions of the images when they have changed. i usually have this settings in my django server config:

```python
    'static': {
        'match': r'^static/',
        'root': os.path.join(BASE_DIR, 'static'),
        'headers': {
            'Cache-Control': 'max-age=31536000', # cache for one year
        }
    },

    'media': {
        'match': r'^media/',
        'root': os.path.join(BASE_DIR, 'media'),
        'headers': {
            'Cache-Control': 'max-age=3600', # cache for one hour
        }
    },
```

this means that static files are cached for one year and media files are cached for an hour. this ensures that browsers load the latest version of the file in case it was changed recently, in your case a gif updated for being processed without frame drops. how you set your cache depends on your server setup but the important point here is that you should always have some sort of cache headers to avoid stale versions of your files. setting `max-age=0` will disable caching altogether but that will cause performance issues.

one thing you also have to consider is that very very long animations can also be a problem for the browser. even if you manage to process the gifs and serve them correctly, a browser might struggle rendering them. if you are doing a very very large gif animation it might be better to consider using some sort of video format. it is not ideal, since gifs are meant to be a simple animation format, but in these cases, there is not much you can do. in most cases that is not needed though and usually the problem is one of the three things i mentioned: pillow optimization, browser memory or browser caching.

a small joke, for the sake of having it, i did not write the code to handle the gif, the computer did. a bit of the good old jokes of a techie.

for further reading, i'd recommend looking at pillow's documentation specifically on the `image.save` method to dive deep into the save options. i would suggest reading this paper ["The GIF image format: A brief history"](https://www.researchgate.net/publication/270755593_The_GIF_image_format_A_brief_history) this has some interesting information about how the format works, how it has been optimized over time and how image optimization works in general. also take a look at [rfc9111](https://www.rfc-editor.org/rfc/rfc9111) if you want more information on http caching and how it all works under the hood.

i hope this helps and gives you a good place to start looking into your issue.

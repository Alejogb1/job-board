---
title: "Is there a way to show images in a Wagtail Model Admin Record Listing page?"
date: "2024-12-15"
id: "is-there-a-way-to-show-images-in-a-wagtail-model-admin-record-listing-page"
---

alright, so, you're looking to get images showing up in the wagtail model admin record listings, right? yeah, i've been down that rabbit hole before. it's not as straightforward as it probably should be, but totally doable. let me share some stuff i've picked up over the years.

first thing, wagtail's admin is built on django's admin, so a lot of the concepts are pretty similar, but with a wagtail-specific twist. the basic list view you see, that's just rendering stuff based on the model's fields. directly dropping images in there...not gonna work. django admin, and by extension wagtail admin, isn't designed to render full html snippets by default within the list display columns. it expects simple text or a reference to a display function. it's all about the database.

my first encounter with this problem was way back when i was building an internal tool for managing our product images. the client wanted to be able to quickly see thumbnails of the products in the admin, instead of just file names. i mean, who can keep track of all those `product_123_v3_final.jpg` names without losing their minds. the default admin listing was…painful.

anyway, here’s the core of the approach: we need to create a custom method in our model that constructs the html tag for the image and then tell wagtail to use it for the list view column. we'll take advantage of `format_html` from django to avoid any xss issues. that's pretty essential, you *really* don't want to be messing with raw html strings and user input especially in the admin.

let’s imagine you’ve got a model called, say, `product`, and it has an image field, let's call it `main_image`. here's what your `models.py` would start looking like:

```python
from django.db import models
from django.utils.html import format_html
from wagtail.admin.panels import FieldPanel
from wagtail.images.models import Image
from wagtail.models import Page

class Product(models.Model):
    title = models.CharField(max_length=255)
    main_image = models.ForeignKey(
        'wagtailimages.Image',
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name='+'
    )

    panels = [
        FieldPanel('title'),
        FieldPanel('main_image'),
    ]

    def image_preview(self):
        if self.main_image:
            return format_html('<img src="{}" width="100" height="100" />',
                                 self.main_image.get_rendition('fill-100x100').url)
        return 'No Image'

    def __str__(self):
        return self.title
```

see that `image_preview` method? that’s where the magic happens. if there's an image, we use wagtail's `get_rendition()` method to get a thumbnail (i've hardcoded 100x100 here, but you should make that configurable or dynamic in real projects). importantly, we're passing in `format_html` so django automatically escapes the url for security reasons. if there is no image then return a 'No Image' string, and for the product listing display a `__str__` representation of the title

now, in your `wagtail_hooks.py` or similar where you register your models with the admin, you'll modify the `ModelAdmin` definition:

```python
from wagtail.contrib.modeladmin.options import ModelAdmin, modeladmin_register

class ProductAdmin(ModelAdmin):
    model = Product
    menu_label = 'Products'
    menu_icon = 'form'
    list_display = ('title', 'image_preview')
    search_fields = ('title',)


modeladmin_register(ProductAdmin)

```

the key part here is `list_display = ('title', 'image_preview')`. this tells wagtail to show both the title of the product and the returned string from `image_preview` method in the list view. and boom. images should now be displaying as thumbnails.

now, don't get me wrong, it's tempting to bypass the `rendition` process and just throw the original url in there. you *could*, but you *really* shouldn't unless you plan on showing huge unoptimized images in your admin listing. not good. always, always use renditions for thumbnails. or your admin will load slower than a dial-up modem during a solar flare and no one will thank you for that.

i’ve had experiences where i was pulling in images from external sources. and that needs a totally different approach, since you won't be able to use the `wagtailimages.Image` model. if you are storing just the urls instead of an actual wagtail image instance, you would modify the `image_preview` method in a similar way, like so. imagine you have `image_url` as a CharField that stores the full url instead of foreign key.

```python
    image_url = models.CharField(max_length=255, null=True, blank=True)

    def image_preview(self):
        if self.image_url:
            return format_html('<img src="{}" width="100" height="100" />',
                                 self.image_url)
        return 'No Image'
```

the code still works because we're creating the same html tag, but instead of using the rendition function provided by wagtail, we are using the raw url from a string field, that has been already escaped by django's `format_html` .

one thing i learned the hard way - and it’s a bit unrelated to this, but relevant - is that when it comes to image sizes, browser caching can sometimes mess with you. you change the rendition size, you update your models, but the images still look the same in the admin. remember to force refresh your admin in case the images don't appear correctly. (i once spent like 3 hours debugging this… don’t ask).

now, for more complex stuff like having custom display logic or custom html for different model types, you'd probably need to look into writing custom template tags or inclusion tags and referencing those on the list_display field, but that's another topic for another time. for this you can refer to the django's template tags documentation it has some pretty cool examples for more advanced user interfaces in lists and more. this can make the admin panel much more dynamic, but if we're being honest, you will rarely need it if you just need to show thumbnail representations.

also, if you are dealing with massive amounts of data, you might want to think about the performance impact of fetching the rendition for *each* item in the list. for large tables, it could slow down the rendering. it's not too common but if you are dealing with thousands of records it might be something you should take into account. maybe look at the documentation of django-select2, that is sometimes recommended on these cases, and allows you to lazy load the data in the admin pages. for the majority of cases this is not a concern at all, but i wanted to note it as a possibility.

i hope this helps. honestly, displaying images in the admin might seem daunting at first, but once you get the basic flow, it becomes quite easy, and opens lots of different possibilities for making your wagtail admin more intuitive.
one more thing before i go, if you want to dive deeper into django's admin, have a look at "two scoops of django". it's not wagtail-specific, but it covers a lot of the core concepts really well, and some things overlap with wagtail's admin, it helped me a lot when i started out. and of course the official django documentation is a must read. they always have great examples and details of almost all the issues you might encounter.

---
title: "What Wagtail Django fields are causing the 'Unknown field' error?"
date: "2024-12-23"
id: "what-wagtail-django-fields-are-causing-the-unknown-field-error"
---

Alright, let's talk about those pesky "unknown field" errors in Wagtail, specifically when you're working within a Django project. I've definitely been down that rabbit hole a few times, and it almost always boils down to a mismatch between how your model fields are defined and how Wagtail expects to see them, especially within streamfield blocks or panel configurations. The error message itself is a bit generic, which can be frustrating, but let's break down the most common culprits and how to address them.

The core issue usually arises from Wagtail’s abstraction over Django’s model fields. Wagtail’s `ModelAdmin`, `StreamField` and `panels` in page models need to interact seamlessly with the underlying Django model fields, and any discrepancy between them can result in this "Unknown field" error. We’re dealing with a form of reflection, where Wagtail dynamically constructs forms and interfaces based on the Django model's field definitions. If these aren’t perfectly aligned, boom – you get the dreaded error.

One of the primary causes, in my experience, is incorrect or missing field definitions in the Wagtail `Panel` configurations. Let's say you’ve created a Django model with a field like this:

```python
from django.db import models

class MyModel(models.Model):
    my_text_field = models.TextField(blank=True)
    my_integer_field = models.IntegerField(null=True, blank=True)
    my_date_field = models.DateField(null=True, blank=True)
```

Now, if you intend to use this model within a Wagtail page or a snippet, and you define your panels like this:

```python
from wagtail.admin.panels import FieldPanel, MultiFieldPanel
from wagtail.models import Page

class MyWagtailPage(Page):
    content_panels = Page.content_panels + [
       MultiFieldPanel(
           [
                FieldPanel("my_text_field"),
                FieldPanel("my_intfield"), # Notice the typo
                FieldPanel("my_date_field"),
           ],
           heading="My Content",
       )
    ]
```

You'll absolutely get an "Unknown field" error. See that typo? I’ve been bitten by silly things like this numerous times. Wagtail expects the field name defined in the `FieldPanel` (in this case `"my_intfield"`) to match the Django model’s field attribute name precisely (which is `my_integer_field`). There's no leeway here; it has to be a direct, case-sensitive match. Debugging this often means meticulously comparing the field names in your Django model with those in your `FieldPanel` definitions. It’s good practice to copy and paste names to eliminate the possibility of typos.

Another common location for this error is inside `StreamFields`. Here, things get a bit more complex, but the underlying principle is the same: correct field mapping. Let’s imagine you're creating a custom block with specific fields within a `StreamField` using `StructBlock`:

```python
from wagtail.blocks import StructBlock, CharBlock, DateBlock, IntegerBlock, StreamBlock
from wagtail.fields import StreamField
from wagtail.models import Page

class MyCustomBlock(StructBlock):
    title = CharBlock()
    date = DateBlock(required=False)
    quantity = IntegerBlock(required=False)

class MyPageWithStreamField(Page):
    body = StreamField(
        [
            ("custom_block", MyCustomBlock()),
        ],
        use_json_field=True
    )

    content_panels = Page.content_panels + [
        FieldPanel("body"),
    ]
```

This example should be fine, assuming that the `MyCustomBlock` fields align with your intended data. However, if you try referencing a non-existent field when working with these blocks in a template, for example, you'll see a "template does not exist" or a rendering issue, not the specific "Unknown field" error. The 'Unknown field' error in Streamfields usually occurs when you declare the block fields themselves *incorrectly* within the definition of the `StructBlock` itself. For example:

```python
from wagtail.blocks import StructBlock, CharBlock, DateBlock, IntegerBlock
from wagtail.fields import StreamField
from wagtail.models import Page


class MyBrokenBlock(StructBlock):
    title_field = CharBlock() #This works as intended
    date_info = DateBlock(required=False) # This is incorrect
    quantity_value=IntegerBlock(required=False) # This is incorrect

class MyPageWithBrokenStreamField(Page):
    body = StreamField(
        [
            ("broken_block", MyBrokenBlock()),
        ],
        use_json_field=True
    )

    content_panels = Page.content_panels + [
        FieldPanel("body"),
    ]
```

In the `MyBrokenBlock` example above, although the field names don't directly cause the Wagtail "unknown field" error, it sets the stage for downstream problems because you're now using field names that don't match your intended model logic within templates. While Wagtail won't flag the names declared in `MyBrokenBlock` itself as unknown, any attempt to use those fields with names such as "date_info" or "quantity_value" in your Django templates will fail because Wagtail is expecting the block fields to be `date` and `quantity` from our first example. The error you might see could be anything from an inability to render a specific part of the template to runtime exceptions. This highlights how subtle mismatches in naming can cause major headaches.

Finally, another area where this error pops up is with custom `ChooserPanel` classes, particularly when dealing with foreign keys. Let’s assume you have a model with a foreign key relationship:

```python
from django.db import models
from wagtail.admin.panels import FieldPanel, ChooserPanel
from wagtail.models import Page
from wagtail.snippets.models import register_snippet


class MyRelatedModel(models.Model):
    name = models.CharField(max_length=255)
    # Additional fields here

    def __str__(self):
       return self.name

register_snippet(MyRelatedModel)


class MyPageWithFK(Page):
   related_item = models.ForeignKey(
        MyRelatedModel,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="pages",
   )

   content_panels = Page.content_panels + [
        ChooserPanel("related_item"),
   ]
```

This example above generally works well. However, if you have complex relationships with multiple models and are attempting to use different models in different panels or as foreign keys without setting them up properly, you can encounter "Unknown field" errors. For example, trying to use the `FieldPanel` instead of the `ChooserPanel` when managing a foreign key would generate this error.

```python
from django.db import models
from wagtail.admin.panels import FieldPanel, ChooserPanel
from wagtail.models import Page
from wagtail.snippets.models import register_snippet


class MyOtherRelatedModel(models.Model):
    title = models.CharField(max_length=255)

    def __str__(self):
       return self.title

register_snippet(MyOtherRelatedModel)

class MyPageWithBadFK(Page):
    another_related_item = models.ForeignKey(
        MyOtherRelatedModel,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="pages_bad",
   )

    content_panels = Page.content_panels + [
        FieldPanel("another_related_item"), #Incorrect panel here
   ]
```

In the `MyPageWithBadFK` class, using a `FieldPanel` for a foreign key named `another_related_item` is incorrect. Foreign keys require the `ChooserPanel`. Using the `FieldPanel` in this context is telling Wagtail that it needs to create a standard text field for the model attribute. It doesn’t know what to do with it and thus will throw an "unknown field" error.

To prevent this, always use the `ChooserPanel` for foreign key relationships that are intended to be managed through a chooser interface in Wagtail's admin. If you try to use `FieldPanel` with a `ForeignKey` or `OneToOneField`, Wagtail doesn't understand how to render the widget to show a chooser, and you'll get an "unknown field" error.

In summary, the “Unknown field” error in Wagtail often boils down to discrepancies between model field definitions and their usage within `FieldPanel`, `ChooserPanel` or `StreamField` configurations. It is vital to pay extremely close attention to the spelling and case of the field names, especially when dealing with `StreamFields`, nested `StructBlocks`, and foreign key relationships.

For a deeper dive into Wagtail's field handling and configuration, I'd recommend reading the Wagtail documentation, particularly the sections on `StreamField`, block definitions, and the admin interface customizations. Additionally, exploring Django's documentation on model fields is crucial. A good starting point is the official django documentation on forms and model fields. The book "Two Scoops of Django" by Daniel Roy Greenfeld and Audrey Roy Greenfeld also provides invaluable insights into best practices for Django, which translates well into working with Wagtail.

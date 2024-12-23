---
title: "How can I use a Wagtail Page model object as a default value for a streamfield block?"
date: "2024-12-23"
id: "how-can-i-use-a-wagtail-page-model-object-as-a-default-value-for-a-streamfield-block"
---

Alright, let’s tackle this interesting challenge. I’ve definitely run into this scenario before, and it's not as straightforward as one might hope, particularly when you’re aiming for that seamless user experience with Wagtail. The goal, as I understand it, is to use a pre-existing Wagtail page, perhaps a ‘contact page’ or an ‘about us’ page, as a default option within a streamfield block. This means that when a user creates a new page that utilizes this streamfield, this specific Wagtail page should pre-populate a chosen block. We can definitely achieve this, and let’s dive into how.

The core issue here stems from the fact that Streamfield blocks are data structures, and they don't inherently understand the concept of ‘live’ page objects. They store data, not references to other models directly. Thus, attempting to directly insert a page object as a default won’t work as expected; it will generally result in an error, or something equally ineffective. We need to use the page’s id or a similar identifier and then make sure that during rendering we convert this into the actual page object. Let me give you some practical examples, based on different approaches I’ve used in past projects, highlighting the pros and cons of each.

**Approach 1: Using the Page ID and a `render` method**

This first technique involves storing the target page’s ID in the streamfield block’s value and then retrieving the page during the rendering of the block. I found this to be a very robust method with relatively good performance.

Here's how that might look in code:

```python
from django.db import models
from wagtail.core import blocks
from wagtail.core.models import Page
from wagtail.core.fields import StreamField


class RelatedPageBlock(blocks.StructBlock):
    default_page_id = blocks.IntegerBlock(label="Default Page ID")  # Store the ID, not the object directly

    def get_context(self, value, parent_context=None):
        context = super().get_context(value, parent_context=parent_context)
        try:
            default_page = Page.objects.get(id=value['default_page_id'])
            context['default_page'] = default_page
        except Page.DoesNotExist:
            context['default_page'] = None
        return context

    class Meta:
        template = 'blocks/related_page_block.html'
        icon = 'site'
        label = 'Related Page'


class HomePage(Page):
    content = StreamField(
        [
            ('related_page', RelatedPageBlock())
        ],
        null=True,
        blank=True
    )

    class Meta:
        verbose_name = "Homepage"
```

Then, in your `blocks/related_page_block.html` template, you would use:

```html
{% if default_page %}
    <h2> <a href="{{ default_page.url }}">{{ default_page.title }}</a> </h2>
    <p>{{ default_page.specific.introduction|truncatewords:50 }}</p>
{% else %}
    <p>No page selected.</p>
{% endif %}
```

In this snippet, `default_page_id` is an integer block meant to store the id of the target page. During rendering via the `get_context` method, the page is fetched using this id. The benefit of this approach is that it decouples the data from page objects, ensuring minimal issues during migrations and it performs well, as you're only retrieving a single page when necessary. The disadvantage is that you must remember to update this ID if the original target page is deleted.

**Approach 2: Using an `IntegerBlock` and a Page Chooser in Wagtail Admin**

In Wagtail's admin interface, there are built-in page choosers that help users select pages. While you can't use this directly in a streamfield’s `default` field, you can still utilize the page chooser to make the ID input straightforward for content editors.

The code would look something like this:

```python
from django.db import models
from wagtail.core import blocks
from wagtail.core.models import Page
from wagtail.core.fields import StreamField
from wagtail.admin.edit_handlers import FieldPanel

class RelatedPageBlock(blocks.StructBlock):
    default_page_id = blocks.IntegerBlock(label="Related Page")

    def get_context(self, value, parent_context=None):
        context = super().get_context(value, parent_context=parent_context)
        try:
            default_page = Page.objects.get(id=value['default_page_id'])
            context['default_page'] = default_page
        except Page.DoesNotExist:
            context['default_page'] = None
        return context
    
    class Meta:
        template = 'blocks/related_page_block.html'
        icon = 'site'
        label = 'Related Page'

class HomePage(Page):
    content = StreamField(
        [
            ('related_page', RelatedPageBlock())
        ],
        null=True,
        blank=True
    )
    content_panels = Page.content_panels + [
         FieldPanel('content')
    ]

    class Meta:
        verbose_name = "Homepage"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Set default value on init
        if not self.content:
            # Set default content, ensure you have an existing page
            default_page = Page.objects.first()
            if default_page:
              self.content = [('related_page', {'default_page_id': default_page.pk})]
```
In this version, a key addition is in the `HomePage` class's init method. I'm setting the default value of the streamfield upon a new instance creation, grabbing a default page (in this case simply the first page in the database). Here, I am also passing a value within a dictionary using `default_page_id`. This dictionary will be directly inserted within your `StreamField` block. Additionally, the way we render it remains the same as our first example, fetching the page in the `get_context` method.

**Approach 3: Utilizing a `PageChooserBlock`**

Wagtail has a `PageChooserBlock` that allows users to select pages, but it still stores a page ID internally. So, while we cannot use it as a default value for the streamfield block directly, we can use it to configure the values in the backend.

This solution is straightforward, but it needs a bit of javascript manipulation to inject default values. Let me present a simplified version of the code (without javascript code injection):

```python
from django.db import models
from wagtail.core import blocks
from wagtail.core.models import Page
from wagtail.core.fields import StreamField
from wagtail.admin.edit_handlers import FieldPanel
from wagtail.core.blocks import PageChooserBlock


class RelatedPageBlock(blocks.StructBlock):
    default_page = PageChooserBlock(label="Related Page")

    def get_context(self, value, parent_context=None):
        context = super().get_context(value, parent_context=parent_context)
        try:
            default_page = value['default_page']
            context['default_page'] = default_page
        except KeyError:
            context['default_page'] = None
        return context

    class Meta:
        template = 'blocks/related_page_block.html'
        icon = 'site'
        label = 'Related Page'


class HomePage(Page):
    content = StreamField(
        [
            ('related_page', RelatedPageBlock())
        ],
        null=True,
        blank=True
    )

    content_panels = Page.content_panels + [
         FieldPanel('content')
    ]

    class Meta:
        verbose_name = "Homepage"


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Set default value on init
        if not self.content:
            # Set default content
            default_page = Page.objects.first()
            if default_page:
                self.content = [('related_page', {'default_page': default_page})]
```

In this example, `default_page` is a `PageChooserBlock` which stores the chosen page, and we render it using the `get_context` method. Similar to the previous example, I've set the default page on page creation.

**Key Considerations and Further Reading**

In all cases, using Page Ids provides flexibility, especially with complex configurations and the potential need to migrate content between databases. When selecting this approach, be sure to implement proper error handling (as in the examples) and perhaps consider using `Page.get_by_path()` if you wish to use a page’s slug instead of its ID, but this might add complexity.

For a deeper understanding of Wagtail streamfields and their internals, I’d recommend consulting the official Wagtail documentation which is very thorough, particularly the sections related to `StreamField` and custom blocks. The book “Two Scoops of Django 3” is a general yet very useful text which discusses complex Django and Wagtail patterns and gives background knowledge which would be useful when implementing more custom solutions. Also, reviewing the source code for Wagtail's `blocks` module itself (particularly `struct_block.py` and `base.py`) can be greatly beneficial to understanding how you can leverage these patterns in your own projects.

In summary, while direct inclusion of a page model in the streamfield’s defaults is problematic, storing page identifiers, and later retrieving the page during rendering, is a practical and effective way to achieve the desired behavior. These three approaches highlight different aspects of handling page object references within a streamfield context, and the choice depends on the specific requirements of your project. I hope this detailed explanation is helpful.

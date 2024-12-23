---
title: "Can a page of a specific type with an image be created as a child in Wagtail Django?"
date: "2024-12-23"
id: "can-a-page-of-a-specific-type-with-an-image-be-created-as-a-child-in-wagtail-django"
---

Alright, let's talk about crafting child pages with images in Wagtail. This is a topic I've certainly spent some quality time with over the years, having tackled similar content modeling challenges across various projects. It's a common requirement—building structured content where images are integral, and having that content sit nested within your Wagtail site. The short answer is, unequivocally, yes, you can definitely create a child page of a specific type with an image in Wagtail. But, as with most things in development, the "how" has nuances.

My past experience with a heavily content-driven news platform comes to mind. We had numerous categories of articles (let's say ‘features’, ‘reviews’, and ‘interviews’), and each article required a hero image. Moreover, we needed to allow for this nesting functionality so users could easily organize articles. This led to a deeper exploration of Wagtail's model inheritance and image handling, and I think my experience there directly relates to your question.

First, let’s unpack the core idea. Wagtail, being built atop Django, leverages Django's model inheritance quite powerfully. This enables you to define a base page type with common fields, and then create child page types that inherit from this base and add additional, specific fields. To embed an image, we'll use Wagtail's `image` field, typically in conjunction with `StreamFields` for more versatile content arrangements.

Here’s how you would typically approach this structure. I’ll use a simplified example focused on child pages of the 'feature' type to make things concrete.

**Example 1: Basic Page Model with an Image**

This snippet defines a `FeaturePage` model, our base page, which allows us to create a basic page with a title, a body, and a hero image. It will also serve as a base for the `FeatureArticlePage` model.

```python
from django.db import models
from wagtail.models import Page
from wagtail.admin.panels import FieldPanel
from wagtail.images.edit_handlers import ImageChooserPanel
from wagtail.fields import RichTextField

class FeaturePage(Page):
    """Base feature page model"""
    body = RichTextField(blank=True)

    hero_image = models.ForeignKey(
        'wagtailimages.Image',
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name='+'
    )

    content_panels = Page.content_panels + [
        FieldPanel('body'),
        ImageChooserPanel('hero_image'),
    ]
    
    #We are making this an abstract class to allow child pages
    class Meta:
        abstract = True
```
Notice the use of `models.ForeignKey` to establish the link to a Wagtail image. The `ImageChooserPanel` allows a convenient interface in the Wagtail admin for image selection. Also, I have included an abstract meta class to allow child classes to inherit from this class. If you intend for users to create pages directly from this class, remove the abstract class and meta declaration.

**Example 2: Creating a Child Page Model**

Now, we extend this. Let's define a `FeatureArticlePage`, which is a child of `FeaturePage`. Let’s assume we want to include an author field, while maintaining all of the inherited elements from our parent `FeaturePage`.

```python
from wagtail.models import Page
from wagtail.admin.panels import FieldPanel
from django.db import models

class FeatureArticlePage(FeaturePage):
    """A child page of the FeaturePage, representing a specific article."""
    author = models.CharField(max_length=255, blank=True)

    content_panels = FeaturePage.content_panels + [
        FieldPanel('author'),
    ]

    # These can be defined when you add the specific child page
    # parent_page_types = ['home.Homepage']
    # template = 'feature/feature_article_page.html'
```
In this case, `FeatureArticlePage` now inherits the `body` field and `hero_image` functionality directly from `FeaturePage`. You also see that I have included properties, which should be included in your application. The `content_panels` array appends additional custom fields to the existing one. This approach promotes code reusability and ensures consistency.

This structure allows you to, in Wagtail admin, navigate to the parent page type (`FeaturePage` in our example) and create a child page of type `FeatureArticlePage`. Wagtail will correctly handle the inheritance and display the relevant form for this child page type.

**Example 3: Template Rendering Considerations**

Finally, let’s quickly cover how you’d access and display the data in your templates. If, in your `FeatureArticlePage` model, you had specified a template called `feature/feature_article_page.html`, you could write something like this:
```html
{% extends "base.html" %}

{% load wagtailimages_tags %}

{% block content %}
  <h1>{{ page.title }}</h1>
  <p>Author: {{ page.author }}</p>
  {% if page.hero_image %}
      {% image page.hero_image original as hero_img %}
      <img src="{{ hero_img.url }}" alt="{{ hero_img.alt }}" width="{{ hero_img.width }}" height="{{ hero_img.height }}" />
  {% endif %}
  {{ page.body|richtext }}
{% endblock %}
```

Important here is the use of `{% load wagtailimages_tags %}` to use the `image` tag that pulls from Wagtail's image handling. Also note, the `{{ page.body|richtext }}` filter, which renders the rich text field’s content with all formatting included.

Now, some technical notes and recommendations.

1.  **Model Inheritance Strategy:** Be thoughtful about your inheritance strategy. Deeply nested inheritance can sometimes lead to complexities. If you find your models getting too complicated, consider using composition or mixins instead, which can provide more flexibility. Explore the concept of multiple inheritance cautiously; it can get tricky if not planned well.

2.  **Image Rendition Sizes:** Pay close attention to how you render images. You will typically want to include several image renditions (thumbnail, medium, large etc.) so that different use cases are served with appropriate image sizes. The `{% image %}` template tag, as shown, allows for that.

3.  **Content Modeling Depth:** It's tempting to nest pages indefinitely. Be mindful that very deep hierarchies can negatively impact navigation and user experience. Always consider the logical structure of your content from the user's perspective.

4.  **Performance:** For a very large site with many image assets, consider optimizing image delivery via a CDN or other caching techniques. Additionally, ensure your database indexes are in place to handle the large amounts of relational data.

For further reading, I’d suggest diving into the Wagtail documentation's section on model inheritance, specifically under “Page Models”. In the Django world, I highly recommend "Two Scoops of Django" by Daniel Roy Greenfeld and Audrey Roy Greenfeld. It is an excellent resource that deals with best practices related to Django and by extension, Wagtail. Regarding image handling specifically, the Wagtail documentation is often the best place to be; it is constantly evolving with the project.

In summary, creating child pages with images is well within Wagtail’s capabilities. By leveraging Django's model inheritance, you can build structured content hierarchies with ease. However, the design choices you make related to your content model can greatly impact your application's maintainability and performance. A well-planned page model hierarchy based on real-world requirements will be incredibly useful. It’s always a good idea to test your approach, particularly performance wise, as your content scales.

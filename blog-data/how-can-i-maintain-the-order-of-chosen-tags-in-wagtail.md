---
title: "How can I maintain the order of chosen tags in Wagtail?"
date: "2024-12-23"
id: "how-can-i-maintain-the-order-of-chosen-tags-in-wagtail"
---

Alright, let’s tackle this. I’ve bumped into this very issue a few times, most notably when building a content management system for a national museum. They were particularly sensitive to tag order – imagine an exhibit where “Ancient Rome” *always* had to come before “Roman Empire,” for example. Getting that to stick within Wagtail required a bit more thought than I initially anticipated, but it's certainly achievable. The default behavior of Wagtail's tag fields doesn't inherently guarantee order preservation; it's typically treated as a set, not a sequence. This means that when you save content with tags, and then retrieve them, you're not guaranteed that the order they were entered will be maintained. That’s something we need to address programmatically.

My experience has shown that we essentially need to introduce a 'sortable' aspect to the relationship between the content page and its associated tags. This usually boils down to manipulating how the underlying many-to-many relationship is managed. We can't just flip a switch in Wagtail; we must approach it by customizing the relationship model, and this can be accomplished using Wagtail's flexible model system, specifically by leveraging `OrderedManyToMany`. We'll avoid simply hacking into the core models since that would be a nightmare for long-term maintainability and upgrades.

The key is to move from a standard many-to-many field to a custom intermediary table. Think of it this way: instead of the tags floating in an unordered cloud attached to your content, we will use a table that also stores the order of tag association. This will let us control the retrieval of tags.

Let's look at some code examples. First, I'll present the base models that often create this issue:

```python
from django.db import models
from modelcluster.fields import ParentalManyToManyField
from wagtail.models import Page
from wagtail.snippets.models import register_snippet

@register_snippet
class Tag(models.Model):
    name = models.CharField(max_length=255, unique=True)

    def __str__(self):
        return self.name

class ContentPage(Page):
    tags = ParentalManyToManyField('core.Tag', blank=True)

    content_panels = Page.content_panels + [
        FieldPanel('tags'),
    ]
```

This is likely where you are starting. The `ParentalManyToManyField` creates a simple, unordered relationship. We need more control. Here's how we modify the models to preserve tag order, using the `OrderedManyToMany` from `modelcluster`:

```python
from django.db import models
from wagtail.models import Page
from wagtail.snippets.models import register_snippet
from modelcluster.fields import ParentalKey, OrderedManyToMany
from modelcluster.models import ClusterableModel
from wagtail.admin.edit_handlers import FieldPanel, InlinePanel
from wagtail.snippets.edit_handlers import SnippetChooserPanel

@register_snippet
class Tag(models.Model):
    name = models.CharField(max_length=255, unique=True)

    def __str__(self):
        return self.name

class ContentPageTag(models.Model):
    page = ParentalKey('ContentPage', related_name='tagged_items', on_delete=models.CASCADE)
    tag = models.ForeignKey('core.Tag', related_name='+', on_delete=models.CASCADE)
    sort_order = models.IntegerField(null=True, blank=True, editable=False)

    panels = [
        SnippetChooserPanel('tag'),
    ]

    class Meta:
        ordering = ['sort_order']

class ContentPage(Page, ClusterableModel):
    tags = OrderedManyToMany(Tag, through=ContentPageTag, related_name='tagged_pages')

    content_panels = Page.content_panels + [
       InlinePanel('tagged_items', label="Tags"),
    ]

    def get_tags(self):
      return [item.tag for item in self.tagged_items.all()]
```

Notice how the `ContentPage` now uses `OrderedManyToMany` and defines an intermediary model, `ContentPageTag`. `ContentPageTag` includes a `sort_order` field, and importantly, it uses an `InlinePanel` in `ContentPage` rather than a `FieldPanel`.  This allows Wagtail to manage the order. `InlinePanel` also lets editors manage the tags and their ordering directly within the page editor UI. The method `get_tags()` will return the tags based on the order set within the admin.

Now, let’s look at a slightly different variation that would allow for using the `FieldPanel` instead of `InlinePanel`. This comes with a small caveat: the ordering is only maintained within the database and during the retrieval, but isn't directly editable within the field in the admin panel. This is a trade-off we can accept in some circumstances where we don't need the inline editor experience and we need it to maintain the order in templates:

```python
from django.db import models
from wagtail.models import Page
from wagtail.snippets.models import register_snippet
from modelcluster.fields import ParentalKey, OrderedManyToMany
from wagtail.admin.edit_handlers import FieldPanel

@register_snippet
class Tag(models.Model):
    name = models.CharField(max_length=255, unique=True)

    def __str__(self):
        return self.name

class ContentPageTag(models.Model):
    page = ParentalKey('ContentPage', related_name='tagged_items', on_delete=models.CASCADE)
    tag = models.ForeignKey('core.Tag', related_name='+', on_delete=models.CASCADE)
    sort_order = models.IntegerField(null=True, blank=True, editable=False)

    class Meta:
        ordering = ['sort_order']

class ContentPage(Page):
    tags = OrderedManyToMany(Tag, through=ContentPageTag, related_name='tagged_pages')

    content_panels = Page.content_panels + [
       FieldPanel('tags'),
    ]

    def get_tags(self):
      return [item.tag for item in self.tagged_items.all()]
```

The main difference here is using a `FieldPanel` which looks like the standard interface, but internally it relies on the `OrderedManyToMany` to maintain order on retrieval. This is suitable when you don't need to control the order via the edit panel itself, but want the order to be consistent for display.

It's important to note that implementing these solutions requires a database migration. You will need to migrate from the simple many-to-many to the intermediary table approach. Django’s migration system will likely assist here, but make sure you have a backup or are working in a test environment first. Also, be mindful of the `related_name` arguments. When using `OrderedManyToMany`, it's crucial to define `related_name` for both forward and reverse relationships. This is needed to retrieve the tags and also reference the `tagged_pages`.

When it comes to resources, I’d suggest looking at the Wagtail documentation surrounding model relationships. Specifically pay attention to the section on “using modelcluster”. Also, check out the `modelcluster` library documentation. Both will help clarify how to create such relations. For a deeper understanding of database relationships in general, the book “Database Internals: A Deep Dive into How Databases Work” by Alex Petrov is a good choice. While not Wagtail-specific, understanding the underlying concepts will improve how you manage database relations.

One final thing to consider: when displaying tags on your templates, always use the custom `get_tags()` method, as the field name (`.tags`) will not return the tags in order. This ensures that the order you set is correctly rendered. Always test your templates to confirm you are displaying your tags how you wish them.

In practice, I’ve found that the intermediate model approach with `InlinePanel` provides the most flexibility, while the `FieldPanel` variation can be a simpler approach when order control is not needed from within the edit panel, just on retrieval in the templates. Both maintain the desired order you've specified. The museum content team found the inline panel approach very intuitive and it allowed them to manage those important distinctions between terms like “Ancient Rome” and “Roman Empire” effectively. This issue might seem small, but attention to such details often contributes greatly to a satisfying and easy-to-use content editing workflow.

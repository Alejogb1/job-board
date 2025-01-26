---
title: "What snippet issues arose after upgrading Wagtail from 1.13 to 2.0?"
date: "2025-01-26"
id: "what-snippet-issues-arose-after-upgrading-wagtail-from-113-to-20"
---

Wagtail's upgrade from 1.13 to 2.0 introduced significant changes to how snippets, particularly those involving complex relationships and custom models, are handled. This transition revealed several issues, primarily stemming from alterations in the model inheritance, the move towards Django's `on_delete` cascade behavior, and the way Wagtail’s admin interface manages inline panels with foreign key relationships. My experience with a large content-heavy site highlighted these problem areas quite explicitly.

The primary shift that caused a cascade of snippet-related issues was the change in how Wagtail handles model inheritance and relationships. In Wagtail 1.13, a relatively permissive approach often allowed implicit behaviors, especially around foreign keys and on-delete actions. However, Wagtail 2.0 adopted a stricter, more explicit model configuration which closely aligns with Django’s standard practices. This transition, while beneficial for long-term maintainability, required a complete audit of our snippet models and the relationships to other models, leading to several errors that hadn’t been previously apparent. Previously, deleting a parent page might not trigger a cascade of deletions through many connected snippet objects, a scenario that was inconsistent. Under 2.0, deleting parent objects would likely orphan (or more correctly, error) child snippet objects due to the now-required configuration, thus revealing previous flaws in our object model.

The first critical area of concern centered around `ForeignKey` relationships without specified `on_delete` behaviors. In version 1.13, a `ForeignKey` field could implicitly assume an action, which might have been suitable for simple cases, but was dangerous for data integrity. Upon upgrading, Django’s default behavior kicked in, specifically, `CASCADE` for non-nullable fields if not explicitly set within Wagtail's model configuration. This resulted in unexpected cascade deletions when deleting pages or other related objects, often without warning. I observed situations where deleting a page would erroneously delete related snippets that were also associated with other pages or snippets because of the inferred relationship behavior.

Another frequent problem arose with inline panels within Wagtail’s admin. Inline panels are a common method to manage relationships such as one-to-many, often involving snippets. In version 1.13, custom managers for related models were sometimes overlooked or inconsistently applied within the inline edit interfaces. Upgrading to 2.0 revealed cases where we had attempted to use methods of related managers that Wagtail had no knowledge of. For instance, our previous attempts at ordering associated snippets within an inline panel by a custom attribute using a manager's ordering function were no longer consistently respected because the internal mechanisms for inline forms changed considerably. This often rendered previously usable interface components now displaying in incorrect orders or with faulty rendering.

A further issue arose with changes in how Wagtail rendered snippets with complex `StreamFields`. While `StreamFields` functionality remained largely the same, our custom block implementations often relied on outdated patterns. For instance, our snippets might contain custom blocks referencing related snippets or page objects, and the rendering logic for such blocks had to be rigorously re-examined to ensure they now correctly used `get_template` calls, as the block structures changed. A notable side-effect of this issue was that previously stored data within the `StreamField` would error on rendering when accessed through an old path.

To illustrate these issues with concrete examples, consider these scenarios:

**Example 1: ForeignKey without `on_delete`**

Prior to the upgrade, the following snippet model might have functioned, but created a hidden pitfall:

```python
from django.db import models
from wagtail.snippets.models import register_snippet

@register_snippet
class Author(models.Model):
    name = models.CharField(max_length=255)

    def __str__(self):
        return self.name

@register_snippet
class Book(models.Model):
    title = models.CharField(max_length=255)
    author = models.ForeignKey(Author, related_name='books', null=True, blank=True) # Issue here

    def __str__(self):
        return self.title
```

After the upgrade, when deleting an `Author`, any linked `Book` objects would throw a database error because Django required an explicit `on_delete` action for non-nullable foreign keys. The solution was to explicitly define the intended behavior, such as `on_delete=models.SET_NULL` or `on_delete=models.CASCADE`. The following modification shows how we addressed this:

```python
from django.db import models
from wagtail.snippets.models import register_snippet

@register_snippet
class Author(models.Model):
    name = models.CharField(max_length=255)

    def __str__(self):
        return self.name

@register_snippet
class Book(models.Model):
    title = models.CharField(max_length=255)
    author = models.ForeignKey(Author, related_name='books', null=True, blank=True, on_delete=models.SET_NULL)

    def __str__(self):
        return self.title
```

**Example 2: InlinePanel with Custom Manager**

Consider an inline panel designed to link a series of `Book` objects to a `Publisher` snippet. We might have employed a custom manager to order the books within this inline panel before the upgrade, relying on custom functionality that was no longer respected in 2.0:

```python
from django.db import models
from wagtail.snippets.models import register_snippet
from wagtail.admin.edit_handlers import InlinePanel

class BookManager(models.Manager):
    def get_queryset(self):
        return super().get_queryset().order_by('title')

@register_snippet
class Publisher(models.Model):
    name = models.CharField(max_length=255)

    def __str__(self):
        return self.name

@register_snippet
class Book(models.Model):
    title = models.CharField(max_length=255)
    publisher = models.ForeignKey(Publisher, related_name='books', on_delete=models.CASCADE)

    objects = BookManager()

    def __str__(self):
        return self.title

from wagtail.admin.edit_handlers import FieldPanel, InlinePanel
from wagtail.snippets.models import register_snippet

@register_snippet
class Publisher(models.Model):
    name = models.CharField(max_length=255)

    panels = [
        FieldPanel('name'),
        InlinePanel('books', label="Books")
    ]


    def __str__(self):
        return self.name
```

While `BookManager` worked for querying books directly, the `InlinePanel` did not automatically respect the ordering from the custom manager. This required modifying the `InlinePanel` usage by adding the `orderable` attribute, to specify what field was used for ordering within the inline panel. The solution was to explicitly set an `orderable` field:

```python
from django.db import models
from wagtail.snippets.models import register_snippet
from wagtail.admin.edit_handlers import InlinePanel

class BookManager(models.Manager):
    def get_queryset(self):
        return super().get_queryset().order_by('title')

@register_snippet
class Publisher(models.Model):
    name = models.CharField(max_length=255)

    def __str__(self):
        return self.name

@register_snippet
class Book(models.Model):
    title = models.CharField(max_length=255)
    publisher = models.ForeignKey(Publisher, related_name='books', on_delete=models.CASCADE)
    # Added orderable for use in inline panel
    order = models.IntegerField(null=True, blank=True)

    objects = BookManager()

    def __str__(self):
        return self.title

from wagtail.admin.edit_handlers import FieldPanel, InlinePanel
from wagtail.snippets.models import register_snippet

@register_snippet
class Publisher(models.Model):
    name = models.CharField(max_length=255)

    panels = [
        FieldPanel('name'),
        InlinePanel('books', label="Books", orderable_field="order")
    ]


    def __str__(self):
        return self.name

```

**Example 3: StreamField Block Rendering**

Our custom StreamField blocks previously relied on hardcoded paths for associated templates. When Wagtail upgraded, these paths broke, resulting in errors when rendering these custom blocks:

```python
from wagtail.core import blocks
from wagtail.snippets.blocks import SnippetChooserBlock

class CustomSnippetBlock(blocks.StructBlock):
    snippet = SnippetChooserBlock('app_name.MyCustomSnippet')

    def render(self, value, **kwargs):
        snippet_instance = value.get('snippet')
        # incorrect rendering path and no attempt to resolve dynamic template
        return render_to_string('custom_app/blocks/my_snippet_template.html', {'snippet': snippet_instance})
```

The correct approach was to use Wagtail's template loading mechanism by creating our own template within a templates directory and having Wagtail automatically pick it up:

```python
from wagtail.core import blocks
from wagtail.snippets.blocks import SnippetChooserBlock

class CustomSnippetBlock(blocks.StructBlock):
    snippet = SnippetChooserBlock('app_name.MyCustomSnippet')

    def get_template(self, parent_context=None):
      return 'custom_app/blocks/my_snippet_template.html'
```

These examples demonstrate the types of issues encountered during the Wagtail 1.13 to 2.0 upgrade. Effectively addressing these challenges required a comprehensive review of all snippet models and how they interacted with other models using a combination of Django’s ORM and Wagtail’s admin tools. This also included carefully reviewing all custom block implementations.

For resource recommendations, I would suggest the official Wagtail documentation, especially the release notes for version 2.0 and subsequent updates. Pay close attention to the sections dealing with model inheritance, ForeignKey fields, and inline panel configurations. The Django documentation, particularly the areas about model fields and `on_delete` behavior, are also invaluable. Finally, exploring examples of Wagtail snippet implementations on public repositories can offer valuable insights into best practices and common configurations. Reviewing the changelog for the Django framework versions that Wagtail uses also helped in identifying changes that had indirect effects on Wagtail's snippet management. Careful study of these resources greatly assisted in rectifying the issues encountered during the migration process.

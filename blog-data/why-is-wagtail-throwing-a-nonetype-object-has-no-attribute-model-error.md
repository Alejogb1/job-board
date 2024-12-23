---
title: "Why is Wagtail throwing a 'NoneType' object has no attribute 'model' error?"
date: "2024-12-23"
id: "why-is-wagtail-throwing-a-nonetype-object-has-no-attribute-model-error"
---

Let's unpack this. The "NoneType object has no attribute 'model'" error in Wagtail, particularly when you’re dealing with StreamFields or related components, is a classic sign of a misconfigured relationship or a prematurely accessed attribute. I've encountered this particular gremlin more than once during my time developing content management systems, and each time, the root cause has revolved around similar patterns, though the specifics vary.

Basically, this error crops up when Wagtail is trying to access a related model instance or a model property via a relationship where that model instance hasn't been properly instantiated or has returned `None` unexpectedly. It's often a result of the way Wagtail handles dynamic content structures, particularly when using streamfields, where content blocks may reference other models or be dependent on external data.

The most common situation is where your StreamField block, or the logic that consumes it, is attempting to access a model associated with a block *before* the block is fully rendered or has its associated data available. Wagtail's streamfield is powerful because it allows for flexible content creation, but this flexibility introduces the potential for timing issues if data dependencies aren't carefully managed.

From experience, there are three scenarios that repeatedly trigger this error in my projects: first, accessing a related model too early within a custom block's template; second, failing to handle null values when retrieving related data within a block's `get_context` method; and third, issues relating to the order in which blocks are initialized and rendered when using nested streamfields.

Let’s delve into some code examples to illustrate these cases.

**Scenario 1: Early Access in Templates**

Imagine you have a custom streamfield block, `FeaturedArticleBlock`, designed to display a summary of a related article.

```python
# models.py
from django.db import models
from wagtail.core import blocks
from wagtail.core.fields import StreamField
from wagtail.admin.edit_handlers import FieldPanel, StreamFieldPanel

class Article(models.Model):
    title = models.CharField(max_length=255)
    summary = models.TextField()

class FeaturedArticleBlock(blocks.StructBlock):
    article = blocks.PageChooserBlock() # Assuming a page that implements models.Article

    class Meta:
        template = "blocks/featured_article.html"

class HomePage(models.Model): # Simplified Home page for illustration
    content = StreamField([
        ('featured_article', FeaturedArticleBlock()),
    ], use_json_field=True, blank=True)
    panels = [
        StreamFieldPanel('content'),
    ]

```

Now, in the `featured_article.html` template, we might have something like this:

```html
<!-- blocks/featured_article.html -->
<h2>{{ block.value.article.title }}</h2>
<p>{{ block.value.article.summary }}</p>
```

This looks correct at first glance, but here's the problem: `block.value.article` returns a `Page` object, *not* an `Article` instance, unless a custom model was implemented to serve the pages. If you were to directly access a custom model's fields without correctly referencing that, or if the selected page doesn't implement models.Article, the `title` and `summary` attributes won't be directly available, causing the "NoneType object has no attribute 'model'" error. The proper way, assuming the selected page *does* implement the model, is to access the specific model associated with the page using `specific`:

```html
<!-- blocks/featured_article.html Corrected-->
{% with article_page=block.value.article.specific %}
    <h2>{{ article_page.title }}</h2>
    <p>{{ article_page.summary }}</p>
{% endwith %}
```
Here we are using `specific` to get the relevant Article model associated with the selected page, *after* it has been fully populated within the view rendering process. If the specific model is `None`, which could occur if the selected page doesn't implement an Article type, that would be a separate issue, but it would avoid the `NoneType` error I'm describing. This demonstrates how improper access within templates can readily trigger the error due to Wagtail’s page tree structure and polymorphism.

**Scenario 2: Missing Null Handling in `get_context`**

Sometimes, the error comes from within the block's `get_context` method. Let's say we want to provide more context to our `FeaturedArticleBlock`:

```python
# models.py (Continuing with the same example)
class FeaturedArticleBlock(blocks.StructBlock):
    article = blocks.PageChooserBlock()
    show_author = blocks.BooleanBlock(required=False, default=False)

    class Meta:
        template = "blocks/featured_article_extended.html"

    def get_context(self, value, parent_context=None):
        context = super().get_context(value, parent_context)
        selected_page = value.get('article') # or value['article'], both work here

        if selected_page: # Handle case where a page might not be selected
            article_page = selected_page.specific
            context['article_title'] = article_page.title
            context['article_summary'] = article_page.summary

            if value.get('show_author', False):
                # Simulate a related author relationship
                # A real app might fetch from the author page or through a foreign key
                author_name = getattr(article_page, 'author', None)  # Avoid AttributeError
                if author_name:
                    context['article_author'] = author_name
        return context
```

And the corresponding template `blocks/featured_article_extended.html`:

```html
<!-- blocks/featured_article_extended.html -->
<h2>{{ article_title }}</h2>
<p>{{ article_summary }}</p>
{% if article_author %}
<p>By {{ article_author }}</p>
{% endif %}
```

The vulnerability here lies in the fact that `value.get('article')` can return `None` if no article is selected in the StreamField. The `specific` call would then try to operate on a `None` object, resulting in that troublesome "NoneType" error. The conditional check `if selected_page:` along with the use of `getattr` on the `article_page` to safely check for the author avoids this. The inclusion of the `default` parameter in the `show_author` boolean block further prevents potentially raising the `KeyError` in the template if the block isn’t configured initially.

**Scenario 3: Nested StreamField Initialization**

Finally, there's the issue of nested StreamFields and initialization order. Let's use a slightly altered version:

```python
# models.py (Continuing with the same example)
class ArticleSummaryBlock(blocks.StructBlock):
    title = blocks.CharBlock()
    summary = blocks.TextBlock()

class ArticleListBlock(blocks.StructBlock):
    articles = blocks.StreamBlock([
        ('article_summary', ArticleSummaryBlock())
    ])

    class Meta:
       template = "blocks/article_list.html"
    def get_context(self, value, parent_context=None):
        context = super().get_context(value, parent_context)
        context['articles_data'] = value['articles'] # No specific model here
        return context


class HomePage(models.Model):
    content = StreamField([
        ('article_list', ArticleListBlock()),
    ], use_json_field=True, blank=True)
    panels = [
        StreamFieldPanel('content'),
    ]
```

And the template `blocks/article_list.html`:
```html
<!-- blocks/article_list.html -->
{% for article in articles_data %}
<div>
    <h2>{{ article.value.title }}</h2>
    <p>{{ article.value.summary }}</p>
</div>
{% endfor %}
```

This simplified version does not have a specific model, but it illustrates an important point. In scenarios where you have nested streamfields, ensure that the inner block’s data is appropriately accessed, because if the streamfield itself is empty, for instance, `articles_data` would still be available but the underlying access of fields such as `article.value.title` will fail if the inner streamfields haven't been properly initiated. In the above, if no 'article_summary' blocks were added into the inner 'articles' streamfield, then the loop will render nothing, but it won't raise a `NoneType` error.

The error arises when you introduce external data or complex logic inside an inner streamfield block and don't handle the case when the outer or inner streamfield data might not be ready. The `value` for each nested block is populated at different points in the rendering process. If you were attempting to fetch data based on configurations in an outer block before an inner one was fully resolved you would encounter issues. Therefore, it is important to make sure all fields and nested blocks are initiated correctly.

To properly tackle this issue, I usually suggest thoroughly reviewing the stack trace to pinpoint the line of code where the error occurs and then examining the data flow at that point. Check if `block.value` or any nested structure you’re accessing might return a `NoneType`, and use defensive programming techniques like those illustrated here to gracefully handle missing data.

For further understanding of Wagtail's internal structure and handling of relationships, I’d recommend consulting the Wagtail documentation directly; specifically, the sections on StreamFields, custom blocks, and templates. Additionally, reading "Django Unleashed" by William S. Vincent offers a solid deep dive into Django's ORM that will assist in understanding how Django and Wagtail work with databases. Also, the detailed explanations of model inheritance in "Two Scoops of Django 3.x" by Daniel Roy Greenfeld and Audrey Roy Greenfeld are helpful when trying to model and access complex relationships within Wagtail.

These are some of the common pitfalls I’ve seen, and the corresponding solutions, when dealing with this "NoneType" error. Understanding the data flow and using appropriate checks are crucial to maintaining stable Wagtail applications.

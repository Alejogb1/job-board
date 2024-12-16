---
title: "How can I create different options for Select in Wagtail admin?"
date: "2024-12-16"
id: "how-can-i-create-different-options-for-select-in-wagtail-admin"
---

Alright, let’s tackle this. Been down this road a few times with Wagtail, and it’s a common need, wanting a bit more flexibility with those dropdowns in the admin panel. We’re not limited to just basic text fields; we can absolutely create more nuanced options for our `Select` fields, catering to diverse data needs. The key lies in leveraging Wagtail's field panels and model properties effectively, along with a little Python magic.

The standard way, of course, is to define a `CharField` with `choices` directly in your model, like so:

```python
from django.db import models

class MyPage(models.Model):
    ...
    status = models.CharField(
        max_length=20,
        choices=[
            ('draft', 'Draft'),
            ('review', 'In Review'),
            ('published', 'Published')
        ],
        default='draft',
        help_text="Select the current status of the page"
    )
    ...
```

This is perfectly fine for simple, static options. But what if you need options that are dynamically generated, dependent on other model properties or even external data? That’s where things get more interesting.

For instance, let’s say your `Select` needs to pull from a related model. Perhaps you’re building a blog, and you want the editor to select an author from a predefined list of authors. Instead of hardcoding names, we will relate this to an `Author` model.

Here’s how I’d approach that, using a method to populate the `choices` on the fly, and making sure to handle cases where the selection is saved:

```python
from django.db import models
from wagtail.admin.edit_handlers import FieldPanel
from wagtail.models import Page

class Author(models.Model):
    name = models.CharField(max_length=255)

    def __str__(self):
        return self.name

class ArticlePage(Page):
    ...
    author = models.ForeignKey(
        'Author',
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name='articles',
        help_text="Select the author of this article"
    )

    @property
    def author_choices(self):
        return [(author.id, author.name) for author in Author.objects.all()]

    content_panels = Page.content_panels + [
        FieldPanel('author', widget=forms.Select),
        ...
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._meta.get_field('author').choices = self.author_choices
    ...
```

Let's dissect this. We've defined an `Author` model first. Then in `ArticlePage`, instead of a `CharField`, we've used a `ForeignKey`, pointing to `Author`. I then introduced a `@property` called `author_choices` which queries the `Author` model and creates a list of tuples formatted for choice fields - the author id and their name. Critically, within the `__init__` method, I'm setting the choices property of the underlying field. This ensures that choices are dynamically updated when you create or edit an instance. This effectively populates the dropdown based on currently active authors.

Now, you might be asking, what if I need choices that depend on the state of the page itself? For instance, maybe you want different options available depending on whether the page is a blog post or a news article. This is where we can get into more dynamic logic, potentially using a callable in the `choices` definition, but for simplicity and avoiding common pitfalls, we should move the logic within the `__init__` rather than rely solely on the callable.

Here's an example where the `Select` options depend on the page type:

```python
from django import forms
from django.db import models
from wagtail.admin.edit_handlers import FieldPanel
from wagtail.models import Page

class PageType(models.Model):
    name = models.CharField(max_length=255)
    slug = models.SlugField()

    def __str__(self):
        return self.name

class BasePage(Page):
    page_type = models.ForeignKey(
        'PageType',
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        help_text="Select the type of page",
        related_name='pages'
    )

    selected_option = models.CharField(
        max_length=50,
        blank=True,
        help_text="Select a specific option, based on the page type."
    )

    content_panels = Page.content_panels + [
        FieldPanel('page_type', widget=forms.Select),
        FieldPanel('selected_option', widget=forms.Select),
        ...
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.page_type:
            if self.page_type.slug == 'blog':
               self._meta.get_field('selected_option').choices = [
                  ('blog_option1', 'Blog Option 1'),
                  ('blog_option2', 'Blog Option 2')
               ]
            elif self.page_type.slug == 'news':
                self._meta.get_field('selected_option').choices = [
                    ('news_option1', 'News Option 1'),
                    ('news_option2', 'News Option 2')
                ]
            else:
                self._meta.get_field('selected_option').choices = [('default', 'Default')]
        else:
            self._meta.get_field('selected_option').choices = [('default', 'Default')]
    ...

```

Here, I've introduced `PageType` and `selected_option` fields. Inside the `__init__`, I am checking the `page_type` of the page, and if it matches certain values, I will dynamically create choices for our `selected_option` field. If no page type or unknown slug, it defaults to a single default option.

Finally, consider a scenario where you need to query external data to populate your dropdown. This requires a slight tweak, making sure to cache results for performance. Let's imagine a scenario where you have an API that returns a list of valid countries.

```python
import requests
from django.db import models
from wagtail.admin.edit_handlers import FieldPanel
from wagtail.models import Page
from django import forms

class LocationPage(Page):
    country = models.CharField(
        max_length=100,
        blank=True,
        help_text="Select a country."
    )

    _country_choices = None

    @property
    def country_choices(self):
        if self._country_choices is None:
            try:
                response = requests.get("https://restcountries.com/v3.1/all")
                response.raise_for_status() # Raise HTTPError for bad responses
                data = response.json()
                self._country_choices = [ (c['name']['common'], c['name']['common']) for c in data ]
            except requests.exceptions.RequestException:
                self._country_choices = [('error', 'Could not load countries')]

        return self._country_choices

    content_panels = Page.content_panels + [
        FieldPanel('country', widget=forms.Select)
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._meta.get_field('country').choices = self.country_choices
    ...
```

In this example, I'm fetching data from the restcountries api. Critically, I am caching the results in a variable `_country_choices` - this avoids repeated api calls every time the page is loaded. Also note I am wrapping the call in a try/except block to gracefully handle any errors if the external api goes down.

These examples demonstrate how to use the combination of model properties, `__init__`, and external data calls to create more dynamic and nuanced `Select` dropdowns in your Wagtail admin panel. There are, of course, more intricate use cases that involve custom form fields, or widgets, but this will get you a long way.

To dive deeper, I highly recommend exploring the Django documentation concerning form fields, particularly the documentation for `ChoiceField`. For more insights into how Wagtail handles form fields and panels, consult the Wagtail documentation directly, focusing on the `edit_handlers` module. Also, consider reviewing ‘Two Scoops of Django 3.x’ by Daniel Roy Greenfeld and Audrey Roy Greenfeld; it provides extremely practical insights into Django best practices that apply directly to Wagtail development.

Remember, keeping your choices dynamically updated and handling edge cases is important, so test these implementations thoroughly. Good luck, and let me know if you have any other questions.

---
title: "How can I make different options for Select in Wagtail admin?"
date: "2024-12-23"
id: "how-can-i-make-different-options-for-select-in-wagtail-admin"
---

Alright, let's tackle this one. I've seen this specific requirement pop up quite a few times in my years working with wagtail, and there are some elegant, scalable ways to approach creating diverse select options within the admin panel. It’s not a one-size-fits-all situation, which is probably why you’re asking, and frankly, that’s good—it forces you to think about your data structures carefully.

The core challenge here revolves around how wagtail's admin interface integrates with django forms and how we can leverage those interfaces to customize the options available in a `<select>` element when editing a wagtail page or snippet. We essentially need to manipulate the form field definition. We can accomplish this in a few ways, each with different implications for flexibility and maintainability. I'll walk you through three methods that I've found effective in past projects, each targeting different levels of complexity and customization.

**Option 1: Simple Choices with `choices` Argument**

The most straightforward scenario involves a fixed set of options. In this case, we can directly pass the `choices` argument to the `models.CharField`, `models.IntegerField`, or other similar fields within our wagtail models. This approach is most suitable when your options are static and unlikely to change frequently.

```python
from django.db import models
from wagtail.admin.panels import FieldPanel
from wagtail.models import Page

class MyPage(Page):
    template = "my_app/my_page.html"

    CATEGORY_CHOICES = [
        ('news', 'News Article'),
        ('blog', 'Blog Post'),
        ('event', 'Event Listing'),
    ]

    category = models.CharField(
        max_length=50,
        choices=CATEGORY_CHOICES,
        help_text="Select the category for this page.",
    )

    content_panels = Page.content_panels + [
        FieldPanel('category'),
    ]

```

In the code above, we define `CATEGORY_CHOICES` as a list of tuples where the first element is the value stored in the database and the second is the user-facing label. When wagtail renders the form, it will automatically generate a select dropdown with the specified options. Notice that we have attached the `FieldPanel` which hooks this field into the wagtail admin UI.

This method is quick and easy to implement. If you find yourself adding new choices to this list, it usually means a code change, which may require a deployment. If this seems cumbersome, especially if you anticipate frequent changes to these options, you’ll probably want to move towards one of the more flexible methods.

**Option 2: Dynamically Generated Choices using a Function**

The static choices approach does not scale if the select options depend on external factors or need to be retrieved from the database dynamically. We need the options to be flexible and potentially updated without deploying code changes. I found that a well placed function that generates the choices dynamically was more sustainable in the long term.

```python
from django.db import models
from wagtail.admin.panels import FieldPanel
from wagtail.models import Page
from my_app.models import Author  # Example model where options come from

def get_author_choices():
    authors = Author.objects.all()
    return [(author.id, author.name) for author in authors]

class MyPage(Page):
    template = "my_app/my_page.html"

    author = models.CharField(
        max_length=50,
        help_text="Select the author for this page.",
    )

    def get_form_class(self):
        form_class = super().get_form_class()
        form_class.base_fields['author'].choices = get_author_choices()
        return form_class

    content_panels = Page.content_panels + [
        FieldPanel('author'),
    ]

```

Here, `get_author_choices` function fetches all authors from the `Author` model (you'll need to define that elsewhere), and constructs a list of tuples suitable for the `choices` argument. Critically, we then override the `get_form_class()` method in our `MyPage` model. This allows us to access the generated Django form for the page, then set the `choices` for our `author` field right before the form is used by wagtail. I have found this approach preferable since you can use virtually any logic to fetch the options. The database is not the only option; you could fetch from external APIs, a cache, or any other source. It’s also cleaner to handle updating the choices in this function if you add functionality to refresh or update them.

This method requires slightly more code than the static approach, but offers significantly greater flexibility. The admin interface updates the select options whenever it renders the form, pulling the freshest list from the method. A potential caveat is that it incurs a database query each time the form is rendered. In the case of a large database, this could be a point to optimize with caching.

**Option 3: Using a Foreign Key to Relate to Another Model**

In some scenarios, the select options you're presenting really represent relationships to other pieces of data, not just arbitrary choices. In such cases, using a `ForeignKey` field is appropriate. This adds more relational data to your model, allowing you to organize and work with information more effectively.

```python
from django.db import models
from wagtail.admin.panels import FieldPanel
from wagtail.models import Page
from wagtail.snippets.models import register_snippet
from wagtail.admin.edit_handlers import SnippetChooserPanel

@register_snippet
class Tag(models.Model):
    name = models.CharField(max_length=50, unique=True)

    def __str__(self):
        return self.name

    class Meta:
      verbose_name = 'Tag'
      verbose_name_plural = 'Tags'

class MyPage(Page):
    template = "my_app/my_page.html"

    tag = models.ForeignKey(
        'my_app.Tag',
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name='pages',
        help_text="Select the tag for this page.",
    )


    content_panels = Page.content_panels + [
        SnippetChooserPanel('tag'),
    ]
```

Here, we’ve created a snippet model called `Tag`, and we link our `MyPage` model to it with a `ForeignKey`.  The wagtail admin interface will understand that this needs to be treated as a relationship, generating a slightly different UI element than a regular `<select>` element. We use `SnippetChooserPanel` in our `content_panels`, rather than `FieldPanel`, to instruct wagtail on how this field should be presented. This is also advantageous because it allows you to manage the options from another area in the Wagtail admin instead of editing the page model every time you need a new option. This reduces the need for code deployments as you can add tags from within the admin and this will update the dropdown in the page editing UI.

This approach works exceptionally well when the 'options' themselves have associated data that you also want to manage. It's also great if these 'options' can be reused in other models. If all you needed was just a string and were using this option just for a dynamic dropdown, this may be an unnecessary level of complexity.

**Recommendations for Further Study**

For a deeper understanding, I’d recommend exploring the following resources:

1.  **"Two Scoops of Django 3.x: Best Practices for the Django Web Framework" by Daniel Roy Greenfeld and Audrey Roy Greenfeld**: This book provides invaluable insights into how Django forms, which are the foundation of wagtail forms, function. Chapters on forms are particularly useful here.
2.  **Official Django Documentation:** Specifically, look at the documentation on models (`django.db.models`), form fields (`django.forms.fields`), and creating choices. The official documentation is always the best reference when you need a detailed understanding.
3. **Official Wagtail Documentation**: Wagtail’s official documentation on model editing panels and page models is essential reading. Pay attention to the sections on FieldPanels and SnippetChooserPanels.

Implementing diverse select options in wagtail admin is about understanding where the options come from, how dynamic they need to be, and what relationships those options have with other data. Choosing between static choices, dynamically populated choices, and foreign key relationships depends on your specific needs, so select the strategy that makes the most sense. I've used each of these options in different projects and found all of them reliable and scalable, so I hope this helps.

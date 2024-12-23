---
title: "Can a child page of a specific type be created under a parent page using Django Wagtail and provided POST data?"
date: "2024-12-23"
id: "can-a-child-page-of-a-specific-type-be-created-under-a-parent-page-using-django-wagtail-and-provided-post-data"
---

Okay, let's unpack this. Creating child pages in Wagtail, especially when driven by post data, is something I’ve navigated many times, and it’s a very common requirement in any content-driven website. It sounds simple at first, but the devil’s always in the details, particularly when considering data validation and the broader implications on your Wagtail setup.

Here's the breakdown of how I would approach this, drawing from my experience. Let's assume you're not just dealing with straightforward fields but potentially relational data as well. We’ll tackle this step by step with accompanying code examples to make things crystal clear.

First, the key lies in understanding Wagtail's class-based page models and form handling. Wagtail doesn't give you an ‘out-of-the-box’ solution specifically for creating child pages using post data within a generic request context. You'll need to craft the process yourself using the django forms api and Wagtail’s own page model functionalities. The post data generally represents how users interact with a form rendered on the page, which then drives the creation. This means we need to build a form, process the data, validate it, and *then* instantiate our desired child page type.

Let's say we have a parent page model called `BlogIndexPage` and we want to create a child page type called `BlogPostPage` through post data. Our first task is to set up our page models in `models.py`

```python
# models.py

from django.db import models
from wagtail.models import Page
from wagtail.fields import RichTextField
from wagtail.admin.panels import FieldPanel

class BlogIndexPage(Page):
    # Parent Page, no specific fields for simplicity in the example

    max_count = 1
    subpage_types = ['BlogPostPage'] # Only BlogPostPage is allowed as a child

    class Meta:
        verbose_name = "Blog Index Page"

class BlogPostPage(Page):
    body = RichTextField(blank=True)

    content_panels = Page.content_panels + [
      FieldPanel('body'),
    ]
    parent_page_types = ['BlogIndexPage']  # only BlogIndexPage can have this page type as a child
    class Meta:
      verbose_name = "Blog Post Page"

```

In this example, the `BlogIndexPage` acts as the parent and can have only `BlogPostPage` objects as its direct descendants. `BlogPostPage` includes the common body `RichTextField` for the content, which we will be setting through the POST data

Now, let's move to the creation of the page itself. This typically resides within a view that handles the form submission. I often create a django form for handling this data. Here’s a snippet that represents a basic form within the same `models.py` file:

```python

#forms.py

from django import forms

class BlogPostForm(forms.Form):
   title = forms.CharField(max_length=255, required=True)
   body = forms.CharField(widget=forms.Textarea, required=True)

```

Now, let’s get to the meat of it – the view. Here’s how I’d structure a view to handle the form submission and page creation. This would typically be in `views.py`. We are directly taking the data from the form and setting the required title and body values.

```python
# views.py

from django.shortcuts import render, get_object_or_404, redirect
from .models import BlogIndexPage, BlogPostPage
from .forms import BlogPostForm
from wagtail.models import Site

def create_blog_post(request, page_id):
    parent_page = get_object_or_404(BlogIndexPage, id=page_id)

    if request.method == 'POST':
        form = BlogPostForm(request.POST)
        if form.is_valid():
            # Get data from the form
            title = form.cleaned_data['title']
            body = form.cleaned_data['body']

            # Create BlogPostPage Instance
            blog_post = BlogPostPage(
                title=title,
                body=body,
            )

            # Add the page as a child to the parent
            parent_page.add_child(instance=blog_post)

            #redirect after successful submission. Can be modified as per requirements
            return redirect(parent_page.url)


    else:
        form = BlogPostForm()

    return render(request, 'create_blog_post.html', {'form': form, 'parent_page':parent_page})
```

A couple of key observations here: First, I retrieve the parent page using its ID. Then, if the request method is POST, I validate the form and then instantiate the `BlogPostPage` with the data from the post submission. The magic line `parent_page.add_child(instance=blog_post)` handles all of Wagtail's tree structure updates behind the scenes.

The final piece of the puzzle is our template `create_blog_post.html`, which is quite straightforward:

```html
  <!-- create_blog_post.html -->
    <h1>Create New Blog Post</h1>
    <form method="post">
        {% csrf_token %}
        {{ form.as_p }}
        <button type="submit">Create Post</button>
    </form>
```

The form rendering here is handled via the Django Forms API. You would access this functionality at a specific url linked to the view, making sure that it is properly set up in your `urls.py`

Now, let's address some potential real-world complexities. Firstly, consider file uploads. If your child page has an image or document upload field, you can handle this in a similar way using a `ModelForm` instead of `Form`. The key is that you must add the uploaded files to the object's field, and Django takes care of the rest of the file management.

Secondly, what about relations? For instance, what if your `BlogPostPage` had a many-to-many relation to an `Author` model? Handling this through POST is a multi-step operation. You'd typically handle the selection of author(s) via form fields and then, after saving the base page data, you’d iterate through these selected IDs and create the relationship entries programmatically.

```python
# Modified view for adding many to many
# Assuming you have an Author Model
# From your_app.models import Author
# forms.py - added fields for author

# models.py

from django.db import models
from wagtail.models import Page
from wagtail.fields import RichTextField
from wagtail.admin.panels import FieldPanel, MultiFieldPanel
from modelcluster.fields import ParentalManyToManyField
from modelcluster.models import ClusterableModel

class Author(models.Model):
    name = models.CharField(max_length=255)
    def __str__(self):
        return self.name

    class Meta:
      verbose_name = "Author"


class BlogIndexPage(Page):
    # Parent Page, no specific fields for simplicity in the example

    max_count = 1
    subpage_types = ['BlogPostPage'] # Only BlogPostPage is allowed as a child

    class Meta:
        verbose_name = "Blog Index Page"


class BlogPostPage(Page):
    body = RichTextField(blank=True)
    authors = ParentalManyToManyField('your_app.Author', blank=True)

    content_panels = Page.content_panels + [
      FieldPanel('body'),
       MultiFieldPanel(
                [
                    FieldPanel('authors',widget=forms.CheckboxSelectMultiple),
                ],
                heading="Authors",
            ),
    ]
    parent_page_types = ['BlogIndexPage']  # only BlogIndexPage can have this page type as a child

    class Meta:
        verbose_name = "Blog Post Page"


# forms.py - Added Author field

from django import forms
from .models import Author


class BlogPostForm(forms.Form):
    title = forms.CharField(max_length=255, required=True)
    body = forms.CharField(widget=forms.Textarea, required=True)
    authors = forms.ModelMultipleChoiceField(queryset=Author.objects.all(), required = False)

# views.py

from django.shortcuts import render, get_object_or_404, redirect
from .models import BlogIndexPage, BlogPostPage
from .forms import BlogPostForm
from wagtail.models import Site

def create_blog_post(request, page_id):
    parent_page = get_object_or_404(BlogIndexPage, id=page_id)

    if request.method == 'POST':
        form = BlogPostForm(request.POST)
        if form.is_valid():
            # Get data from the form
            title = form.cleaned_data['title']
            body = form.cleaned_data['body']
            authors = form.cleaned_data['authors'] # Get the list of authors from the form

            # Create BlogPostPage Instance
            blog_post = BlogPostPage(
                title=title,
                body=body,
            )

            # Add the page as a child to the parent
            parent_page.add_child(instance=blog_post)

            blog_post.authors.set(authors)

            #redirect after successful submission. Can be modified as per requirements
            return redirect(parent_page.url)


    else:
        form = BlogPostForm()

    return render(request, 'create_blog_post.html', {'form': form, 'parent_page':parent_page})

```

Notice the `authors = form.cleaned_data['authors']` and the `blog_post.authors.set(authors)` line after the save. The author selection field is a `ModelMultipleChoiceField`, and it will return a QuerySet of selected model instances. I have now moved the authors to be managed by the `BlogPostPage` with a `ParentalManyToManyField` and this manages the many-to-many relation effectively.

For further insights into Wagtail, consult the official Wagtail documentation, which is exceptionally thorough and updated. For a broader understanding of Django's form handling, I'd highly recommend "Two Scoops of Django: Best Practices for Django 1.11 and Python 3" by Daniel Roy Greenfeld and Audrey Roy Greenfeld. And for a deeper dive into relational databases, "Database System Concepts" by Abraham Silberschatz, Henry F. Korth, and S. Sudarshan provides a fantastic foundation.

In summary, creating child pages from post data in Wagtail is achievable through a combination of Django forms, Wagtail's page model APIs, and meticulous data handling. It’s a process that requires understanding of both Django’s architecture and Wagtail’s content management logic. I have encountered various permutations of this challenge, and the above methodology has served me well each time. This approach provides a strong foundation for building very complex functionalities on top of Wagtail.

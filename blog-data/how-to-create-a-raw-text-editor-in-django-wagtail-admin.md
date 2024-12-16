---
title: "How to create a raw text editor in Django Wagtail Admin?"
date: "2024-12-16"
id: "how-to-create-a-raw-text-editor-in-django-wagtail-admin"
---

Right then, let’s tackle this. I've spent my share of late nights battling various content management system intricacies, and creating a raw text editor directly within the Django Wagtail admin panel is a challenge I've definitely encountered—more than once, actually. It's not a standard feature out-of-the-box, naturally, but there are several ways to approach it, each with its own set of trade-offs. Let me walk you through the strategies I’ve found most effective, coupled with some practical examples.

Essentially, we're trying to bypass Wagtail's rich text editing capabilities and expose a simple text area for direct text manipulation within the admin interface. This comes in handy when you need precise control over the content, or when you want to manage code snippets, configuration files, or other structured text formats without the potential for rich text editors altering the formatting unexpectedly.

The first, and often simplest, method relies on creating a custom `FieldPanel` within Wagtail's admin settings. Here, you can replace the default rich text editor with a standard HTML `<textarea>`. While this might seem rudimentary, it’s remarkably effective for basic raw text editing. This method is most useful when you need a completely unformatted input area and you are comfortable handling text processing afterwards.

```python
# models.py

from django.db import models
from wagtail.admin.edit_handlers import FieldPanel
from wagtail.core.models import Page

class RawTextPage(Page):
    raw_text_content = models.TextField(blank=True)

    content_panels = Page.content_panels + [
        FieldPanel('raw_text_content', heading="Raw Text Content"),
    ]

```

In this snippet, we simply use `models.TextField` for our content field and then employ `FieldPanel` to create the appropriate input in the Wagtail admin. Importantly, we are using the default `FieldPanel`, which, for a `TextField`, renders a basic `textarea`. This renders a raw text input in the Wagtail admin panel. No special widgets are needed, just the `FieldPanel`. The crucial element here is `models.TextField`, which provides the appropriate underlying storage. This has limitations, though. It doesn't include features like syntax highlighting.

For more sophisticated editing experiences, particularly with structured text like code, you'll likely want to incorporate a code editor component. One straightforward way to do this is through a custom `AdminField` (which is a `FieldPanel` sub-class) utilizing a JavaScript library like CodeMirror or Ace. These libraries bring features like syntax highlighting, line numbering, and other editor-specific features that make the experience significantly smoother.

Below I've created a basic example using CodeMirror integrated into Wagtail. This is more complex because it requires adding custom JavaScript and CSS to Wagtail's admin.

```python
# fields.py
from wagtail.admin.edit_handlers import FieldPanel
from django import forms
from django.utils.html import format_html


class CodeMirrorField(forms.Textarea):
    def render(self, name, value, attrs=None, renderer=None):
        html = super().render(name, value, attrs, renderer)
        return format_html(
            '<div class="code-mirror-container" data-name="{name}">{html}</div>',
            name=name,
            html=html
        )

# models.py (updated)

from django.db import models
from wagtail.admin.edit_handlers import FieldPanel
from wagtail.core.models import Page
from .fields import CodeMirrorField

class CodePage(Page):
    code_content = models.TextField(blank=True)

    content_panels = Page.content_panels + [
        FieldPanel('code_content', widget=CodeMirrorField, heading="Code Content"),
    ]

# wagtail_hooks.py
from django.templatetags.static import static
from django.utils.html import format_html
from wagtail import hooks

@hooks.register('insert_editor_js')
def code_mirror_js():
    return format_html(
        """
        <script src="{code_mirror_js_url}"></script>
        <script>
        document.addEventListener('DOMContentLoaded', function() {{
           var codeContainers = document.querySelectorAll('.code-mirror-container');
           codeContainers.forEach(function(container) {{
             var textarea = container.querySelector('textarea');
             var editor = CodeMirror.fromTextArea(textarea, {{
                 lineNumbers: true,
                 mode: "text/x-python",
                 theme: "material"
             }});
             editor.on('change', function() {{
                textarea.value = editor.getValue();
             }});
           }});
        }});
        </script>
        """,
        code_mirror_js_url=static('codemirror/lib/codemirror.js'),

    )

@hooks.register('insert_editor_css')
def code_mirror_css():
    return format_html(
        '<link rel="stylesheet" href="{code_mirror_css_url}">',
        code_mirror_css_url=static('codemirror/lib/codemirror.css')
    )
```

Note: This code example assumes you've installed `codemirror` using npm or similar, and that you've copied the library’s necessary files into your static directory within the "codemirror" subdirectory. In a real project, you may have to adjust paths to reflect your exact static file structure. For this, you’d need to have CodeMirror’s JavaScript and CSS files located within your `static` directory, along with a minimal implementation setup. The `mode` option within `CodeMirror` can be adjusted to provide syntax highlighting for different types of code, for example: "text/x-python", "application/json", or "text/html". The important aspect here is the ability to attach the JavaScript library to your newly created text area.

This strategy, while slightly more involved, provides a robust editing experience, similar to IDEs. The key here is creating the necessary hooks to load the JavaScript and CSS files for your editor, ensuring that it can correctly interact with the form fields within the admin interface.

For situations demanding greater control, the third method involves creating a complete custom admin interface using Django's forms and templates. This can be overkill for most use cases, but it allows total customization. You can essentially build your own admin panel with direct access to data, independent of Wagtail's core admin. You'd essentially sidestep Wagtail's `AdminField` abstractions and provide your form field.

```python
# admin.py
from django.contrib import admin
from django import forms
from django.shortcuts import render, redirect
from django.urls import path
from .models import CustomTextPage


class CustomTextForm(forms.ModelForm):
    class Meta:
        model = CustomTextPage
        fields = ['raw_text']


class CustomTextAdmin(admin.ModelAdmin):

    change_form_template = "admin/custom_text_editor.html" # Path to our custom template

    def get_urls(self):
        urls = super().get_urls()
        custom_urls = [
            path('add/', self.add_view, name='add_custom_text'),
            path('<int:pk>/change/', self.change_view, name='change_custom_text'),
        ]
        return custom_urls + urls

    def add_view(self, request, form_url='', extra_context=None):
      if request.method == 'POST':
        form = CustomTextForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('/admin/wagtail/pages/') # Redirect to Wagtail page list
      else:
        form = CustomTextForm()

      return render(request, 'admin/custom_text_editor.html', {'form': form, 'add': True})


    def change_view(self, request, object_id, form_url='', extra_context=None):
        instance = CustomTextPage.objects.get(pk=object_id)
        if request.method == 'POST':
            form = CustomTextForm(request.POST, instance=instance)
            if form.is_valid():
                form.save()
                return redirect('/admin/wagtail/pages/')  # Redirect to Wagtail page list

        else:
            form = CustomTextForm(instance=instance)
        return render(request, 'admin/custom_text_editor.html', {'form': form, 'add': False})

# models.py
from django.db import models
from wagtail.core.models import Page

class CustomTextPage(models.Model):
    raw_text = models.TextField(blank=True)

# admin/custom_text_editor.html

{% extends "admin/base_site.html" %}

{% block content %}
    <h1>{{ add|yesno:"Add,Edit" }} Raw Text</h1>
    <form method="post">
        {% csrf_token %}
        {{ form.as_p }}
        <button type="submit">Save</button>
    </form>
{% endblock %}


admin.site.register(CustomTextPage, CustomTextAdmin)
```

In this method we are creating a custom admin interface with a `ModelAdmin` class. This example would require you to create your own urls, and completely bypass the Wagtail Admin page editors. This is a good option if you want to go completely outside of the Wagtail Admin structure for this particular page type.

Ultimately, the choice of method depends on the specific needs of your project. For simple raw text editing, the standard `FieldPanel` will often suffice. If you require syntax highlighting or a more advanced code editing experience, integrating a library like CodeMirror via custom widgets is the way to go. If you are looking for a more complex solution, you can create a completely custom admin interface.

For further reading, I would highly recommend reviewing “Two Scoops of Django 3.x” by Daniel Roy Greenfeld and Audrey Roy Greenfeld for an in-depth look at Django forms and admin customization. Additionally, familiarize yourself with the Wagtail documentation, particularly the sections on custom form fields and admin interfaces. CodeMirror’s website also contains detailed documentation on its various features and options. Understanding these resources will help you adapt and build more advanced implementations to fit specific requirements. I hope this response helps you craft exactly the raw text editor that you need.

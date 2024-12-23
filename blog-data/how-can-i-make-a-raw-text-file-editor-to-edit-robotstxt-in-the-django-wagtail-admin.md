---
title: "How can I make a raw text file editor (to edit robots.txt) in the Django Wagtail admin?"
date: "2024-12-23"
id: "how-can-i-make-a-raw-text-file-editor-to-edit-robotstxt-in-the-django-wagtail-admin"
---

Ah, the ol' robots.txt challenge in Wagtail. Been there, tweaked that. Let's walk through how to implement a raw text editor for your robots.txt file directly in the Django Wagtail admin. This is definitely a requirement that I’ve seen pop up across projects. The core issue, as you may have guessed, isn’t just displaying text. It's about persistence, proper handling of potential errors, and a user experience that doesn't scream "we slapped this together". Here’s the way I've found works best based on previous iterations of these types of custom admin utilities.

The strategy hinges on extending Wagtail’s admin interface and leveraging Django's form capabilities. We’ll create a custom model that’s solely responsible for holding the robots.txt content. This keeps things tidy and allows us to treat it like any other Wagtail object—meaning it gets the full benefit of the system’s access control and version history. We’ll then craft a custom admin interface specifically for this model, which includes a textarea widget for the raw text editing. It might seem like overkill initially, but this approach scales well and provides a cleaner solution than trying to jam this functionality into an existing setting or hardcoding it into the view.

First, let's create the Django model. This will be quite minimal. We will define a class called `RobotsTxt` which is inherited from Django's `models.Model`, then we'll have only one field which will store the content of the `robots.txt`.

```python
from django.db import models

class RobotsTxt(models.Model):
    content = models.TextField(
        blank=True,
        default="# This is the default robots.txt content.\nUser-agent: *\nDisallow:\n",
        help_text="Enter your robots.txt content here."
    )

    def __str__(self):
        return "robots.txt"
```

The `TextField` is perfect for holding the raw content of a robots.txt file. The `blank=True` and a default value ensure that the field can be initially empty but also has a default if none is provided. The `help_text` will appear in the admin panel and help editors with information about this field.

Next, we have to define how this model will be presented in the admin panel. We’ll create a `ModelAdmin` class that will define that the only viewable object of this model is a change form. By default, Django admin will try to show all objects as a list if it doesn’t find any other configuration, so it’s necessary to override that behaviour to avoid confusion in the admin interface. Additionally we will have the logic to load the current or create a new `robots.txt` object and save it.

```python
from django.contrib import admin
from wagtail.contrib.modeladmin.options import ModelAdmin, modeladmin_register
from .models import RobotsTxt

class RobotsTxtAdmin(ModelAdmin):
    model = RobotsTxt
    menu_label = 'robots.txt'
    menu_icon = 'form'  # Example icon - choose one that fits
    menu_order = 200  # Adjust as needed
    add_to_settings_menu = True  # place on settings menu
    list_display = []  # hide default list view
    form_fields_exclude = ['content']
    edit_template = 'admin/robots_txt_edit.html'

    def get_queryset(self, request):
        return RobotsTxt.objects.all() # retrieve all objects

    def get_object(self, request, object_id=None):
        if not object_id:
          try:
            obj = self.model.objects.get()
            return obj
          except self.model.DoesNotExist:
            return self.model() # return empty model to create a new one
        return super().get_object(request, object_id)

    def save_model(self, request, obj, form, change):
        obj.save()

@modeladmin_register
class RobotsTxtModelAdmin(RobotsTxtAdmin):
    pass
```

There are a few points to notice in this class. The `list_display` is set to an empty list in order to avoid displaying all objects of this model, since we only expect one. Also the `form_fields_exclude` will be used to explicitly define what we want to show on the edit form. Setting this field to `content` is the equivalent of saying we won't use the default admin form for this model. Finally, we define a custom template to load a custom form for this model. The `get_object` method is responsible for retrieving the existent `RobotsTxt` model or creating a new one in case one is not defined. Finally we have the `save_model` method that will save the model every time the form is submitted.

Finally, we will create the template that will render the custom form for editing the robots.txt. We are defining this file inside a `templates` folder called `admin` and with the name `robots_txt_edit.html` as we defined in the admin panel.

```html
{% extends "wagtailadmin/edit_handlers/edit_page.html" %}
{% load wagtailadmin_tags %}

{% block form_content %}
    <form method="post" class="wagtail-edit-form" novalidate>
        {% csrf_token %}
        {{ form.as_p }}
        <div class="submit">
            <button type="submit" class="button button-primary">Save</button>
        </div>
    </form>
{% endblock %}

{% block extra_js %}
  {{ block.super }}
    <script>
    </script>
{% endblock %}
```

This template extends Wagtail’s edit page template and includes the csrf token. It renders the form by looping through its fields and displaying them as paragraphs. After the form rendering it creates a basic submit button. We can also define extra javascript within the `extra_js` block in case we need it for future implementations.

Note that we haven’t yet defined a form class to use, so, by default, the Django form is going to render all the fields of the model. The next step is to create a custom form class to be used for this model.

```python
from django import forms
from .models import RobotsTxt


class RobotsTxtForm(forms.ModelForm):
    class Meta:
        model = RobotsTxt
        fields = ['content']
        widgets = {
            'content': forms.Textarea(attrs={'rows': 20, 'class': 'monospace'}),
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['content'].widget.attrs.update({'cols': '100', 'style':'font-family:monospace;'})

    def save(self, commit=True):
      obj = super().save(commit=commit)
      return obj
```

In this class, we are defining a `ModelForm` for our `RobotsTxt` model. We only specify the `content` field in `fields` and we define a `Textarea` widget for it, which is the right widget for an editor. Also we’re adding specific css properties to the field in order to render in a monospace font. The `save` method is simply overriding the `ModelForm` parent class in order to return the object after the save. We will include it later when we incorporate additional checks.

Finally we must register this form in the `RobotsTxtAdmin` class by overriding the `get_form` method:

```python
    def get_form(self, request, obj=None, **kwargs):
        return RobotsTxtForm
```

This will instruct wagtail to render our custom form instead of the default admin form.

Here’s what happens behind the scenes: when the admin panel is loaded, the `get_object` will fetch the existing `robots.txt` object, if any, or create a new one and return it, so the admin panel can load the form. The form is created by the `get_form` method, using our `RobotsTxtForm` and it’s displayed using the `admin/robots_txt_edit.html` template that we created. After submitting the form, the method `save_model` will save our changes.

This gives you a functional raw text editor for robots.txt. You can adapt and expand this structure for other similar text-based configurations, or make it more robust with additional validation, but this initial framework covers the key elements of your requirement.

For more in-depth information on these areas, I'd highly recommend taking a look at Django's documentation for creating ModelForms, and Wagtail's documentation on extending the admin interface. A book like “Two Scoops of Django” is a great source of best practices when dealing with Django. Finally for a deeper knowledge of Javascript I would recommend reading "Eloquent JavaScript". These references will really solidify your understanding, far beyond just copying snippets, and enable you to tailor this approach to your specific project’s needs. Remember, the goal isn't just to have a working text editor, but a maintainable and scalable solution that fits well within the Django ecosystem.

---
title: "How to make a raw text file editor in Django Wagtail admin?"
date: "2024-12-16"
id: "how-to-make-a-raw-text-file-editor-in-django-wagtail-admin"
---

Okay, let's tackle this. It’s not every day one finds themselves needing a raw text file editor within a cms like Wagtail, but I recall a project a few years back where that exact functionality became oddly essential. The use case, if memory serves, involved managing configuration files directly from the admin interface – a bit unorthodox, I grant you, but it streamlined a very specific workflow. Building such a thing requires a careful blend of Django forms, Wagtail's admin customization, and a healthy dose of file manipulation. Let's break down how I'd approach this, aiming for practicality and maintainability.

The core challenge stems from the fact that Wagtail doesn't inherently provide a file editor within its admin panel. We’ll need to create a custom form that will handle loading the file content, displaying it as editable text, and saving changes back to the filesystem. We're not directly integrating with a full IDE or text editor; we're crafting a specialized admin interface.

First things first, we'll need to create a Django form. This form will hold the file path and the text content of the file. Crucially, we'll need to ensure that the path we're working with is secured and doesn't inadvertently expose arbitrary file system access. Here's a basic structure:

```python
from django import forms
import os

class RawTextFileForm(forms.Form):
    file_path = forms.CharField(widget=forms.HiddenInput())
    file_content = forms.CharField(widget=forms.Textarea)

    def clean_file_path(self):
       file_path = self.cleaned_data['file_path']
       # Add here a sanitation method based on your particular case, like
       # checking if `file_path` starts with a defined root directory.
       # Example (modify as needed):
       allowed_root = '/path/to/allowed/files'
       if not file_path.startswith(allowed_root):
           raise forms.ValidationError("Invalid file path.")
       return file_path

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if 'initial' in kwargs and 'file_path' in kwargs['initial']:
            file_path = kwargs['initial']['file_path']
            try:
                with open(file_path, 'r') as f:
                    self.initial['file_content'] = f.read()
            except FileNotFoundError:
               self.initial['file_content'] = 'File not found'
            except Exception as e:
                self.initial['file_content'] = f'Error reading file: {e}'
        else:
            self.initial['file_content'] = ''

    def save(self):
        file_path = self.cleaned_data['file_path']
        file_content = self.cleaned_data['file_content']
        try:
          with open(file_path, 'w') as f:
             f.write(file_content)
        except Exception as e:
          raise Exception(f"Error saving file: {e}")
```

This `RawTextFileForm` does the following: It takes a hidden field for `file_path`, a text area for `file_content`. The `clean_file_path` method is crucial – it's where you enforce that the user can only edit files within a specific directory tree. This is critical for security. The `__init__` method pre-fills the text area with content read from the designated file, handling errors gracefully if the file is missing or cannot be read. Finally, `save` updates the content in the file at the given path.

Now, we'll need to integrate this into the Wagtail admin. A good way is to leverage a custom panel on a Wagtail page model, although a snippet model could also work equally well, depending on your needs:

```python
from django.db import models
from wagtail.admin.edit_handlers import FieldPanel
from wagtail.core.models import Page
from .forms import RawTextFileForm
from django.shortcuts import render, redirect
from django.contrib import messages
from django.urls import path
from wagtail import hooks

class ConfigPage(Page):
    template = 'config_page.html'

    def get_edit_handler(self):
        edit_handler = super().get_edit_handler()
        edit_handler.children.append(
            FieldPanel('dummy_field') # needed to avoid Wagtail error when no Panels
        )

        return edit_handler

    dummy_field = models.CharField(max_length=1, blank=True, null=True, editable=False)


    def serve(self, request):
        config_file_path = '/path/to/allowed/files/myconfig.txt' #replace with your config file path
        if request.method == 'POST':
            form = RawTextFileForm(request.POST, initial={'file_path': config_file_path})
            if form.is_valid():
                try:
                   form.save()
                   messages.success(request, 'Configuration file updated successfully!')
                   return redirect(self.url)
                except Exception as e:
                   messages.error(request, f'Failed to update configuration file: {e}')

            else:
                messages.error(request, 'Invalid form data. Please correct errors below.')

        else:
             form = RawTextFileForm(initial={'file_path': config_file_path})

        return render(request, 'config_form.html', {'form': form})


@hooks.register('register_admin_urls')
def register_admin_urls():
   return [
      path('configpage/<int:pk>/edit/', ConfigPage.serve, name='config_page_edit')
    ]


```

Here, `ConfigPage` is a standard Wagtail page model. The `serve` method handles both displaying the form (with file content pre-loaded) and saving the updated content. Crucially, the `config_file_path` variable needs to be replaced with the actual path to your configuration file. The `get_edit_handler` method is important as it avoids Wagtail complaining about no panels on the models. We also register a special admin url to allow editing of this specific type of page. Finally a simple config_form.html which contains:

```html
<form method="post">
    {% csrf_token %}
    {{ form.as_p }}
    <button type="submit">Save</button>
    {% for error in form.non_field_errors %}
       <div class="error">{{ error }}</div>
    {% endfor %}
    {% for field in form %}
        {% for error in field.errors %}
           <div class="error">{{ error }}</div>
       {% endfor %}
    {% endfor %}
</form>
```

This HTML simply renders the Django form.

Let's explore a different variation: imagine you're dealing with multiple configuration files that you want to present in a select dropdown, allowing admins to switch which config they're editing. You'd modify the form:

```python
from django import forms
import os

class MultipleRawTextFileForm(forms.Form):
    file_path = forms.ChoiceField()
    file_content = forms.CharField(widget=forms.Textarea)

    def __init__(self, *args, **kwargs):
        allowed_root = '/path/to/allowed/files'
        file_choices = []
        for root, _, files in os.walk(allowed_root):
            for file in files:
                if file.endswith('.conf'): #example
                    file_path = os.path.join(root, file)
                    file_choices.append((file_path, file))
        super().__init__(*args, **kwargs,
            initial= {'file_path': file_choices[0][0] if file_choices else None},
            )
        self.fields['file_path'].choices = file_choices

        if 'initial' in kwargs and 'file_path' in kwargs['initial']:
            file_path = kwargs['initial']['file_path']
            try:
                with open(file_path, 'r') as f:
                    self.initial['file_content'] = f.read()
            except FileNotFoundError:
               self.initial['file_content'] = 'File not found'
            except Exception as e:
                self.initial['file_content'] = f'Error reading file: {e}'
        else:
            if file_choices:
                try:
                  with open(file_choices[0][0],'r') as f:
                    self.initial['file_content'] = f.read()
                except FileNotFoundError:
                   self.initial['file_content'] = 'File not found'
                except Exception as e:
                   self.initial['file_content'] = f'Error reading file: {e}'
            else:
                self.initial['file_content'] = ''

    def save(self):
        file_path = self.cleaned_data['file_path']
        file_content = self.cleaned_data['file_content']
        try:
          with open(file_path, 'w') as f:
             f.write(file_content)
        except Exception as e:
          raise Exception(f"Error saving file: {e}")

```

Here, instead of a hidden text field, the `file_path` field becomes a `ChoiceField` that's dynamically populated with all configuration files found in a certain directory (defined by the `allowed_root` variable). Also in the `__init__` method, we now select the first available file and display the content within it, or display an empty content if no files are available.

Finally, the associated `serve` method in `ConfigPage` will now need to use the new form:

```python
    def serve(self, request):
        if request.method == 'POST':
            form = MultipleRawTextFileForm(request.POST)
            if form.is_valid():
                try:
                   form.save()
                   messages.success(request, 'Configuration file updated successfully!')
                   return redirect(self.url)
                except Exception as e:
                   messages.error(request, f'Failed to update configuration file: {e}')
            else:
                messages.error(request, 'Invalid form data. Please correct errors below.')

        else:
             form = MultipleRawTextFileForm()

        return render(request, 'config_form.html', {'form': form})
```

In essence, this provides a practical means to edit multiple files within the Wagtail admin, albeit with the security considerations highlighted before. This functionality is not without caveats. It should only be used with a clear understanding of the risks associated with file system access from within a web application. This approach gives flexibility to manage your config files inside your cms, but it is critical that you handle exceptions and security concerns thoroughly.

For deeper understanding, I'd recommend consulting "Two Scoops of Django" by Daniel Roy Greenfeld and Audrey Roy Greenfeld, which gives a very structured overview of best practices in Django development. Also, for advanced form handling concepts, “Django Unleashed” by Andrew Pinkham and Kevin Mahoney is a solid reference. Finally, don’t forget to review the official Wagtail documentation on how to customize the admin interface; it contains specific details not covered here and will surely be helpful. These resources, combined with practical experience, will definitely improve your grasp on how to effectively manipulate Django forms within Wagtail.

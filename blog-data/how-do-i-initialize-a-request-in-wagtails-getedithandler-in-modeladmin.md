---
title: "How do I initialize a request in Wagtail's `get_edit_handler` in ModelAdmin?"
date: "2024-12-23"
id: "how-do-i-initialize-a-request-in-wagtails-getedithandler-in-modeladmin"
---

,  I've been down this particular rabbit hole before, several times in fact, and it usually comes up when you’re trying to add some dynamic behavior to your Wagtail admin panels. Specifically, initializing the request object within `get_edit_handler` when using a `ModelAdmin` can be a bit tricky if you haven't encountered it before. The problem stems from the fact that the standard `get_edit_handler` doesn't inherently provide a request object, unlike, say, a view function. However, it's not insurmountable; there are effective, albeit sometimes less obvious, methods to achieve this.

The first thing to understand is the context. `ModelAdmin` in Wagtail is designed for managing model instances in a structured way, but it deliberately separates the UI aspects of the admin from the underlying data logic. `get_edit_handler` is where you define the fields and layout of your editing interface. When you introduce the need to access the request context, you’re stepping slightly outside that intended separation of concerns, which is why it requires some careful maneuvering.

Why might you need the request object inside `get_edit_handler` anyway? I've seen several use cases during my career. For instance, one instance required us to dynamically populate a dropdown field based on the current user's permissions. Another project required pre-filling certain form values based on query parameters. Still another involved fetching data from an external api, and needing the user's language preference stored in their request object. These scenarios necessitate access to details usually contained within the request, such as the currently logged-in user, query parameters, or language settings.

Let me break down a few solutions I've employed, with code examples.

**Solution 1: Using `construct_main_form`**

This approach leverages the fact that `ModelAdmin`'s `construct_main_form` receives the request as a parameter. You can intercept that and, in turn, pass the request object to a custom `EditHandler` subclass. This is generally a cleaner solution for maintaining a degree of separation.

First, we define a custom `EditHandler` that can accept a request object:

```python
from wagtail.admin.edit_handlers import ObjectList, TabbedInterface
from wagtail.admin.edit_handlers import BaseChooserPanel
from django import forms
from django.http import HttpRequest


class CustomEditHandler(ObjectList):
    def __init__(self, children, request, *args, **kwargs):
        self.request = request  # Capture the request here
        super().__init__(children, *args, **kwargs)


    def clone(self):
        # Include request in the cloned version
        clone = super().clone()
        clone.request = self.request
        return clone


    def on_model_bound(self):
        super().on_model_bound()
        for child in self.children:
            if isinstance(child, BaseChooserPanel):
                child.request = self.request # inject request into the panels too
```

This `CustomEditHandler` class now stores the request as an instance variable. We also modify the clone function so the request object is present when cloning. Finally, we loop over any children and inject the request when it's a panel so that those nested panels also have the request object.

Next, let's modify the `ModelAdmin` class itself:

```python
from wagtail.contrib.modeladmin.options import ModelAdmin, modeladmin_register
from django.http import HttpRequest
from .models import MyModel # Assume a model called MyModel is present


@modeladmin_register
class MyModelAdmin(ModelAdmin):
    model = MyModel
    menu_label = 'My Models'
    menu_icon = 'folder-open-inverse'
    list_display = ('title', 'id')


    def get_edit_handler(self, instance=None, request=None):
       panels = [
           # Your existing panels go here
            Panel1()
       ]

       return CustomEditHandler(panels, request=request) # Pass the request object

    def construct_main_form(self, request, *args, **kwargs):
        kwargs['request'] = request
        return super().construct_main_form(request, *args, **kwargs)
```

Notice the `construct_main_form` method. We capture the incoming request and inject it as a `kwarg` to `super`. This allows `get_edit_handler` to access it when it's invoked. This method allows a lot of flexibility, and is the method I generally prefer.

**Solution 2: Using Threadlocals Middleware (with caution)**

While not my preferred method, sometimes you encounter legacy code using thread locals, or you have a very unique use case. Thread locals are a way to make data available globally within the scope of a single thread, which is appropriate for a request-response cycle in a web application. Be warned, this approach can make debugging complex.

First, we’d need to implement a simple middleware:

```python
from django.utils.deprecation import MiddlewareMixin
from threading import local

_thread_locals = local()

def get_current_request():
    return getattr(_thread_locals, 'request', None)

class RequestMiddleware(MiddlewareMixin):
    def process_request(self, request):
        _thread_locals.request = request
```

Register `RequestMiddleware` in your `settings.py`. Then, modify the `EditHandler` to get the current request:

```python
from wagtail.admin.edit_handlers import ObjectList, TabbedInterface
from .middleware import get_current_request


class ThreadedEditHandler(ObjectList):

    def __init__(self, children, *args, **kwargs):
        self.request = get_current_request()
        super().__init__(children, *args, **kwargs)

    def clone(self):
        # No modification required, as middleware ensures it exists within thread
        return super().clone()

    def on_model_bound(self):
        super().on_model_bound()
        for child in self.children:
            if hasattr(child, 'request') and child.request is None:
               child.request = self.request
```

Now, modify your `ModelAdmin` to use this new EditHandler

```python
from wagtail.contrib.modeladmin.options import ModelAdmin, modeladmin_register
from .models import MyModel


@modeladmin_register
class MyModelAdmin(ModelAdmin):
    model = MyModel
    menu_label = 'My Models'
    menu_icon = 'folder-open-inverse'
    list_display = ('title', 'id')


    def get_edit_handler(self, instance=None, request=None):
       panels = [
           # your panels
            Panel1()
       ]

       return ThreadedEditHandler(panels)
```

This method injects the current request into the custom handler via middleware. This can be simpler to implement, but it can also obscure dependencies.

**Solution 3: Direct form manipulation (use with care)**

In some very specific cases, you might want to adjust the form directly. This is generally not recommended if you have to change the form very frequently. However, it can be useful for very specific use cases and should be kept as a last resort.

```python
from wagtail.contrib.modeladmin.options import ModelAdmin, modeladmin_register
from django import forms
from .models import MyModel


class CustomForm(forms.ModelForm):
    def __init__(self, *args, **kwargs):
        self.request = kwargs.pop('request', None)
        super().__init__(*args, **kwargs)
        # access the request through self.request, and add logic here

    class Meta:
        model = MyModel
        fields = '__all__'


@modeladmin_register
class MyModelAdmin(ModelAdmin):
    model = MyModel
    menu_label = 'My Models'
    menu_icon = 'folder-open-inverse'
    list_display = ('title', 'id')
    form = CustomForm

    def get_form(self, request, obj=None, **kwargs):
        # add the request to the form
        kwargs['request'] = request
        return super().get_form(request, obj=obj, **kwargs)

```

This example intercepts the `get_form` and injects the request directly into the form. This is the least recommended way, since you can lose the benefits of the Wagtail panels, but is the most direct in terms of code.

**Recommendations for Further Learning**

For understanding Wagtail's architecture, I'd recommend digging into the official Wagtail documentation, specifically sections relating to admin customization and model administration. Beyond that, I also recommend "Django Unleashed" by William S. Vincent for understanding the underlying mechanisms of forms and requests in Django. To properly understand the inner workings of threadlocals, the python official documentation on the `threading` module is a fantastic start. Finally, diving into the Wagtail source code itself can be incredibly illuminating, especially the `wagtail.admin` modules.

Remember, selecting the best approach depends on your specific requirements and the overall architecture of your application. In most cases, using a modified `EditHandler` along with the `construct_main_form` function, as demonstrated in Solution 1, offers the best balance of maintainability and flexibility. I hope this helps you get on the right path; it's definitely one of those things that once you get the hang of it, it feels very natural.

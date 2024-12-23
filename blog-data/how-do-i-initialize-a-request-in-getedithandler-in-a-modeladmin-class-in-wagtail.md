---
title: "How do I initialize a request in `get_edit_handler` in a ModelAdmin class in Wagtail?"
date: "2024-12-23"
id: "how-do-i-initialize-a-request-in-getedithandler-in-a-modeladmin-class-in-wagtail"
---

Alright, let’s tackle this. It’s a common point of confusion when working with wagtail's admin interface customization. I've certainly spent a few late nights debugging similar issues myself. The challenge is that `get_edit_handler` operates within the context of building the admin form, and directly initializing request-specific data in there isn't straightforward. Instead, you often need to use other hooks or methods to accomplish the task. Let me explain, and provide some practical examples.

The core issue is that `get_edit_handler` is called when the form definition is being created – think of it as blueprinting the admin form, rather than actively handling a specific user's request. The actual http request, with its cookies, headers, and user info, only becomes relevant during form processing. Therefore, attempting to grab data directly from a request within `get_edit_handler` will not work as you might expect. This is particularly problematic when, let's say, you want to pre-populate a form field based on the user or some data present in a query parameter.

My experience with this goes back to a project where we needed to implement a custom editing workflow for our content managers. We wanted to default certain page properties based on the user's department. If the request were available inside the get_edit_handler we would be able to extract the department from the user session, but this is not the case. It took a bit of head-scratching to discover the correct way to handle this kind of scenario.

The generally recommended solution involves utilizing the `construct_initial` method of the `BaseForm` class, which your `ModelAdmin` form inherits from indirectly. This method allows you to modify the form's initial data based on the incoming request. Let me show you three different use-cases and code snippets with different approaches on how this can be achieved

**Example 1: Setting initial data based on the user:**

In this scenario, let's assume you have a custom `UserProfile` model that contains a `department` field. This department information will be used to populate a corresponding field on your wagtail page.

```python
from django import forms
from django.contrib.auth import get_user_model
from wagtail.admin.edit_handlers import FieldPanel
from wagtail.models import Page
from wagtail.admin.panels import FieldPanel
from wagtail.snippets.models import register_snippet

User = get_user_model()

class MyPage(Page):
    department = models.CharField(max_length=255, blank=True, null=True)

    content_panels = Page.content_panels + [
        FieldPanel('department'),
    ]

    def get_edit_handler(self):
        edit_handler = super().get_edit_handler()
        return edit_handler

    def __init__(self, *args, **kwargs):
      super().__init__(*args, **kwargs)

    def get_form_class(self):
        form_class = super().get_form_class()
        
        class CustomPageForm(form_class):
          def __init__(self, *args, **kwargs):
            self.request = kwargs.pop('request', None)
            super().__init__(*args, **kwargs)

          def construct_initial(self):
              initial = super().construct_initial()
              if self.request and self.request.user.is_authenticated:
                  try:
                      user_profile = User.objects.get(pk=self.request.user.pk).userprofile
                      initial['department'] = user_profile.department
                  except (User.DoesNotExist, AttributeError):
                      pass  # User might not have a profile, or user is not authenticated
              return initial
        return CustomPageForm

```

Here, the `get_form_class` method is overridden to inject the `request` into the `CustomPageForm` constructor and then we utilize `construct_initial` to populate the 'department' field using the authenticated user's profile.

**Example 2: Initializing based on URL query parameters:**

Let’s consider a case where you want to pre-populate a field based on information passed through the URL query string. Imagine your page has a field for `promo_code` and you want to initialize it with the 'promo' parameter from a URL like `/admin/pages/add/myapp/mypage/?promo=SUMMER2023`.

```python
from django import forms
from wagtail.admin.edit_handlers import FieldPanel
from wagtail.models import Page
from wagtail.admin.panels import FieldPanel

class MyPromoPage(Page):
    promo_code = models.CharField(max_length=255, blank=True, null=True)

    content_panels = Page.content_panels + [
        FieldPanel('promo_code'),
    ]

    def get_edit_handler(self):
        edit_handler = super().get_edit_handler()
        return edit_handler

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_form_class(self):
        form_class = super().get_form_class()
        class CustomPromoPageForm(form_class):
            def __init__(self, *args, **kwargs):
                self.request = kwargs.pop('request', None)
                super().__init__(*args, **kwargs)

            def construct_initial(self):
                initial = super().construct_initial()
                if self.request:
                    promo = self.request.GET.get('promo', None)
                    if promo:
                        initial['promo_code'] = promo
                return initial
        return CustomPromoPageForm

```

In this example, we've adapted the `get_form_class` method and our custom form to extract the 'promo' query parameter and set it as the initial value for the 'promo\_code' field.

**Example 3: More Complex Initialization Logic:**

Finally, let’s say we have a situation where initialization requires calling another service or API. Instead of directly manipulating initial data in a form field, we’ll set up a callback that can then access the request inside a view function using the `process` method.

```python
from django import forms
from wagtail.admin.edit_handlers import FieldPanel
from wagtail.models import Page
from wagtail.admin.panels import FieldPanel
from django.http import HttpResponse

class MyApiPage(Page):
    api_data = models.TextField(blank=True, null=True)

    content_panels = Page.content_panels + [
        FieldPanel('api_data'),
    ]

    def get_edit_handler(self):
        edit_handler = super().get_edit_handler()
        return edit_handler

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def get_form_class(self):
      form_class = super().get_form_class()
      class CustomApiPageForm(form_class):
          def __init__(self, *args, **kwargs):
             self.request = kwargs.pop('request', None)
             super().__init__(*args, **kwargs)

          def process(self, *args, **kwargs):
            
            response = super().process(*args, **kwargs)
            
            if not self.is_valid():
                return response
            if self.request:
                try:
                    # Simulate an api call
                    import time
                    time.sleep(1)
                    self.instance.api_data = f"Data loaded for user: {self.request.user.username}"
                    self.instance.save()
                except Exception as e:
                   print(f"Error fetching data: {e}")
            return response

      return CustomApiPageForm
```
Here we are using the form `process` method to make an API call. You can replace the simulate API call by whatever is needed in your system, this was just an example to illustrate how the `request` is available to perform that type of operation.

**Important Resources:**

To delve deeper into this area, I'd recommend consulting the official Wagtail documentation. It's your most reliable source for this kind of information.

*   **"Wagtail Documentation"**: Start with the official documentation pages on form customization and `ModelAdmin` classes. It contains detailed information and examples about `construct_initial`, `process` and `get_form_class`.

*   **"Django's Class-Based Form Documentation"**: Explore the django's documentation related to the forms class-based views, this will give you a deeper understanding on how `construct_initial` works.

In short, remember that you cannot directly rely on `request` inside `get_edit_handler`. Leveraging `get_form_class` with your own form class to add the request and then using `construct_initial` (or the `process` method) allows you to manipulate form initial values or form processing based on the request context. I’ve found these techniques to be invaluable when building more complex and user-aware admin interfaces with Wagtail. Always test thoroughly and consider edge cases like unauthenticated users when implementing these approaches. Good luck!

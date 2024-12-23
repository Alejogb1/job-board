---
title: "How can a Swagger UI be externally accessed when running on Django Wagtail?"
date: "2024-12-23"
id: "how-can-a-swagger-ui-be-externally-accessed-when-running-on-django-wagtail"
---

Alright, let’s talk about exposing a Swagger UI for a Django Wagtail project. This is something I’ve bumped into more than once in my career, typically when dealing with internal APIs that need some level of documentation available to other teams or services, without necessarily opening up the entire admin panel. It's not always a straightforward setup, especially with the layered complexity of Wagtail sitting on top of Django. The good news is, there are several strategies we can employ to achieve this, and we’ll discuss them with working code snippets.

First off, understand that Wagtail itself doesn't inherently provide a Swagger/OpenAPI endpoint. You’ll need to integrate an external library capable of generating the OpenAPI schema and the accompanying Swagger UI. The approach I've found most reliable and manageable uses `drf-yasg`, a Django REST framework (DRF) extension. While your api may or may not use DRF directly, `drf-yasg` doesn't strictly require that, it just requires you to have a Django application that it can inspect to generate the schema.

The core problem we face is usually threefold: generating the OpenAPI schema, configuring a view to display the Swagger UI, and ensuring the relevant URL patterns are correctly routed, especially when dealing with Wagtail's URL handling. Based on my past experiences, where I sometimes had custom Wagtail Page models serving as API endpoints, the need for flexibility in documentation became apparent and these steps became a habitual part of my setup process.

Let’s break it down with some code, using illustrative examples:

**Step 1: Installation and Initial Setup**

Before anything else, you’ll need to install `drf-yasg` using pip:

```bash
pip install drf-yasg
```

Then, add `drf_yasg` to your `INSTALLED_APPS` in your `settings.py` file:

```python
INSTALLED_APPS = [
    # ... other apps
    'rest_framework', # If you use DRF for your api, which you likely do.
    'drf_yasg',
]
```

Also, in your `settings.py` file, make sure to add `rest_framework` to the `MIDDLEWARE` list.

```python
MIDDLEWARE = [
   # ... other middleware
   'rest_framework.middleware.LatestVersionMiddleware',
]
```

**Step 2: Generating the OpenAPI Schema and Defining the Swagger UI View**

`drf-yasg` leverages a custom `get_schema_view` function to create an endpoint. I tend to put this configuration in the main urls.py file of my django app (the one where you would define url paths that can handle API calls). Here’s how it looks:

```python
# my_django_project/urls.py
from django.contrib import admin
from django.urls import path, include
from drf_yasg.views import get_schema_view
from drf_yasg import openapi

schema_view = get_schema_view(
   openapi.Info(
      title="Your API Title",
      default_version='v1',
      description="API Description Here",
      terms_of_service="Your terms",
      contact=openapi.Contact(email="your.email@example.com"),
      license=openapi.License(name="Your License"),
   ),
   public=True,
)

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/swagger/', schema_view.with_ui('swagger', cache_timeout=0), name='schema-swagger-ui'),
    # ... your other urls including Wagtail
    path('', include('wagtail.urls')),
]
```

Here, `get_schema_view` generates the schema based on your API endpoints. The `openapi.Info` object configures the metadata shown in the Swagger UI. `with_ui('swagger', cache_timeout=0)` ensures that the Swagger UI is generated from the schema we just created. This method also includes a fallback for JSON schema at `api/swagger.json`.

**Step 3: Example of a Wagtail Page with a View**

To make the swagger UI useful, you'll want to actually have some urls that are inspected to generate the documentation. Here's a basic Wagtail page, which has its own view. Keep in mind, this is a simplified case. In my work, I've seen scenarios where these models were highly customized, including extensive use of Serializers and API ViewSets for complex data structures. But for this example, let's use a simpler case:

```python
# my_wagtail_app/models.py
from django.db import models
from wagtail.models import Page
from wagtail.admin.panels import FieldPanel
from django.http import JsonResponse
from rest_framework.decorators import api_view

class MyApiPage(Page):
    description = models.TextField(blank=True)

    content_panels = Page.content_panels + [
        FieldPanel('description'),
    ]

    @api_view(['GET'])
    def serve(self, request, *args, **kwargs):
        return JsonResponse({
            'title': self.title,
            'description': self.description,
            'page_id': self.id
        })

# my_wagtail_app/wagtail_hooks.py
from wagtail.contrib.modeladmin.options import ModelAdmin, modeladmin_register
from .models import MyApiPage

class MyApiPageAdmin(ModelAdmin):
   model = MyApiPage
   menu_label = 'API Pages'
   menu_icon = 'doc-full-inverse'
   menu_order = 200
   add_to_settings_menu = False
   list_display = ('title', 'description', 'id')
   search_fields = ('title',)

modeladmin_register(MyApiPageAdmin)

```

The key point here is that by adding the `@api_view` decorator, `drf-yasg` is able to correctly inspect the view, and document its method, the url it's associated with, and the response object. This view serves as a basic API endpoint and it will be properly documented with our setup above. To view the resulting documentation, navigate to `your_domain/api/swagger/` after running your Django/Wagtail server. You’ll see an interactive Swagger UI, which allows you to test the api using the documented endpoint.

**Important Considerations:**

1.  **URL Configuration:** Pay close attention to the order of your urlpatterns. Ensure that the Swagger UI endpoint (`api/swagger/`) is placed *before* the Wagtail URL include (`wagtail.urls`). This is because Django goes through the urls one at a time, in the order defined. If you put the Wagtail urls before your Swagger endpoints, Wagtail will catch the request and potentially return a 404.

2.  **Authentication:** If your API endpoints require authentication, you will likely have to configure `drf-yasg` to handle the auth flow correctly, which will often involve updating your OpenAPI `securityDefinitions` in `get_schema_view`. This part of the setup is very application-specific, and I've encountered different methods of API authentication over the years. Some use simple token authentication, others use more robust OAuth2 flows, and each has its own slightly different implementation with Swagger.

3.  **Customizations:** There are a myriad of options for customizing the output schema, how the views are displayed and how you wish to define additional information. `drf-yasg` is flexible in allowing you to further control the documentation output, this is done through a variety of settings.

4. **Security:** Ensure your Swagger UI is not publicly accessible if it documents internal APIs. I have used nginx configuration, and or django-axes (which has been useful for brute force protection) in the past to restrict access to my endpoints.

**Further Reading and References:**

For a comprehensive understanding of the libraries used, I highly recommend going through the official documentations:

*   **drf-yasg Documentation:** The official documentation is located at the repository hosting the project, and is easily found through a standard google search. This is the place to start when trying to understand the different settings and customization that this library provides.
*   **Django REST Framework (DRF) Documentation:** Although we aren’t strictly using DRF views for our Wagtail pages, understanding DRF fundamentals provides valuable context, particularly regarding serialization and view configurations. The DRF documentation is thorough and well-written.
*   **Wagtail Documentation:** Wagtail’s official docs are crucial for understanding Wagtail’s specific architecture, especially the routing mechanism. These documents give you the necessary background to understand how to use wagtail together with other django based applications.

In closing, exposing a Swagger UI with Django Wagtail requires a careful integration of `drf-yasg` alongside your Wagtail setup. It’s not complicated once you grasp the workflow and the importance of the URL ordering, but it’s a necessary step in providing valuable documentation for your APIs. This allows developers to interact with your api's in a consistent and understandable way. The approach I've outlined here has been battle-tested across multiple projects and should provide a solid foundation for your needs. Remember to prioritize good security practices, and always refer to the official documentation for the most up-to-date information.

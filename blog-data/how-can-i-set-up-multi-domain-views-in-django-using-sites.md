---
title: "How can I set up multi-domain views in Django using Sites?"
date: "2024-12-23"
id: "how-can-i-set-up-multi-domain-views-in-django-using-sites"
---

Alright, let's tackle multi-domain setups in Django, leveraging the `django.contrib.sites` framework. This isn't an uncommon challenge, and I've definitely been down this path a few times, most notably when migrating a monolithic app into separate front-end experiences for distinct client segments. It can be a little tricky if you're not used to thinking about your application from a site-aware perspective, but the payoff is significant in terms of flexibility and maintainability.

The core concept behind using the `Sites` framework is that it allows you to associate parts of your application's logic—specifically, views, models, and settings—with individual domains. Each domain is represented as a `Site` object within your database. Now, this isn't simply about domain names; it's more about separating logical units of your application by domain, which allows a single Django instance to serve multiple, effectively independent, web applications.

Essentially, when a request comes in, Django's middleware looks at the `Host` header (the domain name) and tries to match that to a `Site` object in the database. It then makes that `Site` object available throughout the entire request lifecycle. This is crucial because you can then use it in your views, templates, and even models to conditionally determine behavior and render different things based on the domain.

To get started, ensure you’ve added `'django.contrib.sites'` to your `INSTALLED_APPS` in your `settings.py` file. Also, make sure the `SITE_ID` setting is set. It's common practice to keep this at '1' initially, as we will see later.

Now, the database component: After installing the app, you’ll need to run `python manage.py migrate` to create the necessary tables. Once that's done, head over to your Django admin, and you should see a section for "Sites". Here you'll create entries corresponding to each domain you intend to handle. Each entry requires a domain name and a display name. For instance, you might have `example.com` with the display name "Main Site" and `api.example.com` with the display name "API Endpoint". Crucially, each of these gets a unique `id` in the database.

The real magic happens in your views and settings. Let me illustrate with a practical example:

**Example 1: Domain-Specific Templates and View Logic**

Suppose you want to render different homepage templates depending on the domain.

```python
# views.py

from django.shortcuts import render
from django.contrib.sites.shortcuts import get_current_site

def homepage(request):
    current_site = get_current_site(request)

    if current_site.domain == 'example.com':
        return render(request, 'homepage_main.html')
    elif current_site.domain == 'api.example.com':
        return render(request, 'api_homepage.html')
    else:
        return render(request, 'default_homepage.html')
```

Here, the `get_current_site(request)` function retrieves the `Site` object associated with the incoming request. We then check the `domain` attribute and render different templates accordingly. If none match, we serve a default. You'll need to define those templates in your template directories. This approach is straightforward, but if you have many sites, it can lead to unwieldy if-else chains.

**Example 2: Domain-Specific Settings**

Often you'll need different settings per domain, things like API keys, specific URLs, or email configurations.

```python
# settings.py

from django.conf import settings
from django.contrib.sites.models import Site

def get_site_settings(request):
  try:
        site = Site.objects.get(domain=request.META.get('HTTP_HOST', ''))
        site_id = site.pk
  except Site.DoesNotExist:
        site_id = settings.SITE_ID  #Fallback to default site.

  if site_id == 1: #id of example.com site from admin
        return {
            'MY_API_KEY': 'api-key-for-example-com',
            'MY_URL': 'https://example.com/api/',
            'EMAIL_HOST': 'smtp.example.com',
        }
  elif site_id == 2: # id of api.example.com from admin
        return {
            'MY_API_KEY': 'api-key-for-api-example-com',
            'MY_URL': 'https://api.example.com/',
             'EMAIL_HOST': 'smtp.api.example.com',
       }
  else:
    return {
           'MY_API_KEY':'default-api-key',
           'MY_URL':'https://default.com/api',
           'EMAIL_HOST': 'smtp.default.com',
       }

def site_settings_middleware(get_response):
    def middleware(request):
        request.site_settings = get_site_settings(request)
        response = get_response(request)
        return response
    return middleware

# Add to your MIDDLEWARE settings
MIDDLEWARE = [
    ...,
    'your_app.middleware.site_settings_middleware',
     ...,
]
```
Here we’re defining a site settings middleware that makes specific settings available via request.site_settings. This demonstrates how to load settings dynamically by site. You then use `request.site_settings['MY_API_KEY']` etc within your code.

**Example 3: Dynamic URL Resolution**

Let’s say your application needs different landing pages based on the domain and a user is referred via a token:
```python
#urls.py

from django.urls import path
from . import views

urlpatterns = [
    path('landing/<str:token>/', views.landing_page, name='landing_page'),
]
```
```python
# views.py
from django.shortcuts import render, redirect
from django.urls import reverse
from django.contrib.sites.shortcuts import get_current_site

def landing_page(request, token):
    current_site = get_current_site(request)

    if current_site.domain == 'example.com':
          return redirect(reverse('main_site_landing_page', kwargs ={'token':token}))
    elif current_site.domain == 'api.example.com':
         return redirect(reverse('api_site_landing_page', kwargs ={'token':token}))
    else:
        return render(request, 'default_landing.html', {'token':token})

```

In this setup, the `landing_page` view is called with a token, it determines the site, and dynamically redirects the user to a different named url, specific to a particular site.

Important Considerations:

1.  **Site ID:** In the default settings.py file, the variable `SITE_ID` is typically set to 1. This is the id for the initial site created, usually the primary site. Ensure that your fallback behavior handles the case when a site cannot be determined from the incoming request. In the second example, the middleware will fall back to the SITE_ID value if a site does not match the request domain.

2.  **Caching:** You'll likely want to cache `Site` objects aggressively, especially if you have many sites. Django provides a way to invalidate the cache when the database changes, so you can safely implement caching within your `get_current_site` implementation or as a custom function based on the example. This prevents unnecessary database queries on each request, which is beneficial for performance.

3.  **Template Context Processors:** You can create a template context processor to automatically pass the current site object to your templates. This reduces boilerplate in each of your views.

4.  **Middleware:** Consider adding custom middleware to manipulate request attributes, as shown in example 2. This enables site-specific behavior at the middleware level, which can be advantageous.

5.  **Testing:** Thorough testing of multi-site behavior is critical, focusing on both functionality and performance. Tools like `django.test.Client` allow you to simulate requests to different domains.

For further reading, I’d recommend delving into the Django documentation for `django.contrib.sites`, focusing particularly on the `get_current_site` shortcut, the `Site` model, and the concept of site-aware views. Additionally, you might find the book “Two Scoops of Django” by Daniel Roy Greenfeld and Audrey Roy Greenfeld quite helpful for practical tips. Also, the source code for Django's `sites` application is incredibly helpful as a concrete example of how the framework is designed to operate. These resources should offer a deeper understanding of the technical underpinnings and provide further guidance as you explore more advanced use cases. Remember, that in large projects, properly leveraging sites allows for modularity and better control over your applications.

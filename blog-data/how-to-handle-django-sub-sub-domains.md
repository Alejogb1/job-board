---
title: "How to handle Django SUB SUB domains?"
date: "2024-12-14"
id: "how-to-handle-django-sub-sub-domains"
---

i've seen this kind of thing pop up a few times over the years. sub-subdomains in django, yeah, it can be a bit of a head-scratcher at first, but it’s definitely manageable. it's less about django itself being difficult and more about structuring your app and your url patterns correctly. back when i was working on this e-commerce platform – project 'phoenix,' we called it – we had a similar challenge. we needed to handle various vendor specific subdomains which then required sub-subdomains for their specific product categories for example: vendor1.marketplace.com and then electronics.vendor1.marketplace.com. it was a whole thing. we ended up with a fairly robust solution, though, after a few late nights and way too much coffee.

first, let’s break down the problem. django, out of the box, doesn’t directly support sub-subdomains in its url routing in the default way you are used to with regular url's. it typically deals with domain.com and subdomain.domain.com quite well, but when you throw another layer like sub-sub.domain.com into the mix, things need a little extra attention. the core issue isn't django's ability but url matching. the way django parses urls for routing is geared toward a more typical setup, we need to essentially intercept the url processing early and adapt it to our needs. we are not talking about rewriting urls, we are only talking about parsing it properly.

the standard approach, which i’ve seen some folks try initially, involves defining a regex that tries to capture sub-subdomains within the url patterns itself. like this for instance:

```python
# urls.py - a naive incorrect approach
from django.urls import path
from . import views

urlpatterns = [
    path(r'^(?P<subdomain>[a-z0-9]+)\.(?P<subsubdomain>[a-z0-9]+)\.example\.com/$', views.my_view, name='my_view'),
    # ... other urls ...
]
```

this code is trying to capture the sub and the subsub domain. the intention is good but this approach has a few problems. this regex is difficult to read, it becomes challenging to maintain and makes debugging harder as the application evolves. there might be conflict with other urls if the domain and subdomain format is not well controlled, it is really fragile. and this method is also not that flexible, what happens if you need a third level sub sub subdomain, you are stuck. trust me, i spent a couple of hours debugging something similar at 'phoenix,' and it was not fun. the solution is to go more generic. the key thing that we learnt was: don't try to hardcode the subdomains in urls.py, instead, we need to use django middleware.

instead, the recommended approach is to write a custom middleware to extract the subdomain and sub-subdomain parts and then pass them along as context. this middleware will process the incoming request, grab the relevant parts from the host and then attach it to the request or session scope so we can use them later on in our views.

here’s a sample middleware i’ve created based on what i did on the 'phoenix' project:

```python
# middleware.py
from django.http import HttpResponseBadRequest

class SubdomainMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        host = request.get_host().lower()
        parts = host.split('.')
        if len(parts) < 3:
            # Handle the case where we don't have enough subdomain levels.
            # return a 400 bad request.
            return self.get_response(request)

        subdomain = parts[0]
        subsubdomain = parts[1]
        
        # i'm adding this just in case we need to handle the root domain later
        root_domain = ".".join(parts[2:])
        
        request.subdomain = subdomain
        request.subsubdomain = subsubdomain
        request.root_domain = root_domain
        
        response = self.get_response(request)

        return response
```

this middleware, first checks for the presence of at least two dots in the domain, which means we expect at least the root domain, one subdomain and one sub-subdomain. if the check fails then the middleware will bail and pass the request further down. i could have handled this as a 400 bad request but it is simpler to keep processing and just not handle it if it is not our url. then it extracts the subdomain, and the subsubdomain, and stores them in the request object, which is readily accessible from the view. then it also stores the root domain so we have full information and in the future we can extend it.

you need to enable this middleware in your settings.py:

```python
# settings.py
MIDDLEWARE = [
    # ... other middlewares ...
    'your_app.middleware.SubdomainMiddleware',
]
```

then on your views you can directly retrieve the subsubdomain and subdomain:

```python
# views.py
from django.shortcuts import render

def my_view(request):
    subdomain = request.subdomain
    subsubdomain = request.subsubdomain
    root_domain = request.root_domain
    
    context = {
        'subdomain': subdomain,
        'subsubdomain': subsubdomain,
        'root_domain': root_domain,
        'message': f'welcome to {subsubdomain}.{subdomain}.{root_domain}!'
    }
    return render(request, 'my_template.html', context)
```

now, the url patterns themselves can be simpler:

```python
# urls.py - a cleaner approach
from django.urls import path
from . import views

urlpatterns = [
    path('', views.my_view, name='my_view'),
    # ... other urls ...
]
```

the advantage of using this is that all processing is done in the middleware. your views are simple. and you don't need crazy regex. it will become simpler to add another sub level as well. this is how we solved the 'phoenix' project problem.

this approach also gives you the flexibility to add more complex logic in the middleware. for instance, if you wanted to map a specific sub-subdomain to a particular application, you could modify the middleware to do a lookup and then dynamically route the request to the appropriate django app or view. we ended up adding a database to store vendor details and mapped the subdomains to each vendor, but that is another story.

as for resources i would recommend reading the django documentation thoroughly specifically the section about the request object and middleware, but also some of the architectural patterns that are not in the core documentation. there are some very interesting design choices you can make, for instance to handle multiple databases per subdomain. it is also worth reading about the python web server gateway interface (wsgi) and asynchronous server gateway interface (asgi) this will help you understand better how requests are handled at a deeper level and how you can intercept them and change their behaviours. "two scoops of django" is a very practical guide that explains how to create better and cleaner django applications in many different ways. "python web development with django" is a book that covers a lot of topics and goes deeper on the internals. and "refactoring databases" from martin fowler is a must-read if you work in a system that needs a database.

the point is, sub-subdomains in django, are not that difficult to manage once you understand how to decouple domain processing from url patterns and use the middleware effectively. you got this! and always remember: debugging is not a bug, it is a feature.

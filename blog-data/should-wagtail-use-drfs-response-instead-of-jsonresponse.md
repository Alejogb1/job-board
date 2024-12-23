---
title: "Should Wagtail use DRF's Response() instead of JsonResponse()?"
date: "2024-12-23"
id: "should-wagtail-use-drfs-response-instead-of-jsonresponse"
---

Okay, let's unpack this. I've actually spent a fair amount of time grappling with the nuances of json handling in web frameworks, and while both `JsonResponse` and DRF's `Response()` serve a similar purpose, they operate on fundamentally different levels of abstraction. I recall a project a few years back, a fairly large content management system built on Wagtail, where we initially leaned heavily on `JsonResponse` for api endpoints. As the project scaled and the requirements grew more complex, we started running into limitations which eventually led us to a deeper evaluation and a migration towards DRF's `Response`. The switch wasn’t trivial, but it certainly provided the flexibility we needed.

The primary difference, as many have pointed out, lies in what each is designed to handle. `JsonResponse`, which originates directly from Django itself, is essentially a shortcut for creating an `HttpResponse` with a specific mime type (`application/json`) and serializing Python data structures into JSON. It’s straightforward, easy to use, and perfectly adequate for many simple api use cases. However, it’s fairly rigid. The data you pass to it needs to be serializable to JSON directly, and there’s very little built-in support for anything beyond that.

DRF’s `Response()`, conversely, is a much more powerful abstraction. It operates at the heart of Django Rest Framework’s serialization and content negotiation mechanisms. When you use a `Response` instance, you’re essentially feeding data into a system designed to handle complex data transformations, format changes, error handling, and much more. Instead of directly receiving a json-serializable object, `Response()` accepts python objects of any kind, passes them to a serializer defined in the request context, which in turn creates a json (or other format) output based on the content negotiation. This gives you a lot of benefits.

For instance, let's say you need to handle pagination for your api results. With `JsonResponse`, you’d have to manually construct the json structure for pagination including things like the number of pages, current page and so on, and ensure its consistent across all your endpoints. DRF, on the other hand, provides built-in tools (such as its `PaginationSerializer`) for automating this.

Here's a basic example using `JsonResponse` in Wagtail to illustrate the core usage. Imagine a simple Wagtail model called `BlogPost` and that we have a function returning data to our API:

```python
from django.http import JsonResponse
from .models import BlogPost  # Assuming your BlogPost model is in the models.py

def blog_posts_json(request):
    posts = BlogPost.objects.all()
    data = [{'title': post.title, 'content': post.content} for post in posts]
    return JsonResponse(data, safe=False)
```

Here, we pull out the data from the BlogPost model, then manually structure a list of dictionaries which can be safely serialized to json. Now let's look at a simple DRF implementation, where we would likely be using a viewset.

```python
from rest_framework.response import Response
from rest_framework import viewsets, serializers
from .models import BlogPost

class BlogPostSerializer(serializers.ModelSerializer):
    class Meta:
        model = BlogPost
        fields = ('title', 'content')

class BlogPostViewSet(viewsets.ViewSet):

    def list(self, request):
        posts = BlogPost.objects.all()
        serializer = BlogPostSerializer(posts, many=True)
        return Response(serializer.data)
```

See the difference? We define a `serializer` that knows how to handle the `BlogPost` model and we let DRF and `Response` handle the output formatting. There's no manual construction of the list of dictionaries, we just provide `Response` with the serialized data.

Let's look at a slightly more complex example involving status codes and error handling, that really shows the flexibility of DRF's `Response`. With `JsonResponse`, you're largely stuck with status code 200 unless you delve deeper into manual `HttpResponse` management. Let's imagine we want to return a 404 if we can't find the blogpost:

```python
from django.http import JsonResponse, HttpResponseNotFound
from .models import BlogPost
from django.shortcuts import get_object_or_404

def blog_post_json(request, pk):
    post = get_object_or_404(BlogPost, pk=pk)
    data = {'title': post.title, 'content': post.content}
    return JsonResponse(data) # This will return a 200. We would have to raise the 404 from the get_object_or_404 to get our correct response.
    #return HttpResponseNotFound('Post not found') If we wanted to return a 404, it would require a separate HttpResponse response
```

Now the DRF example:

```python
from rest_framework.response import Response
from rest_framework import viewsets, serializers, status
from rest_framework.exceptions import NotFound
from django.shortcuts import get_object_or_404
from .models import BlogPost

class BlogPostSerializer(serializers.ModelSerializer):
    class Meta:
        model = BlogPost
        fields = ('title', 'content')

class BlogPostViewSet(viewsets.ViewSet):
    def retrieve(self, request, pk=None):
        post = get_object_or_404(BlogPost, pk=pk)
        serializer = BlogPostSerializer(post)
        return Response(serializer.data)
```

DRF will take care of the 404 error because of `get_object_or_404`. We are not required to manually create the 404 response. If you wanted a 400 for bad input, you'd just raise a `ValidationError` in your serializer and again, DRF would handle creating the 400 response for you automatically. This illustrates the flexibility DRF brings to the table.

Now, back to the original question of whether Wagtail should use DRF's `Response` over `JsonResponse`. The answer depends on the complexity of the API you are building. If you're building simple endpoints, returning basic data without need for content negotiation, pagination, or intricate error handling, `JsonResponse` is perfectly acceptable. It's lightweight and performs well for simple tasks.

However, as you move into more complex territory, the advantages of DRF's `Response` become significantly more apparent. It handles serialization, content negotiation, custom response formats, detailed error reporting, and all of this in a cleaner, more consistent way. Additionally, by leveraging DRF's serializers, it enhances the testability and maintainability of your API code.

I recommend checking *Django Rest Framework’s documentation*, which is invaluable in understanding all the features DRF brings. For a good general overview of api design principles, I suggest reading *Build APIs You Won’t Hate* by Phil Sturgeon. Further, to truly understand how Django's `HttpResponse` system works under the hood, it would be worthwhile to explore the Django source code itself or consult *Two Scoops of Django* by Daniel Roy Greenfeld and Audrey Roy Greenfeld. These resources will clarify many of the concepts that I have discussed here, and they were instrumental for me in building more robust and scalable apis with Django and Wagtail.

So, in conclusion, my experience has taught me that while `JsonResponse` has its place, `Response()` provides a much better foundation for building anything more than the most trivial apis. The benefits of the additional abstraction, serialization support, and flexibility DRF offers significantly outweigh the perceived simplicity of `JsonResponse`. If your Wagtail project includes building more than the simplest of apis, DRF's `Response` is the better choice.

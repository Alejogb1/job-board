---
title: "Why are my Wagtail URLs not working?"
date: "2025-01-30"
id: "why-are-my-wagtail-urls-not-working"
---
Wagtail's URL resolution relies heavily on a combination of its page model structure and configured URL routing, deviating significantly from Django's more generic URL patterns. When URLs fail to resolve, the root cause invariably stems from a mismatch between these interconnected components.

The most common failure point arises from incorrect parent-child relationships within the Wagtail page tree. Wagtail, unlike a traditional database-backed content management system, operates on a hierarchical page structure. Each page must reside under a parent, including the root page. If you've added a new page type, and subsequently a page instance, but neglect to place it correctly under an existing branch of the tree (which could be a simple 'home' page), the URL associated with the new page will not function as anticipated. This is because Wagtail constructs URLs based on the path traversed down the tree. An orphaned page, without a recognized lineage, lacks a valid URL.

Further complexities arise from the `route()` method inherent in Wagtail page models. Every page type you define must inherit from Wagtail's `Page` model and should define its own `route()` method, either explicitly or implicitly via inheritance. If the model's `route()` method is either missing or improperly configured, Wagtail's url generation fails. The `route()` method's primary function is to determine whether a given request matches a page and, if so, to prepare the response. Mismatches between this method's logic and the URL structure of the site lead directly to non-working links. For instance, a custom `route()` method that implements strict language prefix matching could fail to resolve a page if the browser omits the language code from the url.

Additionally, subtle issues can occur within the `get_url_parts()` method, also inherited from the `Page` model, which dictates the path segments for url construction. Altering `get_url_parts()` incorrectly, such as by removing the slug component or not including the parent path in the return tuple, will cause a mismatch between the database records, page-tree structure, and generated URLs.

Another point to review is the correct population of the `slug` field. Wagtail utilizes the `slug` field (auto-populated or manually) to differentiate individual page instances under their parent page. If no slug is provided, Wagtail typically attempts to create one from the page title. However, a manual override or a programmatically controlled slug creation that fails can result in a url that is either missing parts or duplicated. Slugs must be unique within the context of a parent page; thus, two pages with the same slug under the same parent cause url generation to fail unpredictably.

URL conflicts between applications within the same Wagtail project can also manifest as broken links. If you have a custom Django application integrated with your Wagtail instance, ensure that your application URLs are correctly namespaced and that no URL patterns clash with Wagtail's internal routes. Poorly managed namespaces or overly broad URL patterns can interfere with Wagtail's own handling of requests.

Finally, configuration mistakes within the Django settings file, especially concerning `WAGTAIL_SITE_NAME` and `WAGTAILADMIN_BASE_URL`, can contribute to URL issues. Mismatched or unset configuration parameters create situations in which Wagtail can’t properly identify the site's hostname and derive appropriate URLs.

Here are examples demonstrating common issues:

**Example 1: Incorrect Parent-Child Relationship**

```python
# models.py (inside your Wagtail app, e.g., my_app)
from wagtail.models import Page
from wagtail.fields import RichTextField
from django.db import models

class MyPage(Page):
    body = RichTextField()
```

*Commentary:* In this simplified `MyPage` model, nothing is inherently wrong in terms of the code definition. However, when creating a `MyPage` instance in the Wagtail admin, if this page is not placed under a pre-existing root or home page, the associated url will fail. The default `route()` method of the base `Page` model needs to walk the tree, and if the parent-child relation does not exist, it won't be able to generate the full path, causing a 404 error. The remedy involves making sure your page is placed under a valid parent and not left "floating" in the admin page tree. The solution can also involve creating a homepage (using another custom model inheriting from `Page`) as the entry point for your site's page structure.

**Example 2: Incorrect Custom `route()` Method**

```python
# models.py (inside your Wagtail app, e.g., my_app)
from wagtail.models import Page
from wagtail.fields import RichTextField
from django.http import Http404

class LanguageSpecificPage(Page):
    language = models.CharField(max_length=2, choices=(('en', 'English'), ('fr', 'French')))
    body = RichTextField()

    def route(self, request, path_components):
         if not path_components:
             return super().route(request, path_components)

         if len(path_components) > 0 and path_components[0] == self.language:
             return super().route(request, path_components[1:])
         else:
            raise Http404
```

*Commentary:* This example implements a custom `route()` method for a language specific page. The code expects a language prefix at the beginning of the URL segment. If we request a URL without the language code, this route method will raise a `Http404`.  The issue lies in the assumption that every URL will have the language prefix, which might not always be the case, especially when using internal Wagtail links. Also, if `path_components` does not contain enough segments then this code will attempt to access indexes that do not exist leading to an `IndexError` or similar error. The correction involves either modifying the way you generate URLs for these pages or amending the `route()` method to support cases with and without the language prefix, potentially falling back to a default language.

**Example 3: `get_url_parts()` Misconfiguration**

```python
# models.py (inside your Wagtail app, e.g., my_app)
from wagtail.models import Page
from wagtail.fields import RichTextField

class CustomURLPage(Page):
    body = RichTextField()

    def get_url_parts(self, request=None):
        site_id, root_url, path = super().get_url_parts(request)
        return site_id, root_url, (self.slug,)
```

*Commentary:* In this code example, the `get_url_parts()` method has been overridden to remove the parent’s URL components, only returning the page’s slug as its url. This is incorrect, since the URL should be the full path from the site’s root to the page. While the generated URL might seemingly function, navigating to this page would typically fail, as the parent-child hierarchy of the page is ignored. The fix involves ensuring that the return tuple from this method contains all path components that lead to this page, including segments from parent pages in the Wagtail tree. The above example also ignores the `request` argument, which can lead to issues when used with template based url generation.

For further investigation and more detailed information, I recommend reviewing the official Wagtail documentation sections on:
* Page models and their properties, including the `route()` method and the tree structure
* URL routing and resolution mechanics
* Template rendering and template tags related to urls
* Django settings related to Wagtail (especially site configuration)
* URL namespace management, and how it interacts with different applications in the same project.
Additionally, examining the source code of Wagtail's core files, especially `wagtail.core.models.Page` and `wagtail.core.urls`, can offer valuable insights into how URLs are internally managed.

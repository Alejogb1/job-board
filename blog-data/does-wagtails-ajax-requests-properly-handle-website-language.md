---
title: "Does Wagtail's AJAX requests properly handle website language?"
date: "2024-12-23"
id: "does-wagtails-ajax-requests-properly-handle-website-language"
---

Alright, let's tackle this. The interplay between ajax and internationalization in a complex cms like wagtail is something I’ve had to address more than once, and it's definitely not as straightforward as it might seem at first glance. When we’re talking about ajax requests within a wagtail environment, particularly concerning different languages, we're touching on a few key areas: how the browser’s language preferences interact with the application, how wagtail structures its urls for different locales, and crucially, how any custom ajax calls you’ve implemented are handling the `accept-language` header. It's a multifaceted puzzle.

From past experience working on a multilingual portal for an international charity, I remember stumbling upon a rather tricky situation where the frontend ajax calls were returning content in the default language rather than the user's currently selected language. It took a while to pinpoint the cause, and the lessons learned have shaped how I approach this issue today. Fundamentally, ajax requests don’t inherently "understand" language preferences. The browser provides the information, primarily through the `accept-language` header, but it’s up to the application to actually parse and react to that. So, no, wagtail’s standard ajax requests don’t magically handle website language, it depends on how the underlying frameworks manage this and also on any additional logic that you have implemented to support multiple languages.

Let's delve into the details. By default, wagtail uses url prefixes to distinguish between different languages. For instance, `/en/` might denote the english version, and `/fr/` the french one, and so on. This is configured in `settings.py` using `WAGTAIL_I18N_ENABLED = True` along with definitions in `LANGUAGES` field. When you navigate through the site using these url prefixes, wagtail’s built-in views and templates are aware of the current language, which is managed by the middleware. However, if you’re making an ajax request to an endpoint that isn’t built into wagtail's normal page handling mechanism, the same level of implicit understanding doesn't exist automatically.

That's where your responsibility comes in. You’ve got a few primary strategies to ensure that your ajax requests are sensitive to language:

1.  **Ensure the `Accept-Language` Header is Being Respected:** The browser automatically sends the `accept-language` header with every request, containing the user's preferred language. Any custom view you create needs to interrogate this header and dynamically return the content in the appropriate language. This involves processing the header value, comparing it with the available translations, and serving the correct variant.
2.  **Incorporate the Locale Directly in the Ajax Endpoint:** If the language prefix is part of the URL being called by ajax, then wagtail, combined with django's i18n handling, will automatically take care of serving the appropriate translation of your model.
3. **Explicitly Pass the Language Preference:** You might find cases where it is more practical to send the locale as a parameter in the query string, rather than parsing the request header.

Let's go through some code examples. Here's how you might implement the first strategy within a django view used by your ajax endpoint. This assumes you are rendering serialized data (e.g., JSON):

```python
from django.http import JsonResponse
from django.utils.translation import gettext as _
from django.utils.translation import get_language_from_request

def my_ajax_view(request):
    language = get_language_from_request(request)
    # Your logic to fetch data based on the language. Here's a simplified example.
    data = {
        "message": _("Hello, world!")
    }
    if language == 'fr':
        data["message"] = "Bonjour, le monde!"
    return JsonResponse(data)
```

In this first example, we are using django's `gettext` function to get the appropriate translation of the message, which will return the default language if no translation exists. We then manually check the current language and replace the translated string if required. This approach means your database must contain translated text. This approach works because we leverage django's i18n capabilities, but it doesn't directly interface with wagtail's page content.

Now, consider a situation where the data originates from a wagtail model (e.g., a snippet). In this situation, the best solution is to rely on wagtail's built-in model translation logic. We must construct urls with the correct language prefix:

```python
from django.http import JsonResponse
from wagtail.snippets.models import get_snippet_model
from wagtail.models import Page
import json

def snippet_ajax_view(request, snippet_pk, page_pk=None):
  language = get_language_from_request(request)

  SnippetModel = get_snippet_model("yourapp", "YourSnippetModel")
  try:
      snippet = SnippetModel.objects.get(pk=snippet_pk)
  except SnippetModel.DoesNotExist:
      return JsonResponse({"error": "Snippet not found"}, status=404)

  if page_pk:
      try:
          page = Page.objects.get(pk=page_pk)
          locale_prefix = f"/{page.locale.language_code}"
      except Page.DoesNotExist:
          locale_prefix = f"/{language}"
  else:
      locale_prefix = f"/{language}"


  data = {
      "title": snippet.get_translation(locale_prefix).title,
      "body": snippet.get_translation(locale_prefix).body if hasattr(snippet, "body") else None,
  }


  return JsonResponse(data)
```

Here, we're explicitly getting the appropriate translation of a snippet model and making sure the url has the appropriate language prefix. It's essential to understand that wagtail's translation mechanism is based on creating translated copies of pages and models within the database. Accessing the data using the `get_translation` method returns the translated model instance for a given language prefix. If you are not leveraging wagtail's language system for your custom models, you would have to either implement it yourself or use the first approach (i.e. manually adding translated string). This also introduces a dependency on wagtail specific classes, and you should avoid this if you are not using wagtail models.

Finally, the third approach, explicitly passing the locale, looks like this:

```python
from django.http import JsonResponse
from django.utils.translation import gettext as _
from django.utils.translation import activate
import json

def locale_ajax_view(request):
    locale = request.GET.get('locale', 'en')
    activate(locale)
    data = {
        "message": _("Hello, world!")
    }
    return JsonResponse(data)
```

In this scenario, we are passing the `locale` as a url query parameter and manually activating the translation system for the current request.

Remember, simply passing the language code to the ajax endpoint does not change anything. The underlying django code must also be configured to actually leverage it by setting the `LANGUAGE_CODE` appropriately and enabling `django.middleware.locale.LocaleMiddleware`. Wagtail's i18n configuration should set all the django values appropriately, so you must ensure that you are not overwriting these values incorrectly if you are working outside of wagtail's standard urls.

To gain deeper expertise in this area, I would recommend reading the official Django documentation on internationalization and localization (i18n and l10n), which forms the bedrock for how wagtail handles language. Also, delve into wagtail’s documentation specifically pertaining to multilingual sites. The book, "Two Scoops of Django" by Daniel Roy Greenfeld and Audrey Roy Greenfeld, also provides excellent insights on django's internationalisation and how to handle complex real-world scenarios.

In summary, while wagtail provides a solid foundation for multilingual websites, the responsibility of handling language preferences within ajax requests ultimately rests on the developer. By thoughtfully managing headers, url structures, and directly manipulating wagtail models, you can ensure a consistent and localized experience for all users, no matter their language. The key is to understand the mechanisms of both django and wagtail and to meticulously ensure that the relevant logic is applied to all custom endpoints.

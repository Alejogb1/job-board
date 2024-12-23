---
title: "Why isn't my Django Wagtail template correctly identifying the page type in an if statement?"
date: "2024-12-23"
id: "why-isnt-my-django-wagtail-template-correctly-identifying-the-page-type-in-an-if-statement"
---

Okay, let's tackle this. I've been down this path more times than I care to remember – that frustrating moment when your wagtail templates seem to be completely ignoring your carefully constructed `if` statements based on page types. The root cause almost always lies somewhere within how context is being passed and how wagtail handles model inheritance, and it's rarely a simple syntax error. I remember a particularly painful project, a sprawling university website where we had to build a complex hierarchy of custom page types. The 'if' statements were seemingly random in their behavior, and it took a good bit of debugging to get to the bottom of it.

So, why does this happen? The problem generally surfaces when you're attempting to use `isinstance` or direct comparisons against class names directly within your template tags, rather than leveraging Wagtail's built-in tools. Django templates, by their nature, deal with the context passed to them. When you pass a wagtail page into the context, you’re generally dealing with a `Page` object, but remember that `Page` in itself is the abstract base class. When you create a custom page type that inherits from `Page`, you’re actually working with an instance of that subclass, *not* just a generic `Page`.

Let me break this down further. Typically, when you are in your template and want to check the page type, you might attempt something like this:

```html+django
{% if page.specific|isinstance:"myapp.models.HomePage" %}
  <p>This is the homepage!</p>
{% endif %}

{% if page.specific|isinstance:"myapp.models.ArticlePage" %}
  <p>This is an article page!</p>
{% endif %}
```

This *might* work sometimes, but it's unreliable and brittle. The core issue is that the `page` variable passed to your template isn't just any `Page` object; it's a `Page` instance that often contains additional data specific to the actual subtype. Here’s where the `.specific` property becomes crucial. `page.specific` attempts to convert the generic `Page` instance into its most specific subclass – if it’s a `HomePage`, it'll return a `HomePage` instance, and so forth. Failing to use this usually leads to your conditional checks failing. Furthermore, relying on direct string comparisons or `isinstance` checks on the raw `page` object instead of the specific version will almost always cause issues.

Also, it's worth mentioning that context processors, or how `page` gets into the template, could also be the culprit. But let's assume standard Wagtail conventions are in play for the time being.

Here's an example of the most reliable way to check page types in Wagtail templates, and a couple of variations for you:

**Example 1: Using `.specific` for type checking**

```html+django
{% if page.specific.is_type('home.HomePage') %}
    <p>Welcome to our homepage.</p>
{% elif page.specific.is_type('article.ArticlePage') %}
    <p>Read our detailed article.</p>
{% elif page.specific.is_type('event.EventPage') %}
    <p>Check out our upcoming event.</p>
{% else %}
    <p>This is a generic page.</p>
{% endif %}
```

In this example, we are utilizing `.specific` along with the built-in `is_type` helper, which is very efficient and recommended by Wagtail. This approach will correctly identify your page types and won't fail due to subtle inheritance issues.

**Example 2: Alternative check using class name comparison**

```html+django
{% if page.specific|stringformat:"%s" == "myapp.models.HomePage" %}
 <p>Welcome to the home page using string comparison</p>
{% elif page.specific|stringformat:"%s" == "myapp.models.ArticlePage" %}
  <p>This is an article page using string comparison.</p>
{% endif %}
```

This method is less preferred as it relies on string comparison, but it still leverages the `.specific` attribute. While it will work, using `is_type` is cleaner, less error-prone, and more readable for most use cases. However, I include it to illustrate how the value from `.specific` can be used differently. The crucial part is using `stringformat:"%s"` to coerce the specific instance into a string representation to enable this comparison.

**Example 3: Checking for a base class**

```html+django
{% if page.specific|isinstance:"myapp.models.BasePage" %}
  <p>This page inherits from BasePage</p>
{% else %}
   <p>This is not a page with BasePage</p>
{% endif %}
```

In this last example, imagine your project includes a base `BasePage` class from which many page types derive. While checking against specific classes is common, sometimes it’s useful to detect if a page is of a certain base type and then handle it in a similar way in your templates, allowing you to group related behavior more easily. Again, we still use `.specific` to ensure we're operating on the correct instance.

**Why `isinstance` can be problematic *without* `.specific`:**

Without `.specific`, your `isinstance` checks will often compare the instance of the *abstract* `Page` class against your concrete class, which will always return `false`, or worse, lead to unexpected behavior. Your template will effectively be comparing "a generic page" against "a specific type of page," leading to erroneous evaluations.

**Key Takeaways and Recommended Reading:**

To summarize, always use `page.specific` when dealing with `if` conditions based on page types in your wagtail templates. `is_type` is highly recommended for cleaner and more maintainable code, but alternative string comparisons using `stringformat:"%s"` are also viable.

For further reading, you should definitely check out the official Wagtail documentation. It is exhaustive and offers a lot of insight into these kinds of issues. Look specifically at the sections on templates and page models. The Django documentation also provides deeper dives into template syntax and context processing that would be beneficial.

For a more theoretical understanding of object oriented programming principles and inheritance, “Head First Design Patterns” by Eric Freeman et al., is a solid starting point, it explains concepts like inheritance and polymorphism clearly. While not directly wagtail focused, the foundational knowledge will help understand why this specific issue arises. For a deeper look into design patterns as applied in python, you can delve into “Fluent Python” by Luciano Ramalho. These resources have been invaluable to my work and I highly recommend them. They will provide a strong foundation for both diagnosing and resolving issues like the one you’re encountering.

Debugging these issues can be tricky, but methodically inspecting the context using django's `{{ page }}` in the template, combined with a solid understanding of `.specific` and inheritance, will lead you to the right solution. It’s almost always about ensuring you're comparing the *actual* type, and not an abstract base type, within your templates.

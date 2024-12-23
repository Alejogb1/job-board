---
title: "How can I get data from the Wagtail promote tab?"
date: "2024-12-23"
id: "how-can-i-get-data-from-the-wagtail-promote-tab"
---

Okay, let's tackle this. Getting data out of the Wagtail promote tab isn't as straightforward as querying a simple model, and I've definitely spent more time than I care to remember navigating this specific challenge in past projects. It's often an area where the abstraction layer can feel a little opaque until you grasp the underlying mechanisms. The promote tab, as you likely know, houses things like the page's slug, SEO title, description, and social media images. These aren't directly attributes of the main `Page` model itself. Instead, they are stored in a separate, related model, specifically the `PageRevision` and subsequent `Page` instances when they are published. This two-step process – revisions and then the publish process that generates a new final page – is key to understanding how to extract the information efficiently.

The first thing to note is that the data isn't readily available in the direct `Page` object when accessing it within the Wagtail admin panel (or even in many custom views if you're not explicitly thinking about this). When you edit a page in Wagtail, the changes, including the promote tab content, are first saved as a `PageRevision`. Think of this as a draft. Once that revision is published, that’s when the changes are baked into a *new* `Page` object. This means we have a slight data extraction problem because we are usually interested in the most recently *published* data rather than drafts.

I've found that the most reliable approach is to work backwards from the *published* pages, and then specifically pull information from the last revision of *that* page which contains the promoted data. We usually won’t work with *all* the revisions. If we wanted, we would get data from *every* revision, but our specific interest is the current, published values. For the majority of use cases, dealing with `Page.objects.live()` will give you the most accurate representation of your published data.

Let's break this down with some concrete examples. Suppose you need to access the seo title and description for all live pages. Here's a common approach:

```python
from wagtail.models import Page

def get_live_page_seo_data():
    """
    Retrieves SEO title and description for all live pages.

    Returns:
        A list of dictionaries, each containing the 'title', 'seo_title', and 'seo_description'.
    """
    page_data = []
    for page in Page.objects.live():
        revision = page.get_latest_revision()
        if revision:
            seo_data = revision.content.get('promote_tab', {})
            page_data.append({
               'title': page.title,
               'seo_title': seo_data.get('seo_title', ''),
               'seo_description': seo_data.get('seo_description', '')
            })
    return page_data

# example usage:
if __name__ == '__main__':
   data = get_live_page_seo_data()
   for item in data:
      print(item)

```

In this first example, we're iterating over each *live* page. `Page.objects.live()` gives us only published pages. For each one, we use `get_latest_revision()`. Critically, the revision’s content is a dictionary, where, nested under the `'promote_tab'` key, we find the SEO title (`seo_title`) and the SEO description (`seo_description`). This structured approach avoids directly accessing private attributes and aligns with Wagtail's API principles. I made a mistake early on in my Wagtail journey by trying to access that data more directly, which turned out to be far less robust and more difficult.

Now, what if you're interested in extracting the social media images as well? That’s where things get a little more complex since Wagtail handles images as file objects. Here's an adjustment to the previous example:

```python
from wagtail.models import Page
from wagtail.images.models import Image

def get_live_page_social_data():
    """
    Retrieves social media title, description, and image for live pages.

    Returns:
        A list of dictionaries, each containing page 'title', 'social_title', 'social_description', and 'social_image_url'.
    """
    page_data = []
    for page in Page.objects.live():
      revision = page.get_latest_revision()
      if revision:
        seo_data = revision.content.get('promote_tab', {})
        social_image_id = seo_data.get('social_image', None)
        social_image_url = None
        if social_image_id:
          try:
            image = Image.objects.get(id=social_image_id)
            social_image_url = image.file.url
          except Image.DoesNotExist:
             social_image_url = None  # Handle missing images

        page_data.append({
             'title': page.title,
             'social_title': seo_data.get('social_title', ''),
             'social_description': seo_data.get('social_description', ''),
             'social_image_url': social_image_url
        })
    return page_data


# example usage
if __name__ == '__main__':
  data = get_live_page_social_data()
  for item in data:
      print(item)
```

In this iteration, we first extract the `social_image` which is not the actual image file, but the image *id* as stored in the `PageRevision.content` dictionary. We use that id to fetch the relevant image record from the Wagtail image model via `Image.objects.get(id=social_image_id)`. Then, to get the image url itself, we go through `image.file.url`. This additional step is crucial and highlights that the promoted data doesn't exist as a simple string value and needs to be retrieved from a related model. We’ve also added a `try-except` block to gracefully handle situations where the image might be missing or have been deleted and we cannot get it, avoiding common errors of missing image IDs that I've tripped over in the past.

Finally, let’s consider a scenario where you're working with custom page models that extend `Page`. The core logic remains the same, but you need to ensure you're filtering the correct type of page, and make sure your query is efficient:

```python
from myapp.models import MyCustomPage  # assuming a model exists called 'MyCustomPage'
from wagtail.models import Page
from wagtail.images.models import Image


def get_custom_page_seo_data(page_type=MyCustomPage):
    """
     Retrieves SEO data for a specific type of custom page, not the general Page model.

    Args:
      page_type: The page model to query against.

    Returns:
      A list of dictionaries, each with title and seo-related fields.
    """

    page_data = []

    for page in page_type.objects.live():
      revision = page.get_latest_revision()
      if revision:
        seo_data = revision.content.get('promote_tab', {})
        social_image_id = seo_data.get('social_image', None)
        social_image_url = None
        if social_image_id:
          try:
             image = Image.objects.get(id=social_image_id)
             social_image_url = image.file.url
          except Image.DoesNotExist:
            social_image_url = None # Handles missing images

        page_data.append({
          'title': page.title,
          'seo_title': seo_data.get('seo_title', ''),
          'seo_description': seo_data.get('seo_description', ''),
          'social_image_url': social_image_url
        })

    return page_data

# example usage:
if __name__ == '__main__':
    data = get_custom_page_seo_data()
    for item in data:
       print(item)
```

Here, we're specifically querying instances of `MyCustomPage` using `MyCustomPage.objects.live()` instead of the base `Page` model. This lets us tailor our queries to specific types of pages, and I found early on in Wagtail projects that this approach makes large codebases much more maintainable, even if the logic is similar. The rest of the function is largely unchanged from our previous iteration, which is ideal as we can reuse the same pattern.

A few closing thoughts and recommendations: when you are working with data like this that is fetched from the `content` dictionary of revisions, you can expect data to sometimes be incomplete, especially if some pages have never had their promote tabs edited. Always check the values after you fetch them.

For further reading, I'd highly recommend reviewing the Wagtail documentation thoroughly, particularly the sections on models and revisions. Also, "Two Scoops of Django" (Audrey Roy Greenfeld and Daniel Roy Greenfeld) provides excellent insight on working with model relations and effective data querying, which is crucial when dealing with Wagtail’s structure. "Effective Django" (Matt Harrison) is another book that will guide you into how to write Django code effectively, and I’ve found this book incredibly helpful. Finally, understanding the inner workings of Django's model querysets is a critical skill. I recommend spending the time to go through the Django documentation’s section on this, as this will allow you to write efficient data queries and understand what is happening underneath the hood. These resources have provided a strong foundation for my own work over the years.

Hopefully, this gives you a solid starting point, and sheds some light on the best way to retrieve data from the Wagtail promote tab. It's not immediately obvious, but with some understanding of revisions and the proper approach to using model relationships, it becomes much more manageable.

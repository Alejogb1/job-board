---
title: "How can I add CKEditor to Wagtail blocks?"
date: "2024-12-23"
id: "how-can-i-add-ckeditor-to-wagtail-blocks"
---

Alright, let’s tackle this. Implementing CKEditor within Wagtail blocks is a common scenario, and it often feels like a puzzle at first. I’ve certainly been there, spending hours tweaking configurations and digging through documentation. It’s not always as straightforward as dropping in a single line of code, but the process, once understood, is quite manageable. The crux of the issue revolves around seamlessly integrating a rich text editor into Wagtail’s block structure, which involves understanding the interplay between Wagtail’s `StreamField` and its blocks, and how CKEditor fits into that ecosystem.

First, it’s essential to recognize that Wagtail offers several built-in block types, but CKEditor isn’t one of them out-of-the-box. This means we'll be leveraging Wagtail's flexibility to create a custom block that uses CKEditor. We accomplish this by utilizing Wagtail’s ability to incorporate custom forms and widgets. Specifically, we will use a `TextBlock` with a custom form widget that swaps the default textarea for a CKEditor instance.

Think back to a project I worked on a few years ago, a large content site migration. We needed very granular control over the page layouts, so we were relying heavily on `StreamField` blocks. The basic `RichTextBlock` provided by Wagtail wasn't quite cutting it – we needed custom toolbar configurations and potentially finer-grained control over the HTML that ended up being output. That's where custom CKEditor integrations came into play.

Here’s a breakdown of the steps involved and how I typically approach it:

1. **Install `django-ckeditor`:** First, you'll need to install the `django-ckeditor` package. This is the bridge between your Django application and the CKEditor library. Make sure you're using a version compatible with your Django and Wagtail versions. In a terminal, just run `pip install django-ckeditor`. I've found this step can be tricky if not followed correctly. Ensure that dependencies align.

2. **Define your custom block:** Here's where the real magic happens. Instead of relying on Wagtail's standard blocks, we are crafting a custom one, using a modified form widget that utilizes `django-ckeditor`. Let’s say you want to create a text block that allows for bold, italics, links and specific styles of headers. Here's a code snippet to illustrate how you might construct that in your `blocks.py` file:

```python
from django import forms
from django.utils.html import format_html, format_html_join
from wagtail.core import blocks
from ckeditor.widgets import CKEditorWidget

class CustomCKEditorBlock(blocks.TextBlock):
    def __init__(self, *args, **kwargs):
        # Use a custom form widget
        kwargs['form_classname'] = 'custom-ckeditor-block'
        super().__init__(*args, **kwargs)

    def render_form(self, value, prefix='', errors=None, bound_field=None, **kwargs):
         return super().render_form(value, prefix=prefix, errors=errors, bound_field=forms.CharField(widget=CKEditorWidget()), **kwargs)

    class Meta:
        icon = 'doc-full'
        template = 'blocks/custom_ckeditor_block.html'

```

*   **Explanation**: This code creates a new custom block inheriting from `TextBlock`. The crucial part is `forms.CharField(widget=CKEditorWidget())` within the `render_form` method. This overwrites the standard `TextArea` widget with the CKEditor widget provided by `django-ckeditor`. This snippet will render your editor block inside the wagtail backend, it also includes a class that I use later in the response for styling.

3.  **Configure CKEditor:** `django-ckeditor` provides a number of settings that you can adjust to customize the toolbar. This is crucial, since you will want a specific set of tools to be shown to the end-user. Instead of defining these settings in `settings.py`, or a separate file, I tend to like creating a settings variable within the block itself. This allows for better modularity and different toolbar options for different blocks. Here's a modified block that contains these settings:

```python
from django import forms
from django.utils.html import format_html, format_html_join
from wagtail.core import blocks
from ckeditor.widgets import CKEditorWidget

class CustomCKEditorBlock(blocks.TextBlock):

    ckeditor_config = {
            'toolbar': [
                 ['Format', 'Bold', 'Italic', 'Underline'],
                 ['Link', 'Unlink'],
                 ['NumberedList', 'BulletedList'],
                 ['RemoveFormat']
             ],
             'height': 300
        }
    def __init__(self, *args, **kwargs):
        # Use a custom form widget
        kwargs['form_classname'] = 'custom-ckeditor-block'
        super().__init__(*args, **kwargs)

    def render_form(self, value, prefix='', errors=None, bound_field=None, **kwargs):
         return super().render_form(value, prefix=prefix, errors=errors, bound_field=forms.CharField(widget=CKEditorWidget(config=self.ckeditor_config)), **kwargs)

    class Meta:
        icon = 'doc-full'
        template = 'blocks/custom_ckeditor_block.html'
```

* **Explanation**: Inside this version of the `CustomCKEditorBlock`, we have added `ckeditor_config`, which is a dictionary holding settings for CKEditor. Within that dictionary, you can customize elements such as the toolbar or the editor's height. The new code has made sure to include the `config=self.ckeditor_config` in the `CKEditorWidget` initializer.

4. **Styling the block:** Depending on your site design, you might find it useful to style the wrapper of the block to have a different width or padding for example. Because I included a class name within the block, we can add custom styling in our CSS:

```css
.custom-ckeditor-block {
  background-color: #f9f9f9;
  padding: 15px;
  border: 1px solid #e0e0e0;
}

.custom-ckeditor-block .cke {
  border: none;
}
```

*   **Explanation**: The `custom-ckeditor-block` class wraps your CKEditor block and can be used to make changes to the look and feel. For example, I have here removed any visual border from the CKEditor widget itself. This is useful, since CKEditor includes it's own visual borders.

5.  **Using the block:** Once you've set up your custom block, you can use it within your `StreamField` just like any other block. For example, if you had a `content` field defined as `StreamField` on your page model, you could add it like this in `models.py`:

```python
from wagtail.core import blocks
from wagtail.core.fields import StreamField
from wagtail.admin.edit_handlers import StreamFieldPanel
from wagtail.models import Page

from .blocks import CustomCKEditorBlock # Import our newly created block

class MyPage(Page):
   content = StreamField([
        ('custom_editor', CustomCKEditorBlock()),
   ], blank=True)


   content_panels = Page.content_panels + [
       StreamFieldPanel('content'),
   ]
```

*   **Explanation:** This code creates a content field in a wagtail page that contains the `CustomCKEditorBlock`. As a result, you can use this newly created block when creating content for your site.

**Important Considerations**

*   **Security:** Carefully configure your CKEditor toolbar to only allow necessary formatting. Avoid including functionality that could introduce security risks, such as direct HTML editing unless absolutely necessary and fully sanitized.
*   **Performance:** While CKEditor is a powerful tool, loading multiple instances of it on the same page might impact performance. Optimize your configurations and potentially explore lazy loading if you anticipate many such blocks on a single page.
*   **Documentation:** Explore the official `django-ckeditor` documentation for an exhaustive list of options. Also, refer to the `wagtail` documentation on `StreamFields` and custom blocks.  For foundational knowledge on implementing custom Django form widgets, the book "Two Scoops of Django" by Daniel Roy Greenfeld and Audrey Roy Greenfeld provides valuable insights.
*   **HTML Sanitization**: Ensure you understand how Wagtail handles sanitization of incoming HTML to prevent cross-site scripting attacks. It leverages bleach under the hood and is worth digging into.

In conclusion, incorporating CKEditor within Wagtail blocks requires creating a custom block using `django-ckeditor`. This involves a few key steps: installing the package, defining a custom block with a CKEditor widget, configuring your desired toolbar, and properly using it within a `StreamField`. This setup provides greater flexibility and control compared to the standard rich text block that wagtail provides, allowing for tailored rich text input. Remember, that the devil is in the details of your configuration. This is not necessarily a plug-and-play type of situation.

That's generally how I approach it, and how I solved some similar challenges in the past. Let me know if there's anything more specific you’d like to explore.

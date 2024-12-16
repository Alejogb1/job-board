---
title: "How can features be specified for RichTextBlocks in Wagtail?"
date: "2024-12-16"
id: "how-can-features-be-specified-for-richtextblocks-in-wagtail"
---

Let's tackle this. I've spent a fair amount of time working with Wagtail, and specifying features for RichTextBlocks, especially within larger, more complex projects, can often feel like navigating a labyrinth if you don't have a solid handle on the underlying mechanisms. So, let's break down how we can effectively control the available tools and formatting options for rich text content in your Wagtail models.

From my experience, the core of this lies in Wagtail's `features` parameter within the `RichTextBlock` definition. This parameter, when absent, defaults to a very basic set of features, generally only including bold and italic text. This is usually not sufficient for practical applications. The power here is that we're given granular control over what editing options are presented to the content editor.

The `features` parameter takes a list of strings, and each string represents a specific editing capability. These strings correspond to the various built-in plugins Wagtail provides for rich text editing. Beyond the basic "bold" and "italic", you'll also frequently find "h1," "h2," "h3" for headings, "link" for creating hyperlinks, "image" for embedding images, "ol" and "ul" for ordered and unordered lists, "document-link" for linking to Wagtail documents, and many others. Crucially, you can also specify extensions or customizations that you write yourself, which is where this really gets flexible.

Here's the first example, illustrating how to provide more structured text editing while limiting unnecessary formatting clutter. Let's say we're building a blog and want editors to focus on content hierarchy and simple formatting:

```python
from wagtail.blocks import RichTextBlock
from wagtail.fields import StreamField
from wagtail.models import Page

class BlogPost(Page):
    body = StreamField([
        ('content', RichTextBlock(features=['h2', 'h3', 'bold', 'italic', 'link', 'ul', 'ol'])),
        # other streamfield blocks
    ], use_json_field=True)

    # ... other model fields and methods
```

In this example, I've specified `h2`, `h3` for section headings, `bold` and `italic` for emphasis, `link` for hyperlinks, and `ul` and `ol` for lists. We’re intentionally excluding “h1”, “image” or other potentially distracting features to maintain a cleaner editing experience and consistent site design. Notice the `use_json_field=True` parameter is always a good practice when defining a `StreamField` in later versions of wagtail. It is best practice to avoid the default `JSON` format due to potential issues.

Now, consider a slightly different scenario: a more complex page, perhaps one for product documentation, where we also want to embed code snippets and allow for blockquotes. For this, we’ll expand the feature set. For that we need a pre-configured class to manage our rich text block features, this is to avoid repetition through out the codebase.

```python
from wagtail.blocks import RichTextBlock
from wagtail.fields import StreamField
from wagtail.models import Page

class FeatureConfig:
    """A utility class to store common rich text configurations."""

    @staticmethod
    def standard_rich_text_features():
       return ['h2', 'h3', 'bold', 'italic', 'link', 'ul', 'ol', 'document-link', 'code', 'blockquote']

    @staticmethod
    def advanced_rich_text_features():
       return ['h1', 'h2', 'h3', 'h4', 'bold', 'italic', 'link', 'ul', 'ol', 'document-link', 'code', 'blockquote', 'image']

class ProductDocumentation(Page):
    body = StreamField([
         ('standard_content', RichTextBlock(features=FeatureConfig.standard_rich_text_features())),
         ('advanced_content', RichTextBlock(features=FeatureConfig.advanced_rich_text_features())),
        # other streamfield blocks
    ], use_json_field=True)
    # ... other model fields and methods
```

In this second snippet, I've included `"code"` for inline code and `"blockquote"` for quoting text. This can be especially useful when explaining a process or referencing specific content. I've also created a `FeatureConfig` class with the intention to avoid repetition and to better manage the different feature sets in different contexts. Note that we would need to register custom plugins for `code` if we wanted the rich text editor to correctly display highlighted code blocks, I recommend looking at Wagtail documentation and specific packages, such as `wagtailcode` for this functionality.

Finally, let's touch on custom plugins. If your design system needs something quite specific that isn't covered by the standard set, you'll need to create a custom plugin. This involves extending the TinyMCE editor configuration, and while it’s outside the immediate scope of feature lists, it's crucial for more intricate scenarios.

Here's a simplified illustration to indicate the structure, but please note that this is only a fragment and actual implementation requires considerable setup, and specific plugin creation:

```python
# in your wagtail_hooks.py file

from wagtail import hooks
from django.utils.html import format_html
from django.conf import settings

@hooks.register('register_rich_text_features')
def register_custom_feature(features):
    features.register_editor_plugin(
        'custom_button',
        default=True,
        js=['my_app/js/custom-button.js'], # point to your js implementation
        css=['my_app/css/custom-button.css'],
    )
    features.default_features.append('custom_button')

# Example Javascript - usually part of your static files
# In my_app/js/custom-button.js
# tinymce.PluginManager.add('custom_button', function(editor, url) {
    //editor.addButton('custom_button', {
    //   // ... plugin configuration ...
    //});
//});

# In your css: my_app/css/custom-button.css
// .mce-custom_button {
//   background-color: lightblue;
// }
```

This snippet illustrates, in a high-level manner, how we'd register a custom feature named `custom_button`. This is a quite a simplified example and you will need to delve into Wagtail's documentation and the TinyMCE plugin creation API to fully understand it.  The real work would be within the javascript file `my_app/js/custom-button.js` where you would specify the TinyMCE button's behavior. This might involve adding a custom icon, handling data insertion into the editor, and any logic to format the final output.

For further reading, I recommend the official Wagtail documentation, which is excellent. Specifically, delve into the sections on `StreamField`, `RichTextBlock`, and how to customize rich text editors using the TinyMCE API. Also, the TinyMCE documentation itself can provide you with all you need to create custom plugins. I’d also suggest looking at the source code of existing Wagtail packages, such as `wagtailcode`, or other third party extensions that add specific features to the rich text editor.

In conclusion, effectively controlling rich text features in Wagtail involves careful selection of the `features` parameter, a well structured approach, and understanding the extensibility it provides using the TinyMCE API. With a little planning, you can provide a consistent, focused, and efficient editing experience, leading to higher quality content for your website.

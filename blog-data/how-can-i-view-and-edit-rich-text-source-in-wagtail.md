---
title: "How can I view and edit Rich Text source in Wagtail?"
date: "2024-12-23"
id: "how-can-i-view-and-edit-rich-text-source-in-wagtail"
---

Right, let's tackle this one. I've spent more than my fair share elbow-deep in Wagtail projects, and dealing with rich text editor intricacies is a common scenario. It’s not always straightforward to directly manipulate the underlying HTML source when you're working within the Wagtail admin interface. The system, by design, aims to abstract away the raw markup for content editors, which, while excellent for usability, can sometimes be a hurdle for more technical users or when specific customization is needed.

Essentially, what you're asking is: "How can I get to the raw HTML and make changes, not just fiddle with the buttons?" There isn't a built-in “view source” button in the default Wagtail rich text editor, but luckily, there are a few viable strategies. I remember a project a few years back – a complex news portal. We needed very specific HTML structures and classes to integrate with their bespoke styling framework. The standard rich text editor outputs were just not cutting it. We had to get down and dirty with the raw markup.

The first, and often most straightforward approach, is to leverage the *’source view’* functionality provided by the underlying editor, which is typically a flavor of draft.js or TinyMCE, based on your wagtail version. Wagtail, as of recently, allows you to configure your rich text fields to include a "source view" plugin. This essentially gives you a code editor directly within the rich text area. Think of it as the 'html' tab on older wysiwyg editors.

Here's how you would accomplish this in your `wagtail_hooks.py` file:

```python
from wagtail.admin.rich_text.editors.draftail import DraftailRichTextArea
from wagtail import hooks

@hooks.register('register_rich_text_features')
def register_source_view_feature(features):
    features.default_features.append('code')

@hooks.register('register_rich_text_features')
def register_draftail_source_view(features):
    features.register_editor_feature(
        'draftail_source_view',
        DraftailRichTextArea,
        js=['wagtailadmin/js/draftail.js'],
    )
    features.default_features.append('draftail_source_view')
```

In this snippet, we're registering the `code` feature to enable syntax highlighting within the editor. More crucially, we are registering the `draftail_source_view` feature, which adds a “code” button (usually an angle bracket icon `< >`) to the rich text editor toolbar. Note the path given to the javascript – if you use tinymce, this will differ. Activating this button allows users to directly see and edit the HTML source. You should also ensure that the appropriate javascript dependencies, specific to the editor being used, are included in the template.

Now, on the surface, this might seem like an immediate fix, but there's a critical nuance. The 'source view' edits the content *within* the browser’s editor, which then Wagtail renders to HTML when the page is saved. This might be slightly different from your initial html if you are using specific extensions and attributes. The HTML you see there might not be exactly what you put in, as the editor and Wagtail's rendering pipeline may perform some sanitization and normalization of HTML.

So, what if you need finer control over exactly what ends up in the database or if you're dealing with an older project not using draftail or have more specific processing rules? This is where the second strategy becomes relevant: utilizing `StreamField` blocks for more granular control. Instead of relying on the rich text editor for everything, you can break down content into structured blocks, some of which can explicitly be HTML source blocks.

Here’s a simplified example of how that block definition might look, within your model's definition:

```python
from wagtail.core import blocks
from wagtail.fields import StreamField
from django.db import models

class MyPage(models.Model):
    body = StreamField([
        ('rich_text', blocks.RichTextBlock()),
        ('html_source', blocks.RawHTMLBlock()),
    ], use_json_field=True)
```

Here, the `body` field is a `StreamField`, allowing the admin to add either rich text blocks or raw HTML blocks. The `RawHTMLBlock` accepts a single text area where you paste your full html. Be aware, using this can introduce potential security risks if not handled carefully. The HTML is stored directly into the database. The `use_json_field=True` allows for the streaming of blocks as json, which will improve performance of streamfield blocks.

This approach bypasses the rich text editor completely for the areas where raw HTML is needed, giving you absolute control over the output. If you want more complex content structures, consider adding further block types.

The third tactic, which I’ve used in a few unique situations, involves creating a custom rich text feature with its own specialized processing logic. This is more involved, but offers the most flexibility if the first two options do not meet your needs. It entails building a custom plugin that intercepts the HTML input/output of the editor.

Here is a simple example:

```python
from wagtail.admin.rich_text.converters.html_to_contentstate import  HtmlToContentStateHandler
from wagtail.admin.rich_text.editors.draftail import DraftailRichTextArea
from wagtail import hooks
from django.utils.html import format_html

@hooks.register('register_rich_text_features')
def register_raw_html_feature(features):
    features.register_editor_feature(
        'raw_html',
        DraftailRichTextArea,
        js=['/static/js/raw_html.js'],
    )

    features.register_converter_rule(
      'raw_html',
      HtmlToContentStateHandler(
            'span',
            {'class': 'raw-html'},
            )
      )
    features.default_features.append('raw_html')
    features.register_widget('raw_html', RawHtmlWidget)

from django.forms import widgets

class RawHtmlWidget(widgets.Textarea):

    def render(self, name, value, attrs=None, renderer=None):
      if value:
          return format_html('<textarea style="display:none;" name="{name}" id="{id}">{value}</textarea><div style="background:#eee;padding:10px;border:1px solid #ccc;">{value}</div>', name=name, id=attrs.get('id'), value=value)
      else:
          return super().render(name, value, attrs, renderer)

```

And the `/static/js/raw_html.js` file would look something like this:

```javascript
window.draftail.registerPlugin({
  type: 'raw_html',
  button: {
    icon: 'code',
    label: 'Raw HTML',
    description: 'Insert Raw HTML',
  },
  init: function(element) {
    element.addEventListener('click', function() {
        var selection = window.draftail.editor.getEditorState().getSelection();
        var contentState = window.draftail.editor.getEditorState().getCurrentContent();
        var focusKey = selection.getFocusKey()
        var contentBlock = contentState.getBlockForKey(focusKey)
        var raw_html = prompt("Enter Raw HTML:", contentBlock.text);
        if (raw_html) {
          window.draftail.editor.focus();
          window.draftail.editor.onChange(window.draftail.editor.replaceText(focusKey, 0, raw_html, null));
        }
    })
  },
  render: function(props) {
    return React.createElement("span", { className: "raw-html",
      dangerouslySetInnerHTML: { __html: props.children }
    });
  },
});
```

This is just a bare example. It will generate a code icon in the editor's toolbar, and when the user clicks on the code button a prompt appears to edit the html directly. The `register_converter_rule` ensures that when Wagtail receives the content in the database it will correctly convert back to the format in the editor. Also notice the javascript element which registers a custom component with draftail. You can use this to register any number of custom features which will further extend the flexibility of the platform. For more complex logic, this custom feature could provide a modal for more sophisticated editing.

These are the main routes I've relied on across various Wagtail projects. I'd recommend delving into the official Wagtail documentation, particularly the sections on Rich Text configuration and StreamFields. Also, for a deeper understanding of the underlying editor's architecture and capabilities, the documentation for Draft.js or TinyMCE (depending on your version) will prove invaluable. *Advanced Web Typography* by Richard Rutter is also a great resource to understand the importance of the underlying markup in creating quality user experience.

Ultimately, selecting the appropriate method depends on the level of control you require and the nature of your content. Start simple with the source view, graduate to streamfields if needed, and consider a custom feature for the most nuanced control. Remember, the balance between user accessibility and developer flexibility is key in achieving a maintainable and robust website.

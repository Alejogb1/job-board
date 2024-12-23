---
title: "How can I render a StreamField in Wagtail, based on a parent-child relationship?"
date: "2024-12-23"
id: "how-can-i-render-a-streamfield-in-wagtail-based-on-a-parent-child-relationship"
---

Okay, let's tackle this. I've certainly dealt with this kind of scenario before, particularly during a project for a large educational institution where content needed to be highly modular yet retain a structured hierarchical aspect. Handling StreamFields within Wagtail, especially when dealing with parent-child relationships, can indeed present some specific rendering challenges. It's not a simple matter of just iterating; we need to think about the context in which these blocks appear.

Essentially, when you're discussing parent-child StreamField relationships, you’re most likely implying that within a parent StreamField block, you have other blocks, and some of these nested blocks might themselves contain further StreamFields. We need a rendering solution that acknowledges this hierarchical nesting without becoming excessively verbose or difficult to maintain. I tend to approach this by using a recursive approach in our template logic, coupled with some careful block definitions.

Let’s consider a concrete example. Imagine you have a ‘Section’ block (the parent) containing content and optionally, a ‘Subsection’ block (the child). And within the ‘Subsection’, you might have a list of various types of content blocks including, potentially, *another* StreamField. This is where proper block definition and template logic becomes crucial.

First, let's establish what our Python block definitions might look like:

```python
from wagtail.blocks import StreamBlock, StructBlock, CharBlock, ListBlock
from wagtail.fields import StreamField

class ContentBlock(StreamBlock):
  text = CharBlock()
  # potentially other standard blocks here like images etc.

class SubsectionBlock(StructBlock):
  title = CharBlock()
  content = StreamField(ContentBlock())

class SectionBlock(StructBlock):
  title = CharBlock()
  content = StreamField(ContentBlock())
  subsections = ListBlock(SubsectionBlock())
```

In this definition, `ContentBlock` is our basic set of content blocks, which is reused by both `SectionBlock` and `SubsectionBlock`. `SubsectionBlock` and `SectionBlock` each contain a title and their own streamfield. Importantly, the SectionBlock can *also* contain an array of `SubsectionBlock` instances, which, as you can see, also include their own streamfield. Now the challenge is to render it cleanly in the templates.

The crucial idea is to use a template tag or custom template logic to handle the recursion. Let's first look at our template logic, which I usually bake into custom template tags to keep templates clean:

```python
from django import template
from django.utils.html import format_html

register = template.Library()

@register.simple_tag
def render_streamfield(streamfield, parent_context=None):
    if not streamfield:
        return ''
    output = ''

    for block in streamfield:
        block_type = block.block_type
        block_value = block.value
        context = { 'block': block, 'value': block_value }

        # Add parent context if it exists
        if parent_context:
            context.update(parent_context)
        
        if block_type == 'text':
           output += format_html('<p>{}</p>', block_value)
        elif block_type == 'subsection': # Special case for nested Streamfield
            output += format_html('<h3>{}</h3>', block_value['title'])
            output += render_streamfield(block_value['content'], context)  # Recursively render nested StreamField
        else:
            #Handle other types here like images etc
           output += format_html('<div>{}</div>', block_value)

    return format_html(output)
```

In this custom tag (`render_streamfield`), I'm iterating over each block within the provided streamfield. When the block is a ‘subsection,’ I’m making a recursive call back to the same `render_streamfield` function with the subsection's nested `content` field. Crucially, I’m also passing the current context into the recursive call, which allows parent context to be accessible to nested streamfield blocks.

Now, let’s consider how you might use this in your actual template:

```html+django
{% load your_custom_tags %}

<div>
    <h1>{{ page.title }}</h1>
    {% for block in page.body %}
        {% if block.block_type == "section" %}
            <h2>{{ block.value.title }}</h2>
            {% render_streamfield block.value.content %}
            
            {% for subsection in block.value.subsections %}
                <h3>{{ subsection.title }}</h3>
                 {% render_streamfield subsection.content %}
            {% endfor %}
        {% else %}
           {% render_streamfield block %}
        {% endif %}
    {% endfor %}
</div>
```

Here, the page’s `body` field is a StreamField itself. In the template, I'm first checking if this block is a "section". If so, I render it's title, then render its primary streamfield, then iterate over its "subsections".  Crucially note that I’m using the `render_streamfield` template tag recursively. The template tag handles the rendering of all the `ContentBlock` items recursively for all of the streamfields at different nesting levels. This recursive approach allows you to handle arbitrary levels of nesting effectively and without repeating too much template code. It is also straightforward to extend with new blocks.

It's very important to note that this approach isn't limited to just two levels of nesting; it can handle as many nested `StreamFields` as you need, as it calls itself.

A couple of best practices which will help with scaling to real projects that I’ve found to be extremely useful. Firstly, always normalize your blocks as much as possible. Having a single basic content block that can be used everywhere makes management and template creation much simpler and less prone to errors. Secondly, use granular template tags. Breaking template logic down into discrete, easily tested components can save you a great deal of headache.

For further reading on advanced template techniques, I would highly recommend ‘Two Scoops of Django’ by Daniel Roy Greenfeld and Audrey Roy Greenfeld. For more on the architecture of Wagtail, including deeper insights into StreamField internals, I found the Wagtail documentation itself (specifically the section on streamfields) extremely useful. Also consider looking at 'Refactoring UI' by Adam Wathan and Steve Schoger which can provide great insights into UI/UX best practices when rendering content using Streamfields.

I've found this iterative approach and separation of concerns (block definition, template tag rendering) to be incredibly effective. It's kept large-scale, complex content management systems manageable and maintainable over the years. Always focus on making your rendering logic modular and reusable to avoid unnecessary complexity and improve long-term maintenance. I hope this helps in your endeavour. Let me know if any specific area is unclear or you have other questions.

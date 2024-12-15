---
title: "How to Set Wagtail DateBlock Default to Current Date?"
date: "2024-12-15"
id: "how-to-set-wagtail-dateblock-default-to-current-date"
---

alright, i’ve seen this one a bunch of times, it's a common pain point when you're setting up a wagtail site, especially if you're trying to get a bit more user friendly with date inputs. so, you want a wagtail dateblock to default to today’s date when a content editor creates a new page or block, yeah? i get it. it’s about making their lives easier and ensuring they don't have to manually select the current date every time. 

i can tell you from experience. i remember battling with this back when i was first getting my feet wet with wagtail on a project for a local museum's website; they had a whole section of their site that revolved around event dates and initially, editors were, let's say, less than thrilled about having to manually input the date every single time. i ended up spending a few hours figuring out a decent way around it. i even considered hacking wagtail’s core briefly, but that was just a silly thought. i definitely wasn’t about to do that.

anyway, lets get to the meat of the problem.  wagtail's `dateblock` doesn’t inherently have a “default to today” option, which can feel a little frustrating at first. the solution usually involves a bit of custom logic in your model or in the `clean` method or in `get_context` if you are using the template based approach, which is the easiest and the one i am going to describe. the idea is to intercept the dateblock’s value and set it dynamically using python.

so, we're not going to modify wagtail itself, no need for that kind of chaos; instead, we’re going to leverage wagtail’s api for blocks and the power of python’s date handling. what you're essentially going to be doing is overriding the `get_context` method on the streamfield block (or whatever block you're using that holds your dateblock) and ensure that you’re populating the dateblock’s value with today’s date if no date was explicitly provided when creating the block.

lets look at the code, this is a pretty typical way to do this. i'll show you how to do it in a couple of places, first inside the models file.

```python
from django.db import models
from wagtail.fields import StreamField
from wagtail.blocks import StructBlock, DateBlock, StreamBlock
from datetime import date


class EventBlock(StructBlock):
    event_date = DateBlock()

    def get_context(self, value, parent_context=None):
        context = super().get_context(value, parent_context)
        if not value.get('event_date'):
            context['event_date'] = date.today()
        return context


class EventPage(models.Model):
    body = StreamField([
        ('event', EventBlock()),
    ], use_json_field=True)

    # ... other fields

```

in this example, `eventblock` is a `structblock` containing a `dateblock`. the `get_context` method is overridden. it fetches the existing context, and then it checks if an event_date already exists. if it doesn’t, or it is empty, we add today’s date into the context under the key ‘event_date’.
this works because `dateblock` is a `fieldblock`, internally wagtail handles the conversion of the context into form data. the `value` argument is a dictionary representing the actual values stored inside the block. note this is not a django form and is not doing the rendering directly, so that is why the default needs to be set in `get_context`.

and here is an example with a `streamblock`:

```python
from django.db import models
from wagtail.fields import StreamField
from wagtail.blocks import StructBlock, DateBlock, StreamBlock
from datetime import date


class EventDateBlock(StructBlock):
    event_date = DateBlock()

    def get_context(self, value, parent_context=None):
        context = super().get_context(value, parent_context)
        if not value.get('event_date'):
            context['event_date'] = date.today()
        return context


class CustomStreamBlock(StreamBlock):
    event_date = EventDateBlock()

    def get_context(self, value, parent_context=None):
        context = super().get_context(value, parent_context)
        if isinstance(value, dict) and value.get('type') == 'event_date':
            event_block_value = value.get('value', {})
            if not event_block_value.get('event_date'):
              event_block_value['event_date'] = date.today()
              context['value'] = event_block_value

        return context


class EventPage(models.Model):
    body = StreamField([
        ('custom_stream', CustomStreamBlock()),
    ], use_json_field=True)

    # ... other fields
```

this second example shows the usage of a `stream block` and the additional logic you will need to be able to set the date. the important part is that you will need to check the `type` attribute of the block if you are dealing with nested blocks to determine the correct dictionary to modify. you will need to look up how the data is stored inside the block you are dealing with, in particular if you are going to modify a `streamfield`. 

finally, if you are using a template, you can also do the same thing there, but this is slightly less performant as the data will need to travel through the whole rendering process and therefore the calculation of the default will be done later:

```html+django
{% load wagtailcore_tags %}

{% if page.body %}
    {% for block in page.body %}
        {% if block.block_type == "event" %}
            {% with event_date=block.value.event_date|default:today %}
                <p>Event Date: {{ event_date|date:"Y-m-d" }}</p>
            {% endwith %}
        {% endif %}
    {% endfor %}
{% endif %}
```

in this html django template example, we are iterating over the blocks inside a streamfield and checking the block type. if we find the block type `event` then we are going to print the value of the event date, or the current date if it doesn’t have a value, by using the `default` tag filter. you need to create the `today` tag filter in a django template tag file.

```python
from django import template
from datetime import date

register = template.Library()


@register.simple_tag
def today():
    return date.today()

```

just remember to include the app that contains that code in the `installed_apps` array of your `settings.py`.

the key here is not to modify wagtail itself, it's just about understanding where to intercept the data inside a block or the templates, and that depends on your setup. in a real world project, i've seen variations of this used in countless situations, from event dates to publication dates and beyond. it's a pretty standard little trick to improve the editorial experience.

for further reading i would suggest looking at wagtail documentation on [streamfield](https://docs.wagtail.org/en/stable/topics/streamfield.html) and [blocks](https://docs.wagtail.org/en/stable/reference/wagtail.blocks/). and for python’s date manipulation i recommend the standard python documentation. additionally, the book “fluent python” by luciano ramalho has a really great section on datetime and all the nuances that entails. and here's a tip i wish i'd known back then; make sure you have good tests in place so you don’t end up with surprises later on. 

oh, and a little joke, why was the python developer always so calm? because he knew how to handle exceptions! haha, bad i know. but anyway. yeah, that's about it for setting default dates on wagtail dateblocks. it’s straightforward once you know the way. any other questions you have just let me know and i am more than happy to help.

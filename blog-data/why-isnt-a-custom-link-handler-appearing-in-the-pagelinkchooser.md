---
title: "Why isn't a custom link handler appearing in the PageLinkChooser?"
date: "2024-12-23"
id: "why-isnt-a-custom-link-handler-appearing-in-the-pagelinkchooser"
---

,  It’s a situation I’ve encountered a few times, usually when implementing complex workflows in Wagtail or similar CMS environments, and it can be a bit baffling initially. The issue you're facing – a custom link handler failing to show up in the PageLinkChooser – typically boils down to a few key areas. We need to check our implementation meticulously to make sure all the pieces are fitting together correctly.

From my experience, the most common culprits aren't necessarily coding errors in the link handler logic itself, but rather issues in how the handler is registered, configured, and made discoverable within the framework. It's like trying to fit a square peg into a round hole; the peg might be perfectly formed, but it won’t work if the hole isn't designed for it. So, let's break this down systematically.

Firstly, let's acknowledge what a link handler *is* in this context. Essentially, it's a piece of code that tells the CMS how to interpret and handle specific types of links—links that don't directly resolve to a page within the content management system, such as links to external applications, complex internal identifiers, or even email addresses with specific parameters. Wagtail’s PageLinkChooser is designed to be extendable, allowing for these custom link types. However, this extensibility depends on a precise setup.

The root of the problem often sits in the *registration* of your custom handler. The CMS framework needs to be aware that your handler exists and should be considered when a link is being created or edited. This usually involves modifying a configuration file or using a specific API for registering the handler. Incorrect registration is by far the biggest hurdle I've seen teams stumble over. Let me illustrate that with the following example, working with a fictional CMS which exposes these concepts.

Here's a snippet demonstrating the proper way to register a hypothetical "product-link" handler in a framework with a module-based setup, using a registration mechanism that is common in various cms platforms.

```python
# myapp/link_handlers.py

from cms.link_handling import BaseLinkHandler

class ProductLinkHandler(BaseLinkHandler):
    identifier = 'product-link'

    @classmethod
    def get_queryset(cls, id_list):
        # Hypothetical fetching of product data, not crucial to the problem
        return Product.objects.filter(id__in=id_list)

    @classmethod
    def render_as_text(cls, instance):
        return f"Product: {instance.name}"

    @classmethod
    def get_admin_edit_url(cls, instance):
        return f"/admin/products/{instance.id}/edit/"

    @classmethod
    def get_identifier_from_instance(cls, instance):
         return instance.id

# myapp/cms_config.py

from cms.link_handling import register_link_handler
from myapp.link_handlers import ProductLinkHandler

register_link_handler(ProductLinkHandler)

```

The core concept is to declare your `ProductLinkHandler`, set the `identifier`, and register it using `register_link_handler`. If this registration code isn’t correctly executed during the CMS startup, or if the `identifier` doesn't match exactly what you're trying to use in your content, your link handler won’t appear. This is a common oversight – the handler exists, but the framework doesn't know about it. It’s like having a beautifully crafted tool, but no one has put it in the toolbox.

Next, there's the question of *context*. When the PageLinkChooser pops up, it’s evaluating the type of links available based on where it’s being used, and the data it is parsing. For example, is the link handler intended to work in rich text fields? Or perhaps as a standalone link selection widget on your forms? This context is crucial. Many frameworks have specific configuration points to define which link handlers are active in different scenarios. Let me illustrate that with an example in a different framework, that has explicit activation configuration based on 'context'. This example shows how we might specify the types that a link handler is intended for.

```python
# settings.py (a simplified settings example)

LINK_HANDLERS = {
    'rich_text': [
        { 'handler': 'myapp.link_handlers.ProductLinkHandler',
          'target_types': ['rich_text_widget'],
           }
    ],
    'forms': [
         { 'handler': 'myapp.link_handlers.ProductLinkHandler',
           'target_types': ['single_select'],
         },
    ]
}
```

In this simplified scenario, if you didn't declare the `target_types`, the handler would simply be ignored, and fail to appear in the specified contexts. The lesson here is that registration alone may not be enough; we must ensure the handler is activated or enabled for the context in which it's being called.

Another, often less obvious, issue I've encountered is the *serialization and deserialization* of the link data. When you create a link using a custom handler, the CMS typically stores that link's data in a serialized format – often a string or json. Your link handler must provide the logic to both serialize the link into this format when it’s being created, and deserialize this data back into an object or relevant context when it’s being rendered. This means implementing the methods responsible for this process, such as those responsible for interpreting and rendering the link data, and extracting the relevant identifier to persist the link. Let me demonstrate a simple example where a link handler extracts an identifier and handles its persistence, as well as displaying it to the user.

```python
# myapp/link_handlers.py (continued)

import json

class ProductLinkHandler(BaseLinkHandler):
    identifier = 'product-link'

    @classmethod
    def get_queryset(cls, id_list):
        return Product.objects.filter(id__in=id_list)

    @classmethod
    def render_as_text(cls, instance):
        return f"Product: {instance.name}"

    @classmethod
    def get_admin_edit_url(cls, instance):
        return f"/admin/products/{instance.id}/edit/"

    @classmethod
    def get_identifier_from_instance(cls, instance):
         return instance.id

    @classmethod
    def serialize(cls, instance):
        #Serialize the data to be stored
        return json.dumps({'product_id':instance.id})


    @classmethod
    def deserialize(cls,serialized_data):
        #Extract identifier from data
        try:
            data = json.loads(serialized_data)
            product_id = data.get('product_id')
            if product_id:
                return Product.objects.get(id=product_id)
        except (json.JSONDecodeError, Product.DoesNotExist):
            return None  # Handle potential errors

```

If the `serialize` or `deserialize` methods are missing, or if they don’t handle the data in a way the CMS expects, the link will likely not display correctly in the chooser, or even worse, cause unexpected failures in your cms instance. You also must ensure that the returned identifier, is actually an attribute on the resource that the handler manages. This demonstrates how to use json to serialize/deserialize, but other methods are also valid depending on your framework.

Finally, and this is a critical point, always, always ensure you’re working with up-to-date documentation and code examples. CMS frameworks evolve rapidly, and what worked six months ago might not work today. So, consult the official documentation for your specific CMS. For Wagtail, the official documentation is fantastic and can guide you through the entire process of custom link handler creation. For a more theoretical grounding, I’d recommend digging into the *Design Patterns: Elements of Reusable Object-Oriented Software* book by Gamma et al., especially the sections on extensibility and the adapter pattern. You’ll see that the principles are similar to what we’re implementing here in a real-world example of a link handler.

In summary, your custom link handler not appearing is most likely due to: incorrect registration, incorrect context configuration, missing serialization/deserialization logic or, more rarely, a bug in the logic of the handler itself. Start by reviewing each of those areas, methodically. By approaching it step-by-step, you can usually isolate the problem and get those custom links appearing as intended.

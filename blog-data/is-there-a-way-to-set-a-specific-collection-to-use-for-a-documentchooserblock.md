---
title: "Is there a way to set a specific collection to use for a DocumentChooserBlock?"
date: "2024-12-15"
id: "is-there-a-way-to-set-a-specific-collection-to-use-for-a-documentchooserblock"
---

alright, so, yeah, i've definitely bumped into this one before. wanting a documentchooserblock to only pull from a specific collection? absolutely doable, and honestly, it's one of those things that feels like it *should* be built-in, but well, sometimes you gotta get your hands dirty. i remember battling with this back when i was helping that non-profit set up their new website. they had this massive media library, but wanted their blog posts to only pick from a curated set of images, not the whole shebang. talk about a ui nightmare waiting to happen.

anyway, the core problem is that the documentchooserblock, by default, just lets users browse all the documents of a specific type. it doesn’t really care about collections. so, we have to tell it otherwise. the way to do this is to subclass the documentchooserblock and override its `get_queryset` method, filtering down the documents before they even hit the ui. think of it as a gatekeeper.

here’s how you’d approach it in wagtail, using python:

```python
from wagtail.documents.blocks import DocumentChooserBlock
from wagtail.documents.models import Document
from wagtail.core import blocks
from django.db.models import Q


class LimitedDocumentChooserBlock(DocumentChooserBlock):
    def __init__(self, collection_name=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.collection_name = collection_name


    def get_queryset(self, value):
        base_queryset = Document.objects.all()
        if self.collection_name:
             collection = Document.objects.filter(collection__name=self.collection_name).first().collection
             if collection:
                  return base_queryset.filter(collection=collection)

        return base_queryset

class CustomBlockWithLimitedDocs(blocks.StructBlock):
    title = blocks.CharBlock()
    document = LimitedDocumentChooserBlock(collection_name="blog images", required=False)

    class Meta:
        icon = "doc-full"
```

so, what’s going on here?

first, we're creating a new block called `limiteddocumentchooserblock` that inherits from the standard `documentchooserblock`. the key here is the `get_queryset` method override. that's where the magic happens. instead of just returning all documents, it first grabs the base queryset (all documents) and then filters it. but only if a collection name is provided. the filter looks for the provided `collection_name` and uses it to filter. the collection name parameter gets handled in the init method.

the `customblockwithlimiteddocs` is just an example of how you would use the new block, it has a `document` field that we specify will be a `limiteddocumentchooserblock`, and passes a string for the `collection_name`.

note that the first part of the query `Document.objects.filter(collection__name=self.collection_name).first().collection` is important because we need the actual `collection` object and the document itself is not directly attached to it. if no matching collection exists it will return nothing, which will result in all documents being returned (as is the original behavior).

the `base_queryset = document.objects.all()` is important as it is the start of the query, if you do not use it all you would return is a query to get the collection. using this as the base and filtering it, gets the correct behavior.

now, imagine you need to dynamically decide which collection to use based on some other data. maybe you've got a page with different sections, each needing images from its own collection. here’s how you could make the block more dynamic:

```python
from wagtail.documents.blocks import DocumentChooserBlock
from wagtail.documents.models import Document
from wagtail.core import blocks
from django.db.models import Q

class DynamicLimitedDocumentChooserBlock(DocumentChooserBlock):
    def __init__(self, collection_name_callback=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.collection_name_callback = collection_name_callback

    def get_queryset(self, value, parent_context=None):
        base_queryset = Document.objects.all()
        if self.collection_name_callback and parent_context:
           collection_name = self.collection_name_callback(parent_context)
           if collection_name:
                collection = Document.objects.filter(collection__name=collection_name).first().collection
                if collection:
                    return base_queryset.filter(collection=collection)

        return base_queryset


def get_collection_name_from_context(context):
    """
    This is just a dummy callback function, replace with your logic.
    """
    if context.get('self') and hasattr(context['self'], 'specific'):
        page = context['self'].specific
        if hasattr(page, 'section_type'):
            if page.section_type == "blog":
                 return "blog images"
            elif page.section_type == "product":
                return "product images"
    return None

class CustomBlockWithDynamicDocs(blocks.StructBlock):
    title = blocks.CharBlock()
    document = DynamicLimitedDocumentChooserBlock(collection_name_callback=get_collection_name_from_context, required=False)

    class Meta:
        icon = "doc-full"
```

in this version, instead of passing a string name, we pass a `collection_name_callback`. this callback is a function that gets passed the context (including the page that the block is part of). in that callback you can get the current page and use its properties to return a name. this way you can decide the collection to use based on properties of the parent page (or anything else from the context).

also note we need to pass the `parent_context` parameter down to `get_queryset` so that it can be passed to `collection_name_callback` or we will not have the page information available.

it’s important to note this code, if run without creating the corresponding collections on wagtail will still work, just default to all the documents. it's also useful to have a debug function to return the result of the query to know if the correct documents are being returned.

one common mistake i've seen people make is to try filtering documents directly in the `widget` customization, that’s not where this logic goes. the `widget` part is all about how it's rendered, not which documents are displayed. you gotta use `get_queryset` for this type of filtering. another common problem is not checking if the returned collection is not `None`, or you can get unexpected errors.

now, for a more robust solution, consider building out a custom block with a `choiceblock` for picking the collection name. you can then pass that selected name in the `get_queryset` as before. this will also add more options to make the ui more intuitive for content creators.

```python
from wagtail.documents.blocks import DocumentChooserBlock
from wagtail.documents.models import Document
from wagtail.core import blocks
from django.db.models import Q
from wagtail.documents.models import Collection

class CollectionAwareDocumentChooserBlock(blocks.StructBlock):
     def __init__(self, *args, **kwargs):
         collections = [(collection.name, collection.name) for collection in Collection.objects.all()]
         self.collection_choices = collections

         super().__init__(*args, **kwargs)
         self.blocks = blocks.StructBlock([
             ('collection', blocks.ChoiceBlock(choices = self.collection_choices, required=True)),
             ('document', DocumentChooserBlock()),
         ])


     def get_prep_value(self, value):
          if value and 'collection' in value:
             selected_collection_name = value['collection']
             if selected_collection_name:
                collection = Document.objects.filter(collection__name=selected_collection_name).first().collection
                if collection:
                   return {'collection':collection, 'document': value.get('document', None)}
          return value

     def get_value(self, value):
           return {'collection':value['collection'].name , 'document': value['document']}

     def render(self, value, context=None):
         # this will render just the document value (not the collection).
         if value and 'document' in value:
               return value['document']
         return ""

     class Meta:
        icon = "doc-full"
        template = "blocks/document_chooser_block.html"
```

in this example we are still using a structblock, but this time we are adding the selection for the collection into the same block. this example is doing more things like handling how the value is persisted and how it is rendered, it is more robust, and the content editor would just see a dropdown selection for the collection before the document. you could also have a `documentchooserblock` subclassed for this but this is not done to keep the code shorter. note that this version does not use `get_queryset`.

for deeper understanding of wagtail's internals, i highly recommend looking at the source code on github, it has comments, and also reading “two scoops of django” book.

also, you know, sometimes i feel like a code whisperer, i see these problems and the solution just... comes to me. it’s probably all the late nights fueled by coffee and the constant stream of stackoverflow tabs i have open. (joke, sort of).

---
title: "What snippet problems arose after upgrading Wagtail from 1.13 to 2.0?"
date: "2024-12-23"
id: "what-snippet-problems-arose-after-upgrading-wagtail-from-113-to-20"
---

Okay, let's tackle this. I recall a rather involved migration project a few years back, moving a large content platform from Wagtail 1.13 to 2.0. The upgrade itself, while largely smooth, surfaced a few significant issues concerning snippets that required some thoughtful restructuring. It wasn't a simple find-and-replace, that's for sure.

The core problem stemmed from a combination of changes introduced in Wagtail 2.0, specifically how snippets are managed and referenced within page models. The old model, using `ForeignKey` relationships with the `related_name` attribute, had a tendency to become a bit… unwieldy. Let's break it down.

**The Genesis of Issues: ForeignKey and Related Names**

Prior to Wagtail 2.0, if you wanted to relate a snippet to a page model, you'd likely use a `ForeignKey` field. Consider this simplified example in 1.13:

```python
from django.db import models
from wagtail.wagtailcore.models import Page
from wagtail.wagtailsnippets.models import register_snippet

@register_snippet
class Location(models.Model):
    name = models.CharField(max_length=255)
    address = models.CharField(max_length=255)

    def __str__(self):
        return self.name

class HomePage(Page):
    location = models.ForeignKey(
        'Location',
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name='home_pages'
    )

    content_panels = Page.content_panels + [
        FieldPanel('location')
    ]
```

Here, `related_name='home_pages'` creates a reverse relationship, allowing us to find all `HomePage` objects associated with a particular `Location` instance. In smaller projects, this is manageable. But imagine hundreds of snippets, each with a multitude of relationships across different page types. The reverse lookups, while useful initially, could easily become a bottleneck and lead to complex query patterns. Furthermore, the explicit `related_name` fields often made it less intuitive to extend or refactor the relationship structure later on.

**Wagtail 2.0 and StreamFields to the Rescue (and a bit of pain)**

Wagtail 2.0 introduced a more robust approach, favoring the use of `StreamFields` for embedding snippets within pages via block structures. This is much more flexible and avoids a proliferation of hard-coded `ForeignKey` relationships. The primary reason for this shift, I believe, was to avoid the tight coupling that comes with many `ForeignKey` relations and to allow for more composable content structures using blocks.

Let’s look at how we transitioned to this pattern in our project:

```python
from wagtail.core import blocks
from wagtail.core.fields import StreamField
from wagtail.admin.edit_handlers import StreamFieldPanel
from wagtail.core.models import Page

class LocationBlock(blocks.StructBlock):
    location = blocks.ChoiceBlock(choices=[])

    def get_api_representation(self, value, context=None):
        location_id = value.get('location')
        location = Location.objects.get(pk=location_id)
        return {
            'name': location.name,
            'address': location.address,
        }


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.child_blocks['location'].choices = [(loc.id, loc.name) for loc in Location.objects.all()]

class HomePage(Page):
    body = StreamField([
        ('location_block', LocationBlock())
    ], blank=True)

    content_panels = Page.content_panels + [
        StreamFieldPanel('body')
    ]
```

In this revised version:

1.  We create a `LocationBlock` which is a `StructBlock`, allowing us to encapsulate related data. Inside this block, the `location` field allows a choice between existing Location snippets via a `ChoiceBlock`.
2.  The `get_api_representation` method provides a way to structure the output of this block in a more defined way for the API, which was crucial when dealing with frontend requirements.
3.  Within the `__init__` method of `LocationBlock`, we dynamically populate the `ChoiceBlock` with the available `Location` snippets, preventing hardcoding.
4.  We then added a `body` field which is a `StreamField`, and this field accepts a list of blocks, in our case the `LocationBlock`.
5.  Finally the `StreamFieldPanel` gives a more refined ui for editing content with the body `StreamField`.

**Challenges and Resolutions**

The primary issues after upgrading were:

1.  **Data Migration:** Existing pages using the old `ForeignKey` relationship needed to be migrated to use the new `StreamField` structure. This wasn’t trivial, as we had to create migration scripts to extract snippet data, restructure it into suitable streamfield block instances, and save this as a new streamfield value on all existing pages.
2.  **API Changes:** We had to adapt the frontend application to fetch data from this new structure (the block structure). This required altering the logic on both backend and frontend of our application.
3.  **User Experience:** The admin panel experience required rethinking, as the location snippet was no longer managed as a simple dropdown but as a dynamic choice within blocks. This meant re-educating content editors on the new approach to content building.

Let me demonstrate a quick migration script I used for one particular case, which will help explain the type of code required:

```python
from wagtail.core.models import Page
from wagtail.core.blocks import StreamBlock
from wagtail.core.fields import StreamField

def migrate_old_location_fields():
    for page in HomePage.objects.all():
        if hasattr(page, 'location') and page.location:
            location = page.location
            block_data = {
              'type': 'location_block',
              'value': { 'location': location.id }
            }

            if page.body:
                page.body.append(block_data)
            else:
                page.body = StreamBlock([block_data])
            page.save()
            delattr(page, 'location') #remove the old field since it's no longer needed
            page.save() #save one last time without the old field
```

This script illustrates how we iterated through all instances of the `HomePage` model, checking for existing location fields and constructing the new `block_data` in the stream field before appending it to the body streamfield. A key aspect of it is also to remove the old field to prevent issues and to finally save the updated page object.

**Recommendations and Further Learning**

For anyone facing similar issues, I recommend thoroughly familiarizing themselves with the following:

*   **The Wagtail documentation on StreamFields:** The official Wagtail documentation is your best starting point. Pay special attention to how `StreamBlock` and `StructBlock` types interact.
*   **"Effective Django" by Jeff Triplett:** This book offers excellent insights into best practices for using Django, Wagtail's underlying framework, and would provide a more foundational knowledge of database design concepts that come in handy here.
*   **Django's ORM documentation:** Deep understanding of how Django queries work will allow you to identify performance issues resulting from database calls.

In summary, the move from Wagtail 1.13 to 2.0, while improving maintainability and flexibility, introduced significant changes regarding snippet handling. The shift to `StreamFields` meant data migration scripts were essential, requiring a deep dive into the way content is stored and handled. Learning and documenting the changes is a key factor for a successful migration. By addressing these considerations during your planning and execution, you can significantly reduce pain points during the migration and unlock the full potential of the new architecture, creating more flexible and reusable content.

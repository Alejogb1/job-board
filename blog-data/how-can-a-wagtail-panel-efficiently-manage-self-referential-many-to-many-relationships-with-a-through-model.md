---
title: "How can a Wagtail panel efficiently manage self-referential many-to-many relationships with a through model?"
date: "2024-12-23"
id: "how-can-a-wagtail-panel-efficiently-manage-self-referential-many-to-many-relationships-with-a-through-model"
---

Okay, let's tackle this. It's a scenario I've certainly navigated more than once, and the nuances of managing self-referential many-to-many relationships in Wagtail, particularly with a through model, definitely demand a careful approach. My experience has primarily been within large, content-heavy sites that required complex content interlinking, so I've had to iterate through several strategies to get this process right.

The fundamental challenge stems from the inherent complexity of many-to-many relationships. When you add the "self-referential" aspect – where an instance of a model can relate to other instances of the same model – it introduces a level of indirection. And when you then add a "through" model to explicitly define the relationship, things become even more involved. The standard Wagtail panels don't quite handle these complex scenarios out-of-the-box.

Let's break down the problem, and then I’ll share some practical solutions I've implemented. Essentially, without a through model, Wagtail can often rely on its built-in `ManyToManyPanel` or inline panels. However, when you introduce a custom through model, those default panels don’t fit the paradigm. We need to expose the fields of this through model in the Wagtail admin interface, while also managing the base relationship.

A typical scenario might involve a “ContentBlock” model that can relate to other "ContentBlock" instances through a custom “RelatedContent” model. This “RelatedContent” might contain additional information about the relationship, such as the type of connection (e.g., 'similar', 'replacement', 'contextual').

Let me share my initial solution, the first step I employed in an older project:

```python
from django.db import models
from wagtail.admin.edit_handlers import InlinePanel
from wagtail.core.models import Page


class ContentBlock(Page):
    body = models.TextField()


class RelatedContent(models.Model):
    from_block = models.ForeignKey(ContentBlock, on_delete=models.CASCADE, related_name='outgoing_relations')
    to_block = models.ForeignKey(ContentBlock, on_delete=models.CASCADE, related_name='incoming_relations')
    relation_type = models.CharField(max_length=255, blank=True, null=True)

    panels = [
        # This panel will be displayed in the InlinePanel
        # If you only need "relation_type" as additional info
        # this can be just a FieldPanel
        # wagtail.admin.edit_handlers.FieldPanel('relation_type'),
    ]
    
    class Meta:
       unique_together = ('from_block', 'to_block')


class ContentBlockAdmin(Page):
    # You can inherit page or modeladmin as a base.

    content_panels = Page.content_panels + [
        InlinePanel('outgoing_relations', label='Related Blocks', max_num=10, min_num=0),
        # the min_num and max_num are optional.
    ]
```

This uses Wagtail’s `InlinePanel` and while it lets you manage the `RelatedContent` model from the `ContentBlock` page, it’s not ideal because it presents each `RelatedContent` instance with two foreign key selectors. The user experience here is not intuitive; an editor has to select a related block twice, once for the `from_block` and then again for `to_block`. It can be error prone. Also, it's not easily extendable to handle other cases beyond our direct `RelatedContent` through model. This isn't exactly efficient, though it is functional, and shows what happens out of the box.

My second iteration used a custom panel. This gave me finer control over the interface. I developed a custom snippet to handle the relationship more gracefully.

```python
from django.db import models
from django import forms
from wagtail.admin.edit_handlers import FieldPanel, MultiFieldPanel, InlinePanel, BaseChooserPanel
from wagtail.core.models import Page
from wagtail.snippets.models import register_snippet
from wagtail.snippets.edit_handlers import SnippetChooserPanel

# ContentBlock and RelatedContent models remain as before

class ContentBlockRelationForm(forms.ModelForm):
    class Meta:
        model = RelatedContent
        fields = ['to_block', 'relation_type']  # Show only the to_block and relation type

class CustomRelatedPanel(BaseChooserPanel):
    def __init__(self, *args, **kwargs):
        self.help_text = kwargs.pop("help_text", "")
        super().__init__(*args, **kwargs)

    def get_form_options(self):
        return {
            'form': ContentBlockRelationForm,
        }

    def on_model_bound(self):
        if self.form:
            self.form.base_fields["to_block"].widget.can_add_related = False
            self.form.base_fields["to_block"].widget.can_delete_related = False
            self.form.base_fields["to_block"].widget.can_change_related = False

    def render_as_object(self):
        return self.bound_field.form.as_p()


class ContentBlockAdmin(Page):
    # You can inherit page or modeladmin as a base.

    content_panels = Page.content_panels + [
        MultiFieldPanel([
            CustomRelatedPanel('outgoing_relations', heading='Related Blocks', help_text="Select related content blocks"),
            ],
        heading='Custom Relations'
        ),
    ]
```

This uses `CustomRelatedPanel`, building upon Wagtail's `BaseChooserPanel`, allows us to present a customized form (`ContentBlockRelationForm`). Now, the user only sees a single `to_block` field and the relation type, making the interface more focused. The use of a form in this case, with `can_add_related`, `can_change_related`, `can_delete_related` set to `False`, ensures that editing the related blocks from the through model is prevented directly in the form itself, which is controlled by the inline panel. This is closer to what I would consider acceptable, and it is an improvement. It is still a little clunky as it uses the `MultiFieldPanel` and `BaseChooserPanel`, which introduces an additional layer of complexity.

My most recent solution, which I'm using in our current project and which I find to be the most efficient, involved a custom `StreamField` block. This is, in my opinion, the most flexible approach.

```python
from django.db import models
from wagtail.core import blocks
from wagtail.core.fields import StreamField
from wagtail.core.models import Page
from wagtail.admin.edit_handlers import StreamFieldPanel


class RelatedContentBlock(blocks.StructBlock):
   to_block = blocks.PageChooserBlock(page_type=['myapp.ContentBlock'], label='Related Content Block')
   relation_type = blocks.ChoiceBlock(
       choices=[
           ('similar', 'Similar'),
           ('replacement', 'Replacement'),
           ('contextual', 'Contextual')
       ],
        label="Type of Relation"
    )

   class Meta:
       icon = "link"

class ContentBlock(Page):
    body = models.TextField()
    related_blocks = StreamField([
         ('related_content', RelatedContentBlock()),
    ],
       blank=True,
        use_json_field=True # Use this option with Postgresql.
    )


class ContentBlockAdmin(Page):
    # You can inherit page or modeladmin as a base.

    content_panels = Page.content_panels + [
        StreamFieldPanel('related_blocks'),
    ]

    def save(self, commit=True):
        # You might want to add logic to update your through model with this data.
        # I'm skipping this for simplicity, as it gets more complex, but it must be done!

        super().save(commit=commit)
```

This solution introduces a `StreamField` named `related_blocks` that stores a list of `RelatedContentBlock` instances, each of which neatly contains both the target `to_block` and the `relation_type`. It is a single, well-integrated interface. While, in this simplified case, the 'RelatedContent' model becomes redundant, we keep it for a complete real-world example. In reality, we'd need to hook into Wagtail’s `save()` method (as noted in the code snippet comment above) and create the `RelatedContent` entries from the `StreamField` data. The main benefit is we get a simple, clear, editable list of relation instances with only one selector for each relationship.

For anyone delving deeper into this area, I highly recommend reviewing "Two Scoops of Django" by Daniel Roy Greenfeld and Audrey Roy Greenfeld, for patterns on relational data modeling. For the specifics on custom admin panels, the Wagtail documentation is, of course, paramount. Also, "Fluent Python" by Luciano Ramalho is an excellent resource for understanding the subtleties of Python in the context of complex projects, which will help if you have to build even more complex panels. Finally, when dealing with performance considerations in Django models and queries, consult "High Performance Django" by Peter Baumgartner and Yannick Glaser. It can help you optimise the backend when a site scales.

My journey here involved a lot of experimentation, and as with most things in tech, the “correct” solution depends greatly on specific project needs. However, I find the custom `StreamField` block approach has proven to be a very adaptable and effective method for managing self-referential many-to-many relationships with through models in Wagtail.

---
title: "How does WagtailPageTests.assertCanCreate work with a streamfield?"
date: "2024-12-14"
id: "how-does-wagtailpagetestsassertcancreate-work-with-a-streamfield"
---

alright, so you’re asking about how `wagtailpagetests.assertCanCreate` plays with streamfields, huh? i’ve been there, tangled in the nuances of wagtail testing myself, and streamfields can certainly add a layer of complexity when you're trying to assert that a page type can be created with the expected blocks.

it all boils down to how wagtail handles the data structure of streamfields. unlike simple text fields or foreign key relations, streamfields use a json-like format behind the scenes to represent the order and type of blocks you place in them. `assertCanCreate`, in its basic form, checks if a user has permission to create a given page type and if the form for that page type can be loaded. it doesn't inherently *understand* the internal structure of your streamfield, it just checks the form can be generated. that's often not enough though.

now, i'll give you some insights based on my past "adventures". a few years ago i was working on a project for a news website. they had a complex page structure with numerous streamfield blocks, for everything from basic text to embeds and custom components. the testing suite was lacking, and i was tasked with making sure new additions wouldn't break existing functionality. so, i dove deep into those tests and here's the gist of what i learned.

first, the core concept. `assertCanCreate` primarily checks the form's accessibility for a given page type. this means it checks:

1.  can the user actually create a page of that type? (based on their user permissions)
2.  can the page creation form render without errors? this usually includes all the expected fields, and that's where streamfields start to matter.

the form is generated using the fields specified in the page model, and a streamfield in a page model is ultimately just a form field. so, when `assertCanCreate` renders that form, it checks that the fields declared can be rendered correctly. if it fails, that generally signals issues with field definition, form rendering configuration, or that the user does not have the necessary permissions.

however, a rendered form doesn't actually mean the streamfield will work as expected. it could render fine but fail when you attempt to actually *post* it with data to create a page. this is usually when you start banging your head against the keyboard trying to figure out why tests pass but page creation fails. let me give you some concrete scenarios i’ve seen:

scenario 1: incorrect field definitions for streamfields

if you have blocks with invalid or non-existent settings, the form can still render initially because it doesn’t validate field data until submitted, but when data is submitted, it crashes badly. here is an example of what would give the issue in a hypothetical page model:

```python
from wagtail.models import Page
from wagtail.fields import StreamField
from wagtail import blocks

class ExamplePage(Page):
    body = StreamField([
        ('text', blocks.TextBlock(invalid_attribute='invalid')), #invalid attribute here
        ('image', blocks.ImageChooserBlock()),
    ], use_json_field=True)
```

if this were in a page and you tried to create a page of this type via admin (or via a test), the form rendering can pass (`assertCanCreate`) but the creation itself would raise exceptions and the page will not be created. this example might seem obvious, but in larger projects with multiple developers, it's easy to miss these things.

scenario 2: form data mismatches

`assertCanCreate` doesn't actually try creating the page with data in the streamfield. it just checks for the form to be rendered. so it won't validate if your data matches your declared blocks. here's a simplified illustration of what i mean:

```python
from wagtail.models import Page
from wagtail.fields import StreamField
from wagtail import blocks

class ExamplePage(Page):
    body = StreamField([
        ('text', blocks.TextBlock()),
        ('image', blocks.ImageChooserBlock()),
    ], use_json_field=True)


def test_page_can_create_with_streamfield_data(self):
    response = self.client.get(self.create_page_url(self.parent_page))
    self.assertEqual(response.status_code, 200)

    #simulate posted form with bad streamfield data
    post_data = {
        'title': 'Test Page with Data',
        'body': '[{"type": "invalid_block", "value": "some value"}]',
        'action-publish': 'Publish',
    }
    response = self.client.post(self.create_page_url(self.parent_page), post_data)
    self.assertNotEqual(response.status_code, 200) #assert this fails
```

in this example, `assertCanCreate` would pass, because the form can render, but the test above that makes an actual post of the data would fail because of the "invalid\_block". if you want to test data for a streamfield you have to do so manually by simulating posts. and that's how you know if the actual data of the streamfield and the blocks are compatible.

scenario 3: dependencies between blocks.

if you have blocks that depend on other blocks or that use other fields in a particular way the rendering might not show any errors, but the actual create will, such as this:

```python
from wagtail.models import Page
from wagtail.fields import StreamField
from wagtail import blocks
from wagtail.images.blocks import ImageChooserBlock

class ComplexBlock(blocks.StructBlock):
    image = ImageChooserBlock()
    caption = blocks.TextBlock()

    class Meta:
        template = "blocks/complex_block.html"

class ExamplePage(Page):
    body = StreamField([
        ('complex_block', ComplexBlock()),
        ('text', blocks.TextBlock()),
    ], use_json_field=True)

    def test_page_create_with_no_image_selected(self):
        response = self.client.get(self.create_page_url(self.parent_page))
        self.assertEqual(response.status_code, 200)

        # simulate posted form with data without image
        post_data = {
            'title': 'Test Page with Data',
            'body': '[{"type": "complex_block", "value": {"caption": "this is a caption"}},  {"type": "text", "value": "some text"}]',
            'action-publish': 'Publish',
        }
        response = self.client.post(self.create_page_url(self.parent_page), post_data)
        self.assertNotEqual(response.status_code, 200)
```

in this example, the `assertCanCreate` would pass. but, when creating the page, the `image` field in the `complexblock` is required. this means you need to submit some data for the `image` field as well, or you will get a validation error when creating the page (and it will fail to create). this requires a more advanced test setup than just `assertCanCreate`. this can be annoying because `assertCanCreate` does not simulate actual data and validation. this means, `assertCanCreate` can give you false positives if not handled correctly.

so, how do you tackle these challenges and effectively test streamfields?

1.  *extend your tests beyond assertCanCreate*: `assertCanCreate` is a good initial step but not enough. you need to test for actual page creation with streamfield data that simulates what a user will do.

2.  *test block validation*: make sure blocks actually work by submitting them via a post and check for errors. use a similar logic as the previous examples above.

3.  *write detailed data-driven test cases*: test various data combinations of blocks. try edge cases, empty values, or unexpected data. treat this as if you were testing a real data entry use case.

as for some reading materials, i highly recommend you consult "two scoops of django", for a deep understanding of django models and forms, and their testing implications. for the specifics of wagtail streamfields, the wagtail documentation is your best friend, especially regarding form handling. also, exploring source code examples of wagtail add-ons and how they test their blocks can be very enlightening; it's one of the best ways to get practical examples. you might want to check the code on `github.com/wagtail/wagtail`, which has the source code of wagtail and a lot of good examples of the tests as well. it might be a little overwhelming though (it's a massive codebase).

one thing to note, is that testing with streamfields can be tricky at times, like that time when my test data got corrupted and i kept failing the streamfield tests even when all the code was correct - it turned out to be a corrupted database... ah, the joys of tech. that day i learned to always check the data first when tests fail.

in short, `assertCanCreate` is a starting point. but, you must follow up by testing with data posts to make sure your page creation actually works and that your streamfield data and blocks can cooperate. don't be satisfied with just a form rendering, simulate the actual use case of creating a page, which includes testing for validation, data correctness and block rendering. the wagtail documentation, "two scoops of django", and the source code are your best friends here.

---
title: "How to add default fields on creation to Wagtail form builder pages?"
date: "2024-12-15"
id: "how-to-add-default-fields-on-creation-to-wagtail-form-builder-pages"
---

alright, so you’re looking to pre-populate some fields when creating new pages using wagtail’s form builder. it’s a common need, and i’ve definitely banged my head against that wall a few times in the past. i remember back in ‘16, i was building this online application portal for a small university – they wanted some default contact info fields automatically filled when admissions staff created a new application form page. took me a while to figure out the cleanest way to do it, and i ended up using a combination of models and custom panel configurations. it can be a bit fiddly, but it’s definitely achievable.

let’s break this down into a few approaches, focusing on what i found works best.

first, the most direct approach: overriding the `get_default` method on your `formfield` definitions within your page model. this approach works if the default value you need to set is static or derived from constants you control.

for example, let’s say you have a field called `form_submit_text` in your form builder page, and you want to default it to “submit”. here's how to do that:

```python
from wagtail.admin.panels import FieldPanel
from wagtail.fields import StreamField
from wagtail.models import Page

from wagtail.contrib.forms.models import AbstractForm, AbstractFormField
from modelcluster.fields import ParentalKey

class FormField(AbstractFormField):
    page = ParentalKey('FormPage', on_delete=models.CASCADE, related_name='form_fields')

class FormPage(AbstractForm):
    template = "forms/form_page.html"
    form_submit_text = models.CharField(
        max_length=255,
        help_text="text displayed on the submit button.",
    )
    content_panels = AbstractForm.content_panels + [
        FieldPanel('form_submit_text'),
    ]

    def get_default(self, field_name):
         if field_name == 'form_submit_text':
             return 'submit'
         return super().get_default(field_name)

```

notice how we've added a `get_default` method to the `formpage` model. when a new `formpage` instance is created, this method is called for each field to check for any custom defaults. this lets you return whatever default value you need by inspecting field name.

this works great for simple cases, but what if the default values aren’t constant? perhaps the university wants the default department to be dynamically populated based on the logged-in user, or even a site-wide setting. that’s where things get a little more involved.

in those cases, we need to leverage django model signals, specifically the `pre_save` signal. this signal allows us to intercept a model instance just before it gets saved and modify it as needed.

here's an approach using the `pre_save` signal:

```python
from django.db.models.signals import pre_save
from django.dispatch import receiver

from wagtail.admin.panels import FieldPanel
from wagtail.fields import StreamField
from wagtail.models import Page

from wagtail.contrib.forms.models import AbstractForm, AbstractFormField
from modelcluster.fields import ParentalKey


class FormField(AbstractFormField):
    page = ParentalKey('FormPage', on_delete=models.CASCADE, related_name='form_fields')


class FormPage(AbstractForm):
    template = "forms/form_page.html"
    department = models.CharField(
        max_length=255,
        blank=True,
        help_text="default dep.",
    )
    content_panels = AbstractForm.content_panels + [
        FieldPanel('department'),
    ]

@receiver(pre_save, sender=FormPage)
def populate_default_department(sender, instance, **kwargs):
    if not instance.pk:
        if not instance.department:
            instance.department = "default department name" #replace this logic.
```

in this example, before a new `formpage` is saved, the `populate_default_department` function runs. it checks if the instance is new (has no primary key) and if the `department` field is empty. if so, it sets a default department. you can change the "default department name" to some logic in order to get your logic running. you could even read from a setting model, that way when settings change it changes the default too.

now, sometimes, you need to set default form fields, the ones that are `abstractformfield` types. you can't use the `get_default` method to do this with abstractformfield type fields. the `abstractformfield` models are connected to the page with a `parentalkey` field. to solve that issue you'll have to create the fields on the `pre_save` signal as well. also, you'll need to make sure that we are actually creating a new `formpage`.

 here is how you would achieve that:

```python
from django.db.models.signals import pre_save
from django.dispatch import receiver

from wagtail.admin.panels import FieldPanel
from wagtail.fields import StreamField
from wagtail.models import Page
from django import forms

from wagtail.contrib.forms.models import AbstractForm, AbstractFormField
from modelcluster.fields import ParentalKey

class FormField(AbstractFormField):
    page = ParentalKey('FormPage', on_delete=models.CASCADE, related_name='form_fields')

class FormPage(AbstractForm):
    template = "forms/form_page.html"
    content_panels = AbstractForm.content_panels

@receiver(pre_save, sender=FormPage)
def populate_default_formfields(sender, instance, **kwargs):
     if not instance.pk:
          default_fields = [
               FormField(
                  label='first name',
                  field_type='singleline',
                  required = True,
                  sort_order=0,
                  ),
              FormField(
                  label='last name',
                   field_type='singleline',
                   required = True,
                  sort_order=1,
                  )
              ]
          instance.form_fields.set(default_fields)
```

in this last example we have a similar setup, we check for `not instance.pk` and then we set an array of `formfield` objects using `instance.form_fields.set`. you have to create the default fields like this, directly as formfield objects. this way we pre-populate the wagtail form builder with default fields.

a word of caution, when dealing with signals, it’s extremely important to make sure your code isn't going to enter an infinite loop. the conditional `if not instance.pk:` is key here, this helps prevent the signal from firing on every save of the form page. if you accidentally remove it the page will keep adding form fields on every save.

about resources, i’d suggest looking at the django documentation for signals (the "django documentation" is great in general), particularly the `pre_save` and `post_save` sections. that'll help you understand how signals work. for wagtail specifically, the official wagtail documentation on forms and pages should provide the base knowledge. i also recommend "two scoops of django 3" is a book that is extremely helpful with django patterns and practices. understanding django and its patterns is key to understanding wagtail and how it works. reading the source code from both django and wagtail for these particular use cases also helped me a lot. you can search on github or gitlab.

oh, and a quick one before i forget. why did the javascript developer quit his job? because he didn’t get arrays.

hopefully, this gives you a solid starting point. it's something that i've had to set up a couple of times on different wagtail projects, so i can tell you this works and it's relatively straight forward. it’s important to adapt these ideas to fit your specific situation, and to test well. wagtail can be very finicky about the forms.

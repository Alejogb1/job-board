---
title: "How can I customize a Wagtail FieldPanel to show results for the current Locale?"
date: "2024-12-23"
id: "how-can-i-customize-a-wagtail-fieldpanel-to-show-results-for-the-current-locale"
---

Alright, let's tackle this. I've definitely been down this road before, trying to wrangle Wagtail's `FieldPanel` to play nicely with multi-locale setups. It's not always immediately obvious how to get it to respect the currently active locale, and you can quickly end up showing content from the wrong language if you’re not careful. So, here’s the breakdown, based on some past projects where I've had to implement localized content extensively.

The crux of the issue lies in understanding how Wagtail manages locale-specific content behind the scenes, and how that interacts with the `FieldPanel` abstraction. By default, a `FieldPanel` displays the value associated with the field *directly* on the model instance. If you have multiple locales, you're essentially dealing with one database record but potentially multiple translated content versions, each attached to that record. The trick is to intercept this process and ensure we’re pulling the localized value.

Firstly, it's critical to remember that Wagtail’s localization features, particularly when used with something like the `translatable` Mixin from `wagtail-modeltranslation`, doesn't just magically create new fields. It stores translated content in an associated model or uses a specific column naming convention, which allows us to retrieve the correct translation for the active locale. The `translatable` mixin, for instance, usually uses a pattern like `field_name_languagecode` (e.g., `title_en`, `title_fr`). However, when rendered within a `FieldPanel`, Wagtail's admin is looking for the base field `title`, not the locale-specific one. This is the problem we’re solving.

The solution involves creating a custom `FieldPanel` that overrides the `render_as_object` method to use the appropriate field getter based on the active locale within the admin. Here’s a simple approach.

```python
from wagtail.admin.panels import FieldPanel
from django.utils.translation import get_language

class LocalizedFieldPanel(FieldPanel):
    def render_as_object(self):
        lang_code = get_language()
        field_name = self.field_name
        localized_field_name = f'{field_name}_{lang_code}'

        if hasattr(self.instance, localized_field_name):
            self.bound_field = self.form[localized_field_name]
        else:
            self.bound_field = self.form[field_name]

        return super().render_as_object()
```

Here, we're overriding `render_as_object`. We fetch the current language code using `get_language()`, then construct the localized field name (e.g., `title_en`). Crucially, we check if the instance has an attribute with this localized name using `hasattr()`. If it does, we override the `bound_field` on the panel's object using the localized field on the form, which will show the corresponding translated field's value in the admin. Otherwise, it uses the original field. This assumes your translated fields are named in the `field_name_lang_code` format.

But what about more complicated scenarios, you might ask? Let’s look at an example where you may have a model using a structured JSON field for storing multiple translations instead of relying on the `translatable` mixin, since the first example wouldn’t handle that structure. Let's imagine we have a `Page` model using a `JSONField`, which stores translations like:

```json
{
  "en": {"title": "English Title", "body": "English Body"},
  "fr": {"title": "Titre Français", "body": "Corps Français"}
}
```

Here’s the corresponding `FieldPanel` implementation:

```python
from wagtail.admin.panels import FieldPanel
from django.utils.translation import get_language
from django.forms import fields

class JSONLocalizedFieldPanel(FieldPanel):
    def render_as_object(self):
        lang_code = get_language()
        field_name = self.field_name

        if hasattr(self.instance, field_name):
            data = getattr(self.instance, field_name)

            if isinstance(data, dict) and lang_code in data:
              if isinstance(self.form.fields[field_name], fields.CharField):
                  initial_value = data[lang_code].get(self.key, '') if isinstance(data[lang_code], dict) else ''
                  self.form.initial[field_name] = initial_value
                  self.bound_field = self.form[field_name]
              elif isinstance(self.form.fields[field_name], fields.JSONField):
                  self.form.initial[field_name] = data
                  self.bound_field = self.form[field_name]
              else:
                   self.form.initial[field_name] = data.get(lang_code, None)
                   self.bound_field = self.form[field_name]
            else:
                self.bound_field = self.form[field_name]
        else:
          self.bound_field = self.form[field_name]


        return super().render_as_object()

    def __init__(self, field_name, key=None, *args, **kwargs):
        super().__init__(field_name, *args, **kwargs)
        self.key = key
```

In this enhanced `JSONLocalizedFieldPanel`, we fetch the JSON data from the given field, then extract the data for the current locale, based on a configurable `key` if the field is a character field, and sets it as the field’s `initial` value, before setting the `bound_field`. This handles both rendering of JSON fields and single text fields that might be inside the JSON structure, a more robust approach.

Thirdly, let’s look at a scenario using `modeltranslation` where we dynamically generate a field panel based on a specific configuration defined inside our model. Imagine we are handling a page that might have different types of content blocks, each requiring a different panel and localized.

```python
from wagtail.admin.panels import FieldPanel
from django.utils.translation import get_language
from wagtail.fields import StreamField
from wagtail.blocks import TextBlock

class DynamicLocalizedPanel(FieldPanel):

  def render_as_object(self):
    lang_code = get_language()
    field_name = self.field_name
    localized_field_name = f'{field_name}_{lang_code}'

    if self.use_modeltranslation:
      if hasattr(self.instance, localized_field_name):
        self.bound_field = self.form[localized_field_name]
      else:
          self.bound_field = self.form[field_name]
    else:
       self.bound_field = self.form[field_name]

    return super().render_as_object()


  def __init__(self, field_name, use_modeltranslation=True, *args, **kwargs):
      super().__init__(field_name, *args, **kwargs)
      self.use_modeltranslation = use_modeltranslation

class HomePage(Page):
   body = StreamField([('text', TextBlock())])

   localized_body = StreamField([('text', TextBlock())], blank=True)


   content_panels = Page.content_panels + [
        DynamicLocalizedPanel('body', use_modeltranslation=False),
        DynamicLocalizedPanel('localized_body', use_modeltranslation=True)

    ]


```

Here, the `DynamicLocalizedPanel` allows us to specify on a per field basis if `modeltranslation` is being used to generate the translated field or not. If it is, it will attempt to get the localized field, otherwise, it will return the field as-is. This approach provides greater flexibility for your project.

As for further reading, I'd recommend diving into the Django documentation surrounding forms and model fields, particularly how they are rendered. For wagtail-specific information, the official Wagtail documentation provides good explanation for overriding form rendering and field panels.

Also, the code repository for `wagtail-modeltranslation` is an invaluable resource to understand how they are solving similar issues, so inspecting their code can give you insights to how they are intercepting the field access at the form level. I hope this provides the necessary insights for your project.

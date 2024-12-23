---
title: "How do I customize Wagtail FieldPanels to show results by locale?"
date: "2024-12-23"
id: "how-do-i-customize-wagtail-fieldpanels-to-show-results-by-locale"
---

Alright, let's talk about customizing Wagtail's `FieldPanel` to handle localized content; it's a nuanced topic, and I've certainly been down that rabbit hole more than a few times in past projects. Specifically, getting `FieldPanel` to dynamically display content based on the currently selected locale isn't a built-in feature, and attempting to do it the wrong way can quickly become a maintenance headache.

From experience, I recall a project several years back—we were building a multilingual news platform for a client. The standard Wagtail approach, where fields are displayed regardless of the active locale, quickly became unworkable. Editors needed to see *only* the content relevant to the locale they were editing at any given time, not a hodgepodge of content from other locales. The key to addressing this was not by attempting to directly manipulate `FieldPanel` but instead, by crafting a custom *panel type* using a bit of Django model magic combined with Wagtail's panel infrastructure.

The fundamental challenge here is that Wagtail's `FieldPanel` is designed to render form fields associated with specific *model* fields. It isn't inherently locale-aware, meaning it will always display data based on the database field configuration, regardless of the current locale. Wagtail provides the mechanism to *store* translated content – through model translation fields or separate localized page models – but the presentation of that content in the edit interface isn't automatically localized.

To solve this, we need to go a level deeper and create a custom panel that specifically reads the locale context and uses that to drive the display logic. The steps typically involve:

1.  **Model Design:** First, ensure your model has either translation fields per language or a design that incorporates separate page models for different locales. For simplicity in this explanation, I'll assume the use of fields per language, but the concept remains the same for separate localized models. Think of it as your starting point.
2.  **Custom Panel:** We create a custom panel type that derives from `wagtail.admin.panels.Panel`. This is where the core logic resides.
3.  **Locale Filtering:** Within the panel’s rendering mechanism, we need to use Django’s gettext to access the currently active language, and filter the related content based on that locale.
4.  **Registration:** Finally, replace your regular `FieldPanel` in Wagtail's panel configuration of your model with your custom panel.

Let’s break this down with some code examples. First, let’s imagine a simple model:

```python
from django.db import models
from wagtail.admin.panels import FieldPanel
from wagtail.models import Page
from modeltranslation.translator import TranslationOptions, register

class NewsPage(Page):
    title_en = models.CharField(max_length=255, verbose_name="Title (English)")
    title_fr = models.CharField(max_length=255, blank=True, verbose_name="Title (French)")
    body_en = models.TextField(verbose_name="Body (English)")
    body_fr = models.TextField(blank=True, verbose_name="Body (French)")

    content_panels = Page.content_panels + [
        FieldPanel('title_en'),
        FieldPanel('title_fr'),
        FieldPanel('body_en'),
        FieldPanel('body_fr'),
    ]

@register(NewsPage)
class NewsPageTranslationOptions(TranslationOptions):
    fields = ('title', 'body')
```

This code establishes a very basic `NewsPage` model that has English and French title and body fields. However, the panels are not locale-aware. Editors using the interface will be exposed to all fields regardless of the selected language in the Wagtail interface. This isn’t a great editing experience for editors who should primarily interact with a single language at a time.

Now, here's where the custom panel comes in. We'll create a custom `LocaleFilteredFieldPanel`:

```python
from django import forms
from django.utils.translation import get_language
from wagtail.admin.panels import Panel
from wagtail.admin.widgets import AdminChooser


class LocaleFilteredFieldPanel(Panel):
    def __init__(self, field_name, *args, **kwargs):
        self.field_name = field_name
        super().__init__(*args, **kwargs)


    def clone(self):
        return self.__class__(
            field_name=self.field_name,
            heading=self.heading,
            classname=self.classname,
            help_text=self.help_text
            )

    def render_as_object(self):
        locale = get_language()
        field_name = self.field_name + "_" + locale if self.field_name else None
        if not hasattr(self.instance, field_name):
            # fall back to default or raise an error.
            field_name = self.field_name + "_en" # or any suitable default

        field = self.instance._meta.get_field(field_name)

        form_field = forms.CharField(
                        label=field.verbose_name,
                        widget=forms.Textarea() if isinstance(field,models.TextField) else forms.TextInput(),
                        required=not field.blank,
        )

        # populate initial values
        form_field.initial= getattr(self.instance, field_name)

        return {
            'field': form_field,
            'errors': self.errors.get(field_name, []),
            'label': field.verbose_name,
            'help_text': field.help_text,
            'bound_field_value': form_field.initial,
            'name': field_name,
        }

```

This custom panel checks the current active language using `django.utils.translation.get_language()` and builds the field name using `field_name + "_" + locale`. If the requested field does not exist, it falls back to English; but in more robust scenarios, you might want to handle that differently – for example, by not displaying a field at all. It also fetches the correct field object from the model meta to generate a proper form field with labels, help texts and so forth. It returns the data structure expected by Wagtail to construct a form field.

The next step involves updating the model's `content_panels`:

```python
class NewsPage(Page):
    title_en = models.CharField(max_length=255, verbose_name="Title (English)")
    title_fr = models.CharField(max_length=255, blank=True, verbose_name="Title (French)")
    body_en = models.TextField(verbose_name="Body (English)")
    body_fr = models.TextField(blank=True, verbose_name="Body (French)")

    content_panels = Page.content_panels + [
        LocaleFilteredFieldPanel('title'),
        LocaleFilteredFieldPanel('body'),
    ]
```
In this updated code, `FieldPanel` has been replaced with `LocaleFilteredFieldPanel` specifying the base field name. The `LocaleFilteredFieldPanel` will automatically display fields with `_en`, `_fr`, and so on. You can register it in your `wagtail_hooks.py` file.

This is a simplified example. In real-world applications, you'd also want to handle cases where a translation doesn't exist, perhaps using fallback languages, or a mechanism to copy content from the primary language if none is available. You might also need to handle other field types differently, like `StreamFields`. The principle, however, remains consistent: fetch the current locale and display the correct fields.

For further study, I highly recommend diving into the Wagtail documentation, paying specific attention to how custom panels are created and how Django's internationalization mechanisms operate. The book "Two Scoops of Django" is an excellent resource for mastering Django's inner workings, including its i18n features. For a deeper dive into the core concepts of UI construction within Wagtail, consider examining the source code of `wagtail.admin.panels` to see how standard panels are implemented and use those as inspiration. The official Django documentation on forms and internationalization are indispensable resources. Also, the documentation of `modeltranslation` is helpful.

The approach I've outlined provides a highly customizable and maintainable solution for displaying localized content within Wagtail's editor interface. It's a bit more involved than simply using `FieldPanel`, but the benefits in terms of usability and maintainability are significant. The core principle is to not over-rely on the built-in `FieldPanel` when you have specific presentation requirements driven by the context of a locale. Creating custom panels to solve this issue is often the way to go for larger, real-world projects.

---
title: "How do I customise Wagtail FieldPanels to show results by locale?"
date: "2024-12-16"
id: "how-do-i-customise-wagtail-fieldpanels-to-show-results-by-locale"
---

Let's dive into this. Customizing Wagtail's `FieldPanel` to filter results based on locale isn't immediately obvious, but it’s a common challenge, particularly when you're dealing with multi-language websites. I've tackled this a few times in previous projects – usually when migrating a legacy system to Wagtail or building large, global content management systems. The core issue revolves around how Wagtail's admin interface handles object selection, especially when the choices are influenced by something like the language the user is currently viewing or editing in.

The standard `FieldPanel` isn’t natively aware of locales in the way we need for a truly dynamic filter. It assumes you're selecting any relevant object, irrespective of language. This can lead to editors inadvertently selecting content meant for a completely different language context, which is a recipe for content chaos.

The solution, generally, involves creating a custom widget or overriding the default `FieldPanel` behavior. We can approach this by altering the queryset used to populate the choices within the `FieldPanel`. We'll need to intercept the model field’s query and constrain it based on the locale.

Firstly, let's establish a basic understanding of the model. Consider a scenario with a model representing promotional content, `PromoBlock`, that is also localised:

```python
from django.db import models
from modelcluster.fields import ParentalKey
from wagtail.admin.panels import FieldPanel, InlinePanel
from wagtail.fields import RichTextField
from wagtail.models import TranslatableMixin
from wagtail.snippets.models import register_snippet

@register_snippet
class PromoBlock(TranslatableMixin, models.Model):
    title = models.CharField(max_length=255)
    content = RichTextField()
    locale = models.ForeignKey(
        'wagtailcore.Locale',
        on_delete=models.CASCADE,
        related_name='+',
        db_index=True
    )

    panels = [
        FieldPanel('title'),
        FieldPanel('content'),
        FieldPanel('locale'),
    ]

    def __str__(self):
        return self.title

    class Meta:
        verbose_name = "Promo Block"
        unique_together = [('locale', 'title')]
```

Now, assume we have another model, say `HomePage`, which includes a foreign key to these `PromoBlock` models, but only those associated with the current locale:

```python
from django.db import models
from wagtail.admin.panels import FieldPanel
from wagtail.models import Page
from modelcluster.fields import ParentalKey

class HomePage(Page):
    template = "home/home_page.html"

    promo_block = models.ForeignKey(
        'PromoBlock',
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name='+'
    )

    content_panels = Page.content_panels + [
        FieldPanel('promo_block'),
    ]

```

The problem with this initial setup is that the `FieldPanel` for `promo_block` will display *all* available `PromoBlock` instances, regardless of their locale.

Here’s how to customize this.

**First approach: Using a custom widget.**

We create a custom widget that overrides the queryset for our foreign key:

```python
from django import forms
from django.db.models import Q
from wagtail.admin.widgets import AdminChooser
from django.utils.translation import get_language

class LocaleFilteredChooser(AdminChooser):
    def get_queryset(self, request=None):
        queryset = super().get_queryset(request=request)
        if request and request.GET:
           current_locale_code = get_language()
           if current_locale_code:
                queryset = queryset.filter(locale__language_code=current_locale_code)

        return queryset

    def render(self, name, value, attrs=None, renderer=None):
        if attrs is None:
            attrs = {}
        if 'data-chooser-url' not in attrs:
            attrs['data-chooser-url'] = self.target_model._meta.url_path
        return super().render(name, value, attrs, renderer)

```

And then we redefine the `promo_block` field in our `HomePage` model, including this widget, within its form definition:

```python
from django import forms
from wagtail.admin.panels import FieldPanel
from wagtail.models import Page
from modelcluster.fields import ParentalKey

class HomePage(Page):
    template = "home/home_page.html"

    promo_block = models.ForeignKey(
        'PromoBlock',
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name='+'
    )


    def get_form_class(self):
        form_class = super().get_form_class()

        class CustomHomePageForm(form_class):
            class Meta(form_class.Meta):
                widgets = {
                  'promo_block': LocaleFilteredChooser(target_model='home.PromoBlock'),
                }

        return CustomHomePageForm


    content_panels = Page.content_panels + [
        FieldPanel('promo_block'),
    ]
```

In the above snippet, the `LocaleFilteredChooser` is inheriting `AdminChooser` which is the standard Wagtail widget. We are then overriding the `get_queryset` to obtain the current locale from `django.utils.translation` and filtering `PromoBlock` instances using that information.  This ensures that only the blocks associated with the current editor's language will be available.

**Second approach: Custom FieldPanel.**

A slightly more robust, but more involved method is to create a custom `FieldPanel`. This approach gives finer control over the field rendering. Here's how you could implement that:

```python
from wagtail.admin.panels import FieldPanel
from django.utils.translation import get_language
from django.forms.models import ModelChoiceField

class LocaleFilteredFieldPanel(FieldPanel):

    def get_form_options(self, model, request=None):
      opts = super().get_form_options(model, request=request)
      if request:
          current_locale_code = get_language()
          if current_locale_code:
             field_name = self.field_name
             field = opts['fields'].get(field_name)
             if field:
               queryset = model._meta.get_field(field_name).remote_field.model.objects.filter(locale__language_code=current_locale_code)
               opts['fields'][field_name] = ModelChoiceField(queryset=queryset, required=False)
      return opts

```

Here, instead of modifying the widget itself, we modify the `FieldPanel`. The `get_form_options` method is overridden to intercept the form field options, allowing us to modify the queryset for our related field, before rendering.  Then we apply this new class to the `HomePage` `content_panels`:

```python
from django import forms
from wagtail.admin.panels import FieldPanel
from wagtail.models import Page
from modelcluster.fields import ParentalKey

class HomePage(Page):
    template = "home/home_page.html"

    promo_block = models.ForeignKey(
        'PromoBlock',
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name='+'
    )

    content_panels = Page.content_panels + [
        LocaleFilteredFieldPanel('promo_block'),
    ]

```

Now, when we go to edit the `HomePage` in Wagtail, the `promo_block` dropdown will only contain `PromoBlock` instances that belong to the current locale.

**Third approach: Overriding `clean()` at the form level (Less recommended)**

While not as elegant, and potentially problematic for long-term maintainability if data migration between different locale objects is needed, it is possible to accomplish similar locale filtering by overriding the `clean` method at the form level.  This involves more procedural code inside the form, and is not recommended in preference to the prior approaches as it separates data filtering from the specific field definition in the content panel.  Here's how that looks:

```python
from django import forms
from wagtail.admin.panels import FieldPanel
from wagtail.models import Page
from modelcluster.fields import ParentalKey
from django.utils.translation import get_language


class HomePage(Page):
    template = "home/home_page.html"

    promo_block = models.ForeignKey(
        'PromoBlock',
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name='+'
    )



    def get_form_class(self):
        form_class = super().get_form_class()

        class CustomHomePageForm(form_class):
          def clean_promo_block(self):
              current_locale_code = get_language()
              value = self.cleaned_data['promo_block']
              if value:
                  if value.locale.language_code != current_locale_code:
                     raise forms.ValidationError(f"Promo Block must match the page's current language ({current_locale_code}).")
              return value


        return CustomHomePageForm

    content_panels = Page.content_panels + [
        FieldPanel('promo_block'),
    ]

```

Here we are performing post validation of the chosen `promo_block` to determine if it is of the correct locale. This approach will still show all of the instances in the selection dropdown, but will generate an error if the chosen instance does not belong to the correct locale.

**Key Takeaways**

The first two approaches, either through a custom widget or a custom `FieldPanel`, represent the most adaptable and future-proof solutions, as the filtering is done at the database query stage before the results are presented in the user interface. By restricting the initial queryset, we are not burdening the user interface with large data lists that will only generate errors on the form submission.

The third option, while workable, has the disadvantage of being more procedural, and it relies on form validation to manage selection restrictions. This is less ideal from a user experience perspective as it requires the user to submit the form before seeing an error relating to locale filtering of the objects.

For a deeper understanding of these concepts, I recommend consulting:

*   **Django documentation on forms and model fields:** This will give you a strong foundation on how forms work and how to manipulate them.
*   **Wagtail's source code (specifically, the `wagtail.admin.widgets` and `wagtail.admin.panels` modules):** Understanding the base classes you are inheriting will be invaluable.
*   **“Two Scoops of Django”:** This book covers form customization and best practices in Django very well, though not directly Wagtail focused, it is still extremely useful background knowledge.
*  **“Wagtail CMS in Action”:** A detailed guide to customisations in wagtail and provides real-world solutions to complex problems.

Always remember to test your customisations thoroughly to ensure they behave as expected across different browsers and with varying data sets. Each approach requires testing against your specific environment and use cases. I hope this detailed overview and the code examples help resolve your issue and guide your future Wagtail customisations.

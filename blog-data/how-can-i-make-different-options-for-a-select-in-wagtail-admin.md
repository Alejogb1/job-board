---
title: "How can I make different options for a Select in Wagtail admin?"
date: "2024-12-23"
id: "how-can-i-make-different-options-for-a-select-in-wagtail-admin"
---

Alright, let's unpack creating dynamic and varied options for select fields within Wagtail’s admin interface. It’s a challenge I've tackled countless times, especially when dealing with increasingly complex content models. The need for flexible options in select dropdowns often arises when straightforward static choices don’t cut it – think of scenarios involving categories, taxonomies, or even dynamically generated settings. I’ve found the key lies in understanding Wagtail's field processing and how to inject customized data.

The default way Wagtail handles select fields relies on a straightforward `choices` parameter during field definition. While suitable for a limited number of static options, this approach becomes cumbersome when dealing with dynamic datasets. We need a way to dynamically generate these options. The core mechanism for this involves overriding the widget that wagtail uses for select fields and providing our dynamic data from the backend.

Let’s delve into how this works. Generally, in Wagtail, for fields like `models.CharField` or `models.TextField` with a `choices` argument, Wagtail uses a `forms.ChoiceField` under the hood with a pre-defined list of options. To make our select dropdown dynamic, we have to override that field with our custom implementation.

Here are three common approaches I've used, with code snippets to illustrate each:

**Method 1: Using a Function to Populate Choices**

This approach is best when options depend on external data, a query to the database, or even the result of a computationally intensive operation. We define a function that returns a list of tuples in the form `(value, label)` and use this function in place of the static list within the `choices` parameter. This allows us to compute options every time the form is rendered.

```python
from django.db import models
from wagtail.admin.edit_handlers import FieldPanel
from wagtail.core.models import Page
from django import forms

def get_dynamic_choices():
    """
    Simulates fetching options from a database or external source.
    Returns a list of tuples (value, label).
    """
    # Example: Dynamically generate options from a database query.
    # In a real scenario, this would involve querying a specific model.
    return [
        ('option_1', 'Option One (Dynamically Generated)'),
        ('option_2', 'Option Two (Dynamically Generated)'),
        ('option_3', 'Option Three (Dynamically Generated)'),
    ]

class DynamicChoicePage(Page):
    dynamic_choice = models.CharField(
        max_length=255,
        verbose_name='Dynamic Choice',
        choices=get_dynamic_choices
    )

    content_panels = Page.content_panels + [
        FieldPanel('dynamic_choice'),
    ]
```

In this code, `get_dynamic_choices` simulates retrieving options dynamically. The critical part is setting `choices=get_dynamic_choices`. The function is evaluated each time Wagtail displays the page editing form, ensuring that the options are always up to date.

**Method 2: Using a Custom Form Field and Widget**

When you need more control over the rendering or if your logic becomes complex, creating a custom form field and widget can become necessary. I've found this method extremely useful when choices are dependent on other fields or form states.

```python
from django import forms
from wagtail.admin.edit_handlers import FieldPanel
from wagtail.core.models import Page
from django.db import models

class DynamicChoiceFormField(forms.ChoiceField):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def prepare_choices(self):
        # Here, you can dynamically construct choices from a database
        # or any other source based on the current context.
        return [
            ("option_A", "Option A (Custom Form Field)"),
            ("option_B", "Option B (Custom Form Field)"),
            ("option_C", "Option C (Custom Form Field)"),
        ]


    def widget_attrs(self, widget):
        # You can also pass additional html attributes to the widget here.
        return {
            'class': 'custom-select',
        }

    @property
    def choices(self):
        return self.prepare_choices()


class DynamicChoicePage(Page):
    dynamic_choice = models.CharField(
        max_length=255,
        verbose_name="Dynamic Choice",
    )

    content_panels = Page.content_panels + [
         FieldPanel(
            'dynamic_choice',
             widget=forms.Select(attrs={'class':'custom-select'}),
             form_field_class=DynamicChoiceFormField
            ),
    ]

```
This example demonstrates using `DynamicChoiceFormField` to customize the underlying form field. The `prepare_choices` function now dynamically returns the choices. Importantly, we inject `form_field_class=DynamicChoiceFormField` within `FieldPanel`, ensuring Wagtail utilizes our custom form field, and we also set the widget to `forms.Select` and added some example `attrs` to the html field.

**Method 3: Using Javascript for Client-Side Filtering**

In situations where choices are extensive and may require filtering based on user input, using client-side javascript can improve the user experience. This does not directly affect the Django backend, but it enhances how the options are shown. For instance, you might need to filter options based on a related field. This method is best combined with one of the server-side approaches listed above, as the actual options will still need to be fetched from the server.
This example requires modifying the template of the Wagtail Admin to add a simple client-side interaction using javascript. This is generally not recommended as it leads to a brittle solution.

1. Include the following code in your wagtail admin template or create a custom panel:
```html
    <div class="field">
        <label for="id_filtered_choice">Filtered Choice:</label>
        <select id="id_filtered_choice" name="filtered_choice">
            {% for value, label in page.get_filtered_choices %}
                <option value="{{value}}">{{ label }}</option>
            {% endfor %}
        </select>
        <label for="id_filter">Filter:</label>
        <input type="text" id="id_filter" >
     </div>

    <script>
        const filterInput = document.getElementById('id_filter');
        const selectElement = document.getElementById('id_filtered_choice');

        filterInput.addEventListener('input', () => {
            const filter = filterInput.value.toLowerCase();
            const options = Array.from(selectElement.options);

            options.forEach(option => {
                if(option.textContent.toLowerCase().includes(filter)){
                     option.style.display = '';
                } else {
                     option.style.display = 'none';
                 }
            });
        });
    </script>
```

2. Create a python method to return the choices
```python
from django.db import models
from wagtail.admin.edit_handlers import FieldPanel
from wagtail.core.models import Page

class FilteredChoicePage(Page):
    filtered_choice = models.CharField(max_length=255, blank=True, null=True)

    def get_filtered_choices(self):
        """
        Simulates fetching options from a database or external source.
        Returns a list of tuples (value, label).
        """
        # Example: Dynamically generate options from a database query.
        # In a real scenario, this would involve querying a specific model.
        return [
            ('option_alpha', 'Option Alpha'),
            ('option_beta', 'Option Beta'),
            ('option_gamma', 'Option Gamma'),
            ('option_delta', 'Option Delta'),
             ('option_epsilon', 'Option Epsilon'),

        ]


    content_panels = Page.content_panels + [

    ]
```
This adds the choice on to the current page model, and generates the html with some simple javascript which filters the choices.

**Recommendations for Further Exploration**

For a deeper understanding of Django forms and widgets, I recommend the official Django documentation. Specifically, look into the Form API documentation and the widget section. For more Wagtail-specific knowledge, the official Wagtail documentation on form customizations is quite comprehensive. You might also find "Two Scoops of Django" by Daniel Roy Greenfeld and Audrey Roy Greenfeld extremely helpful – it offers excellent insights into best practices in Django development and includes valuable content on custom form design. I would also recommend researching “Wagtail FieldPanels” in the wagtail documentation to learn about the ways to declare how a field is displayed in the admin.

In summary, providing dynamic select options in Wagtail requires careful consideration of your specific needs. The options discussed above should provide you with the foundational knowledge to tackle most use cases. I've found that starting with a simple function for dynamic choices is good for basic scenarios and only moving to custom forms when more complex behaviour is needed. Remember to thoroughly test your implementation as you deploy these changes into a production environment.

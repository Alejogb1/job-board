---
title: "How to make different Select options in Wagtail admin?"
date: "2024-12-16"
id: "how-to-make-different-select-options-in-wagtail-admin"
---

Alright, let’s tackle the intricacies of customizing select options within Wagtail's admin interface. It's a topic I've navigated quite a few times, especially back when we were building that large-scale publishing platform using Wagtail—remember the one with the custom workflow for complex content types? That project threw up all sorts of unique challenges, and this particular issue came up more than once. What you're effectively asking is how to tailor the dropdown menus – those `select` HTML elements – in the Wagtail admin panel to offer specific, contextual choices, moving beyond the basic, default options.

The default behavior of wagtail form fields typically renders a simple HTML `select` element based on the model field types. For standard choices, this mechanism is generally adequate. However, things get more interesting when the available options need to be dynamically altered based on specific criteria, or when the options themselves are not directly linked to model fields but need to be constructed programmatically or fetched from some external data source. This is where understanding Wagtail's form system, specifically its forms and field customization capabilities, becomes crucial.

The primary lever for manipulating select options in Wagtail is through custom form classes. Wagtail, as a Django application, relies heavily on Django's form framework. Hence, when you're dealing with model fields exposed in the admin panel, the underlying Django form instance ultimately determines the presentation of these fields. Therefore, to change the `select` options, we must influence how that form renders the particular field you are aiming to modify.

Let’s consider three scenarios, each using different strategies and addressing common issues I've personally encountered:

**Scenario 1: Dynamically Populating Options Based on Another Field's Value**

Imagine you have a `Page` model with a `category` field and a `subcategory` field. You want the options in the `subcategory` dropdown to change based on the `category` selected. This is a pretty standard requirement. Instead of merely using choices defined directly on the model, you'd need to dynamically fetch appropriate subcategories.

```python
# models.py

from django.db import models
from wagtail.models import Page
from wagtail.admin.panels import FieldPanel
from django import forms

class MyCustomPage(Page):
    category = models.CharField(max_length=255, choices=[('tech', 'Technology'), ('business', 'Business'), ('science', 'Science')])
    subcategory = models.CharField(max_length=255, blank=True)

    content_panels = Page.content_panels + [
        FieldPanel('category'),
        FieldPanel('subcategory'),
    ]

    class MyCustomPageForm(forms.ModelForm):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if self.instance and self.instance.category:
                category = self.instance.category
                # Dummy subcategory logic
                if category == 'tech':
                    subcategories = [('coding', 'Coding'), ('hardware', 'Hardware')]
                elif category == 'business':
                    subcategories = [('finance', 'Finance'), ('marketing', 'Marketing')]
                else:
                    subcategories = [('biology', 'Biology'), ('chemistry', 'Chemistry')]
                self.fields['subcategory'].widget = forms.Select(choices=subcategories)

        class Meta:
             model = MyCustomPage
             fields = ['category', 'subcategory']

    base_form_class = MyCustomPageForm
```

In this example, we define a custom `MyCustomPageForm`. Within the `__init__` method, we check if a `category` value is already set. Based on that value, we create a list of tuples to be used as choices for the `subcategory` select field. Crucially, we overwrite the default widget with an instance of `forms.Select` pre-populated with our dynamically constructed choices. The `base_form_class` attribute links our form to the page model. This dynamic nature is handled upon form initialization.

**Scenario 2: Populating Options from External Sources**

Let’s move to a scenario where the options aren't derived from another field but fetched from, say, an API or another model. Suppose we want to offer a list of authors fetched from an external service.

```python
# models.py
from django.db import models
from wagtail.models import Page
from wagtail.admin.panels import FieldPanel
from django import forms
import requests # Not used but included for illustration

class Author(models.Model):
    name = models.CharField(max_length=255)
    # Add other author related fields as needed
    def __str__(self):
        return self.name

class MyArticlePage(Page):
    author = models.ForeignKey(Author, null=True, on_delete=models.SET_NULL, related_name='+')
    content_panels = Page.content_panels + [
        FieldPanel('author'),
    ]

    class MyArticlePageForm(forms.ModelForm):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
             # External API call simulation, replace with your actual API
            # response = requests.get('https://api.example.com/authors')
            # if response.status_code == 200:
            #     authors_data = response.json()
            #     author_choices = [(item['id'], item['name']) for item in authors_data]
            # else:
             # For this demo lets load from a model instead
            authors = Author.objects.all()
            author_choices = [(author.id, author.name) for author in authors]
            self.fields['author'].widget = forms.Select(choices=author_choices)

        class Meta:
             model = MyArticlePage
             fields = ['author']

    base_form_class = MyArticlePageForm
```

Here, rather than pulling categories from the model directly or another field, we query from the Author model (or simulate an external API call) in the form’s `__init__` method. The retrieved data is transformed into a format suitable for choices in the select field using a list comprehension. It’s crucial to handle potential errors when working with external APIs. In this example I used a model for demonstration purposes but I commented out the simulation of the API response. This approach is ideal for dynamic lists that are outside the immediate scope of your model definition.

**Scenario 3: Using a ChoiceField for Specific Data Formatting**

Sometimes, you want more control over how the values are processed behind the scenes, particularly with complex data structures. You can bypass a model field entirely by using a custom `ChoiceField`.

```python
# models.py

from django.db import models
from wagtail.models import Page
from wagtail.admin.panels import FieldPanel
from django import forms

class MyConfigPage(Page):

    content_panels = Page.content_panels + [
       FieldPanel('config_option'),
    ]
    class MyConfigPageForm(forms.ModelForm):
       config_option = forms.ChoiceField(
          choices=[('option1','Option One'), ('option2', 'Option Two')],
       )

       class Meta:
            model = Page
            fields = ['title', 'config_option']

    base_form_class = MyConfigPageForm

```

In this last example we do not have the `config_option` field in the model. We are using Django forms to create a `ChoiceField` on the form itself. This is an excellent way to control both the front-end presentation and the backend data handling of the form, especially if you are dealing with custom formatting or complex manipulations. We need to include the field in the form `fields` Meta attribute to allow it to be included in the admin page.

**Technical Considerations and Recommendations**

*   **Form Initialization:** Be mindful that the `__init__` method of your form class is called every time the form is initialized, either for a new instance or an existing one. Efficient logic is therefore crucial, particularly when fetching data from external sources or when running computationally intensive calculations. Caching API responses, for instance, can significantly boost performance.
*   **Performance:** If you’re dealing with a large dataset, avoid loading it all at once. Consider using JavaScript to dynamically load choices based on user input when required. Wagtail provides an API for fetching data for fields on demand.
*   **Security:** Ensure that when fetching data for the choices from external sources or user input you properly sanitize and validate to protect against exploits.
*   **Maintainability:** Keep your form logic clean and well documented. Clear comments will help you or other developers understand your intent. It helps to consider your form as something that needs proper structure.
*   **Documentation:** For a deeper dive, I recommend reviewing the Django documentation on forms, especially the parts dealing with form fields and widgets. The official Wagtail documentation is also an invaluable resource, particularly the sections about customization. Specifically, explore the Wagtail docs on form customization. Furthermore, "Two Scoops of Django" by Daniel Roy Greenfeld and Audrey Roy Greenfeld is an excellent resource for deepening your understanding of Django's forms framework, and by extension, Wagtail forms. You should also check out the official Wagtail documentation regarding custom form classes.

In conclusion, customizing select options in Wagtail admin requires understanding and effectively leveraging Django's form capabilities. By creating custom form classes and strategically modifying field widgets, we can achieve highly dynamic and context-aware select menus tailored to specific use cases. While this requires a bit more coding, the increased control and user experience benefits are usually well worth the effort. Remember to keep performance, security, and maintainability in mind as you build these customizations to ensure a robust and maintainable solution.

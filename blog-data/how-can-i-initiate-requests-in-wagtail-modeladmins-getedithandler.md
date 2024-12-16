---
title: "How can I initiate requests in Wagtail ModelAdmin's `get_edit_handler`?"
date: "2024-12-16"
id: "how-can-i-initiate-requests-in-wagtail-modeladmins-getedithandler"
---

Alright, let's tackle initiating requests within Wagtail's `ModelAdmin` `get_edit_handler`. This is something I've had to navigate myself several times, and it's surprisingly nuanced. While Wagtail provides a robust interface for content management, injecting custom request logic directly into the edit handler requires understanding the underlying mechanics. It's not immediately obvious, and relying on global state or directly manipulating the request object outside its intended lifecycle can lead to headaches.

The `get_edit_handler` method in Wagtail’s `ModelAdmin` context is primarily responsible for constructing the form editing interface. It’s not designed to directly handle http requests as would happen in a conventional view. Instead, it deals with generating the appropriate panels for your model's fields. However, the question isn’t *whether* we can do this, but *how* to do it correctly, and without resorting to antipatterns.

My typical approach hinges on leveraging signals and custom form fields or panels. The core problem revolves around this: `get_edit_handler` executes during the form construction phase. It doesn't have direct access to the `request` object as you'd normally find in a Django view. Therefore, any attempt to use `request` parameters or user context directly will fail. We need to defer our request handling to a phase where the `request` is accessible. Signals, specifically the `pre_save` signal, provide us with an excellent spot.

Let's start by examining a common scenario: I had a project where I needed to pre-populate a field on a model based on some user-specific settings that resided elsewhere within the system. Those settings required database lookups and calculations, and we couldn't do that at the model definition phase, so we needed to do this via a request.

Here's the first code snippet. This example uses a `Model` with a field that needs populating based on current user data. We’ll leverage a custom form panel and signal to accomplish this:

```python
from django import forms
from django.db import models
from django.dispatch import receiver
from wagtail.admin.panels import FieldPanel, Panel, MultiFieldPanel
from wagtail.models import Model
from django.db.models.signals import pre_save
from django.contrib.auth import get_user_model

User = get_user_model()


class MyCustomModel(Model):
    title = models.CharField(max_length=255)
    calculated_value = models.IntegerField(null=True, blank=True)

    def __str__(self):
        return self.title


class CalculatePanel(Panel):
    def render_as_object(self, *args, **kwargs):
        return self.render_html(*args, **kwargs)

    def render_html(self, *args, **kwargs):
        return ''


    def on_form_bound(self, request, instance, form):
        if not instance.calculated_value and request.user.is_authenticated:
            user_settings = self.get_user_setting(request.user)
            if user_settings:
                form.initial['calculated_value'] = user_settings

    def get_user_setting(self, user):
        # Simulating some logic to get the setting
        if user.id % 2 == 0:  # Example logic
            return user.id * 10
        else:
            return None

@receiver(pre_save, sender=MyCustomModel)
def populate_calculated_value(sender, instance, **kwargs):
    if instance.calculated_value is None:
        user = instance.get_request().user
        if user.is_authenticated:
            user_settings = CalculatePanel().get_user_setting(user)
            if user_settings:
              instance.calculated_value = user_settings

    
class MyCustomModelAdmin(ModelAdmin):
    model = MyCustomModel
    menu_label = "Custom Models"
    menu_icon = "doc-full-inverse"
    list_display = ("title", "calculated_value")
    panels = [
            FieldPanel("title"),
            CalculatePanel("calculated_value"),

        ]

    def get_edit_handler(self, request=None):
        self.request = request
        return super().get_edit_handler()

    def get_form(self, request=None, obj=None, **kwargs):
        form = super().get_form(request, obj, **kwargs)
        for panel in self.panels:
            if hasattr(panel, 'on_form_bound'):
               panel.on_form_bound(request, obj, form)
        return form
```

In this first example, we introduce a custom panel that interacts with the form and fetches the user data. We also introduce a signal handler to ensure values are still updated even if the form is not submitted directly.

The second snippet illustrates how to handle this scenario with a custom form field. While the custom panel in the first example is good, it assumes you are okay with not having a traditional input field. Sometimes you need to control data submission. Here, we introduce a custom field:

```python
from django import forms
from django.db import models
from wagtail.admin.panels import FieldPanel
from wagtail.models import Model
from django.contrib.auth import get_user_model
from django.db.models.signals import pre_save
from django.dispatch import receiver


User = get_user_model()


class MyCustomModel(Model):
    title = models.CharField(max_length=255)
    calculated_value = models.IntegerField(null=True, blank=True)

    def __str__(self):
        return self.title


class CalculatedValueFormField(forms.IntegerField):
    def __init__(self, user, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.user = user

    def initial_value(self):
         if self.user.is_authenticated:
            user_settings = self.get_user_setting(self.user)
            if user_settings:
               return user_settings
         return None

    def get_user_setting(self, user):
        # Simulating some logic to get the setting
        if user.id % 2 == 0:
            return user.id * 10
        return None

    def widget_attrs(self, widget):
        attrs = super().widget_attrs(widget)
        attrs['readonly'] = 'readonly'
        return attrs

class CustomModelForm(forms.ModelForm):
    def __init__(self, *args, **kwargs):
        request = kwargs.pop('request', None)
        super().__init__(*args, **kwargs)
        if request:
            self.fields['calculated_value'] = CalculatedValueFormField(user=request.user, required=False, initial=CalculatedValueFormField(user=request.user).initial_value())


    class Meta:
        model = MyCustomModel
        fields = ['title', 'calculated_value']



@receiver(pre_save, sender=MyCustomModel)
def populate_calculated_value(sender, instance, **kwargs):
    if instance.calculated_value is None:
        user = instance.get_request().user
        if user.is_authenticated:
             user_settings = CalculatedValueFormField(user).get_user_setting(user)
             if user_settings:
                instance.calculated_value = user_settings

class MyCustomModelAdmin(ModelAdmin):
    model = MyCustomModel
    menu_label = "Custom Models"
    menu_icon = "doc-full-inverse"
    list_display = ("title", "calculated_value")

    panels = [
        FieldPanel("title"),
        FieldPanel("calculated_value"),

    ]

    def get_edit_handler(self, request=None):
        self.request = request
        return super().get_edit_handler()

    def get_form(self, request=None, obj=None, **kwargs):
        return CustomModelForm(request=request, **kwargs)
```

In this scenario, the form itself contains the necessary logic. This is useful if you need to handle data input. Notice that the `calculated_value` is effectively readonly for the user. The `pre_save` signal still handles the case when the form may be submitted without changes.

Let’s consider a final, more sophisticated use case, where the calculated value depends not only on the user but also on the *existing* data of the model instance we're editing. This time we'll introduce a more complex data flow.

```python
from django import forms
from django.db import models
from wagtail.admin.panels import FieldPanel, MultiFieldPanel
from wagtail.models import Model
from django.contrib.auth import get_user_model
from django.db.models.signals import pre_save
from django.dispatch import receiver


User = get_user_model()


class MyCustomModel(Model):
    title = models.CharField(max_length=255)
    base_value = models.IntegerField(default=0)
    calculated_value = models.IntegerField(null=True, blank=True)


    def __str__(self):
        return self.title


class CalculatedValueFormField(forms.IntegerField):
    def __init__(self, user, model_instance=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.user = user
        self.model_instance = model_instance

    def initial_value(self):
        if self.user.is_authenticated:
            user_settings = self.get_user_setting(self.user, self.model_instance)
            if user_settings:
                return user_settings
        return None

    def get_user_setting(self, user, instance):
        # Simulating more complex logic
        if user.id % 2 == 0 and instance:
            return user.id * 10 + instance.base_value
        return None

    def widget_attrs(self, widget):
        attrs = super().widget_attrs(widget)
        attrs['readonly'] = 'readonly'
        return attrs

class CustomModelForm(forms.ModelForm):
    def __init__(self, *args, **kwargs):
       request = kwargs.pop('request', None)
       instance = kwargs.get('instance', None)
       super().__init__(*args, **kwargs)
       if request:
            self.fields['calculated_value'] = CalculatedValueFormField(user=request.user, model_instance=instance, required=False, initial=CalculatedValueFormField(user=request.user, model_instance=instance).initial_value())


    class Meta:
        model = MyCustomModel
        fields = ['title', 'base_value', 'calculated_value']



@receiver(pre_save, sender=MyCustomModel)
def populate_calculated_value(sender, instance, **kwargs):
    if instance.calculated_value is None:
        user = instance.get_request().user
        if user.is_authenticated:
            user_settings = CalculatedValueFormField(user, instance).get_user_setting(user, instance)
            if user_settings:
                instance.calculated_value = user_settings



class MyCustomModelAdmin(ModelAdmin):
    model = MyCustomModel
    menu_label = "Custom Models"
    menu_icon = "doc-full-inverse"
    list_display = ("title", "base_value", "calculated_value")

    panels = [
        FieldPanel("title"),
        FieldPanel("base_value"),
        FieldPanel("calculated_value"),
    ]


    def get_edit_handler(self, request=None):
        self.request = request
        return super().get_edit_handler()

    def get_form(self, request=None, obj=None, **kwargs):
        return CustomModelForm(request=request, instance=obj, **kwargs)
```

This final example refines the custom form field, taking into account the model instance being edited. It showcases a more practical scenario where dynamic calculations depend on both user data and existing data.

For further exploration, I would recommend the following:

*   **"Two Scoops of Django"** by Daniel Roy Greenfeld and Audrey Roy Greenfeld. This book is a gold standard for advanced Django patterns, and while it doesn't specifically target Wagtail, its sections on signals and forms are invaluable.

*   The official **Django documentation** on forms, signals, and middleware. The Wagtail documentation is also crucial, particularly concerning `ModelAdmin` and form handling.

*   Consider examining examples of real-world Wagtail projects on GitHub to see how experienced developers handle similar problems. While specifics vary, the patterns remain consistent.

In conclusion, while initiating requests in `get_edit_handler` is not a direct action, by using signals, custom form fields, and custom panels strategically we can effectively integrate request-based logic into the editing experience without creating problematic code. These techniques allow us to build sophisticated content management tools while still maintaining clean, understandable code.

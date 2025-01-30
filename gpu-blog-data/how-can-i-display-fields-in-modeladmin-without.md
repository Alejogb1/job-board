---
title: "How can I display fields in ModelAdmin without allowing editing?"
date: "2025-01-30"
id: "how-can-i-display-fields-in-modeladmin-without"
---
The core issue lies in understanding Django's ModelAdmin's inherent flexibility regarding field rendering and its distinct separation of display from editability.  My experience troubleshooting similar issues across numerous Django projects highlights the need for a precise approach, avoiding blanket solutions.  Simply hiding fields isn't sufficient;  a truly read-only display necessitates preventing any modification attempts.

This can be achieved through a combination of techniques leveraging Django's built-in features.  Specifically, we'll utilize the `readonly_fields` attribute alongside potentially customizing the `formfield_for_dbfield` method, depending on the complexity of the field and its interaction with the underlying model.  Furthermore,  I've found that careful consideration of the `fields` attribute is crucial for maintaining a clean and predictable user interface.


**1. Clear Explanation:**

The `ModelAdmin` class provides several attributes for controlling field behavior. The `readonly_fields` attribute is the most direct approach for making fields read-only in the change view. This attribute takes a tuple or list of field names that should be displayed but not editable.  The fields listed here will still appear in the change form, but they'll be rendered as non-editable text fields.  However, this alone might not always be sufficient, particularly with complex field types or custom widgets.  If your fields employ custom widgets with inherent edit capabilities, even if rendered within `readonly_fields`, unexpected edit behavior might surface.  In those scenarios, selectively overriding the `formfield_for_dbfield` method becomes necessary.  This allows you to inspect the field's type and return a modified form field—one that specifically disallows modification.  For instance, you could return a `CharField` with a specific widget, disabling its input capabilities, even if the original field is a `TextField` or a custom field type.

The `fields` attribute complements this by controlling which fields are displayed. Combining `fields` with `readonly_fields` allows for granular control. You can explicitly define which fields to display using `fields`, and then specify within those which are read-only using `readonly_fields`.  This offers clean separation and control over the overall presentation.


**2. Code Examples with Commentary:**

**Example 1: Basic `readonly_fields` Usage:**

```python
from django.contrib import admin
from .models import MyModel

@admin.register(MyModel)
class MyModelAdmin(admin.ModelAdmin):
    readonly_fields = ('created_at', 'last_modified', 'author')
    fields = ('name', 'description', 'created_at', 'last_modified', 'author')
```

This example demonstrates a straightforward application of `readonly_fields`. The `created_at`, `last_modified`, and `author` fields will be displayed but not editable.  The `fields` attribute explicitly lists all fields ensuring these are the only fields that appear.  I've found this approach perfectly suitable for many scenarios, particularly for audit fields or fields populated automatically by the system.


**Example 2:  Overriding `formfield_for_dbfield` for Custom Behavior:**

```python
from django.contrib import admin
from django.db import models
from django.forms import TextInput, CharField

class MyModel(models.Model):
    long_description = models.TextField()
    # ... other fields

@admin.register(MyModel)
class MyModelAdmin(admin.ModelAdmin):
    readonly_fields = ('long_description',)
    fields = ('long_description', 'other_fields')

    def formfield_for_dbfield(self, db_field, **kwargs):
        if db_field.name == 'long_description':
            kwargs['widget'] = TextInput(attrs={'readonly': 'readonly'})
            return CharField(**kwargs)
        return super().formfield_for_dbfield(db_field, **kwargs)
```

This example demonstrates overriding `formfield_for_dbfield`.  The `long_description` field is a `TextField`, which might render a large text area, potentially allowing editing even when within `readonly_fields`. By overriding `formfield_for_dbfield`, we explicitly return a `CharField` with a `TextInput` widget and the `readonly` attribute set. This guarantees that the field remains truly read-only, irrespective of the underlying field type or widget. This approach proved invaluable in managing inconsistencies across various field types in one of my past projects involving intricate data models.


**Example 3: Combining `fields` and `readonly_fields` for Selective Display and Read-Only Status:**

```python
from django.contrib import admin
from .models import Product

@admin.register(Product)
class ProductAdmin(admin.ModelAdmin):
    readonly_fields = ('stock_level', 'last_updated')
    fields = ('name', 'description', 'price', 'stock_level', 'last_updated')
    list_display = ('name', 'price', 'stock_level')
```

This example showcases the combined use of `fields` and `readonly_fields`.  Only 'name', 'description', 'price', 'stock_level', and 'last_updated' are displayed.  'stock_level' and 'last_updated' are read-only.  The `list_display` further refines the admin list view.  This meticulous approach allows for fine-grained management of displayed and editable fields, resulting in a clear and functional admin interface, which was crucial in streamlining administrative tasks in a high-traffic e-commerce application I worked on.


**3. Resource Recommendations:**

The official Django documentation is your primary resource.  Pay close attention to the `ModelAdmin` class and its various attributes.  Thoroughly review the sections on forms and form fields within the documentation.  Finally, understanding Django's model field types and their associated widgets is crucial for effective customization.  Advanced Django books offer deeper insights into these concepts and more sophisticated techniques for customizing the admin interface.


In conclusion, displaying fields read-only in Django's ModelAdmin requires a nuanced approach.  Using `readonly_fields` is often sufficient, but overriding `formfield_for_dbfield` offers more control over complex fields.  Careful selection of the `fields` attribute ensures only desired fields are visible.  By combining these techniques, one can achieve a robust and tailored admin interface that precisely balances display and editability.  My experience consistently underscores the importance of a detailed understanding of these functionalities for creating a clean and efficient Django administration experience.

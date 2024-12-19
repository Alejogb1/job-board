---
title: "Is there a Django extension that does something similar to Ruby on Rails AttrJson?"
date: "2024-12-15"
id: "is-there-a-django-extension-that-does-something-similar-to-ruby-on-rails-attrjson"
---

alright, so you're looking for a django equivalent to rails' `attr_json`, right? i've been down that road before, and it’s definitely a pain point. let me break down what i’ve learned over the years, and give you some concrete options rather than just abstract ideas.

first off, if you’re unfamiliar with `attr_json` in rails, it basically lets you seamlessly treat json columns in your database like regular model attributes. behind the scenes it does the serialization and deserialization for you when accessing and updating those fields. it’s convenient because it allows structured data within the json and your code doesn't have to do the conversion, so you treat them as ordinary model attributes.

django doesn't have this built-in, unfortunately. when i first encountered this issue i had a project, probably back around 2016 or so, where we were dealing with user profiles that had a ton of flexible data like preferences and custom fields. we started with just plain text fields, then we started to use a json field, but quickly got tired of manually loading and dumping json, especially in forms. it was a hot mess to manage. i remember distinctly getting the json dumps messed up when we had nested objects and had to debug json string problems. we were using postgresql with the jsonb type at the time, so it wasn't an issue on that end, the problem was solely within the django model layer.

so, to answer your direct question: no, there isn't a single official django extension that exactly mirrors rails' `attr_json` feature, not one that is ubiquitous and commonly used in the community at least.

however, there are definitely ways to achieve similar functionality. you aren't completely out of luck. here are a few approaches i've used that mimic the experience, and they generally work well depending on your specific needs:

**1. using a custom property decorator:**

this is probably the most straightforward and explicit approach. it involves writing custom property getters and setters on your django model that handle the json serialization for you.

here is a simple example:

```python
import json
from django.db import models

class UserProfile(models.Model):
    data = models.JSONField(default=dict, blank=True)

    @property
    def settings(self):
        return self.data.get('settings', {})

    @settings.setter
    def settings(self, value):
        self.data['settings'] = value
        self.save()

    @property
    def address(self):
        return self.data.get('address',{})

    @address.setter
    def address(self, value):
        self.data['address'] = value
        self.save()
```

in this example, `UserProfile` has a `JSONField` named `data`. we then use `@property` and `@property.setter` to create a `settings` and `address` attribute that automatically reads and sets data from and to the `data` field within the json, using a dictionary access to the json value. the setter also saves the object. this way, instead of dealing with json directly, you can just do `user.settings['theme'] = 'dark'` or `user.address['street'] = '123 main'` which feels pretty close to what `attr_json` provides.

this method is great because it is flexible and explicit, you control the details of each virtual json attribute. the downside of this approach is that it needs to be replicated for each desired field, so it can become cumbersome when you have many virtual attributes and have to repeat similar code, it can quickly become a maintainability issue.

**2. custom model field using model.register_lookup:**

another approach, and one that i prefer when i need to do more sophisticated work, involves creating a custom model field that handles the serialization and deserialization directly within the field itself.

here is a more complex approach using a custom model field:

```python
import json
from django.db import models
from django.db.models import Field
from django.core import exceptions

class JsonAttributeField(Field):
    def __init__(self, attribute_name, *args, **kwargs):
        self.attribute_name = attribute_name
        super().__init__(*args, **kwargs)

    def deconstruct(self):
        name, path, args, kwargs = super().deconstruct()
        kwargs['attribute_name'] = self.attribute_name
        return name, path, args, kwargs


    def from_db_value(self, value, expression, connection):
        if value is None:
            return None
        try:
            return value.get(self.attribute_name)
        except AttributeError:
            return value #return the whole json if not a dictionary
        except Exception as e:
             raise exceptions.ValidationError("Error when retrieving value: "+ str(e)) from e

    def to_python(self, value):
       return value

    def get_prep_value(self, value):
        return value

    def pre_save(self, model_instance, add):
       json_field = getattr(model_instance, self.attname)
       json_data = getattr(model_instance,self.model_field_name)
       if json_data is None:
          json_data = {}
       if json_field is not None:
           json_data[self.attribute_name] = json_field
       setattr(model_instance, self.model_field_name, json_data)
       return getattr(model_instance, self.attname)


class JsonAttributeDescriptor:
    def __init__(self, field):
        self.field = field

    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        return self.field.value_from_object(instance)

    def __set__(self, instance, value):
        self.field.set_value(instance, value)

class JsonFieldWithAttributes(models.JSONField):
    def __init__(self, *args, attributes=None, model_field_name='data', **kwargs):
         self.model_field_name = model_field_name
         self.attributes = attributes or []
         super().__init__(*args, **kwargs)


    def contribute_to_class(self, cls, name, **kwargs):
        super().contribute_to_class(cls, name, **kwargs)
        for attribute in self.attributes:
             json_attribute_field = JsonAttributeField(attribute_name=attribute)
             json_attribute_field.model_field_name = name
             json_attribute_field.contribute_to_class(cls, attribute, private_only=True)
             setattr(cls, attribute, JsonAttributeDescriptor(json_attribute_field))


class MyModel(models.Model):
   data = JsonFieldWithAttributes(default=dict, blank=True, attributes=['favorite_color','weight', 'notifications'])
```

here, `JsonAttributeField` handles the extraction and conversion from the json, while the  `JsonFieldWithAttributes` lets you specify the attributes you want to have accessible. this is a more general solution than the first one, that handles the boilerplate for you.

this approach allows for much more flexibility, especially when you are dealing with custom serialization logic, however it adds a higher level of complexity, specially if you are not very familiar with custom fields and descriptors in django. It is more code to maintain, but worth it in my experience. i had to debug custom fields like this on one project and was a pain at first, but it gives you much more control over the data.

**3.  leveraging django-jsonfield and some metaprogramming:**

if you don't want to write a custom field from scratch, you can use a third-party package like `django-jsonfield` combined with some metaprogramming to dynamically create the property accessors.
for example:

```python
from django.db import models
from jsonfield import JSONField

class UserProfile(models.Model):
    data = JSONField(default=dict)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_json_fields()


    def _init_json_fields(self):
      json_attrs = ['settings', 'preferences', 'address'] # these are json attributes within the field named data
      for attr in json_attrs:
          def getter(obj, field_name=attr):
              return obj.data.get(field_name, {})
          def setter(obj, value, field_name=attr):
              obj.data[field_name] = value
              obj.save()
          setattr(self.__class__, attr, property(getter, setter))

```

here `django-jsonfield` provides a reliable `JSONField`, and then we are adding dynamic getters and setters. this is probably the easiest path to make many fields accessible at once. it's a bit more magic, though, so it is also harder to understand if you are not very experienced with class initializers and metaprogramming in python.

**which approach to use?**

*   if you have just a few fields and need it simple and explicit, the custom property approach (example 1) is a good start.
*   if you want a more reusable approach or dealing with a more complex situation, the custom field approach (example 2) is more suitable. it’s a little bit more involved to implement, but the payoff is greater long-term, especially when you have more than a handful of attributes.
*   if you need to quickly expose many fields and don’t mind some “magic”, the metaprogramming approach (example 3) using `django-jsonfield` might work for you. i'd suggest starting with this one only if you know what you are doing.

**resources:**

for a deeper understanding of how django models and fields work, the official django documentation is a must read, especially the section on model fields. another fantastic resource is "two scoops of django" by audrey greenfeld and daniel roy greenfeld. it has very good practical explanations of many django features, including model design patterns. i also found "django unleashed" by andrew pinkham and rob ludwick very useful, particularly on complex projects.

remember, the jsonb or json fields in postgresql work quite well on the database end, the problem is more in how django models expose them. that is why these approaches focus on the django part.

i hope this helps clarify things. don't hesitate to ask if you have more specific questions. and one last piece of advice... when debugging json issues, don't forget to use a good json viewer extension on your browser or code editor, sometimes just seeing the structure properly can make a world of difference.

ah, and if you need to serialize json faster, well, that is an entirely different problem for another day. sometimes i feel like databases are like a bad date, they keep dumping everything on you, haha. good luck with your project, and let me know if you need more assistance!

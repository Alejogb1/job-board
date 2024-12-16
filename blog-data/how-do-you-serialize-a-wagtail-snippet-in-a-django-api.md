---
title: "How do you serialize a Wagtail snippet in a Django API?"
date: "2024-12-16"
id: "how-do-you-serialize-a-wagtail-snippet-in-a-django-api"
---

Let's dive into this. Serializing wagtail snippets for consumption in a django api can feel a bit like navigating a less-traveled path initially, but it's certainly achievable with a good understanding of both django rest framework (drf) and how wagtail structures its data. I’ve faced this exact scenario a few times, notably during a project where we were decoupling a wagtail cms frontend from a react application. The challenge, as it usually is, lay in presenting those snippets in a format that’s easily digestible by a separate application.

The core problem revolves around the fact that wagtail snippets are essentially model instances. When creating an api, we need to convert these instances into a serializable format – typically json. Django rest framework excels at this, but it needs a little guidance, particularly given that wagtail's snippet models often include fields that drf doesn't automatically know how to handle, like `RichTextField` or `ForeignKey` relationships to other snippets or pages. We need to create custom serializers to manage this process.

Essentially, the fundamental idea is to treat the wagtail snippet like any other django model when creating a serializer. The difference, and the trick to making it work smoothly, is carefully defining the fields to be serialized and ensuring any relationships are properly represented in the output. We should specifically avoid returning serialized objects based on internal ids that are prone to changes during database migrations. Instead we should return the natural key of the wagtail snippets, a feature that has existed since at least wagtail 2.0.

Consider a scenario where I had a ‘testimonial’ snippet model. It had a `CharField` for the author's name, a `RichTextField` for the testimonial text, and a `ForeignKey` to an 'organization' snippet. The straightforward approach of using a standard drf `ModelSerializer` wouldn’t quite work out of the box, primarily because of the `RichTextField` and the ForeignKey relationship. We’d need to explicitly handle these.

Here’s a first-cut code example:

```python
# serializers.py
from rest_framework import serializers
from wagtail.snippets.models import register_snippet
from wagtail.core.fields import RichTextField
from django.db import models

# Assuming you have a 'organization' snippet registered
# as an example
class Organization(models.Model):
    name = models.CharField(max_length=255)
    
    def __str__(self):
        return self.name

    class Meta:
        verbose_name = "Organization"
        verbose_name_plural = "Organizations"

register_snippet(Organization)

class Testimonial(models.Model):
    author = models.CharField(max_length=255)
    text = RichTextField()
    organization = models.ForeignKey(
        Organization,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name='testimonials'
    )
    
    def __str__(self):
        return self.author

    class Meta:
        verbose_name = "Testimonial"
        verbose_name_plural = "Testimonials"

register_snippet(Testimonial)

class OrganizationSerializer(serializers.ModelSerializer):
    class Meta:
        model = Organization
        fields = ['name']

class TestimonialSerializer(serializers.ModelSerializer):
    text = serializers.CharField(source='text')
    organization = OrganizationSerializer(read_only=True)

    class Meta:
        model = Testimonial
        fields = ['author', 'text', 'organization']

```
In the above example, `OrganizationSerializer` serializes a linked organization snippet, while the `TestimonialSerializer` handles the `RichTextField` by setting it up as a `CharField` with a `source` field set to 'text'. We include a nested serializer that will handle the output of the `ForeignKey` relationship. `read_only=True` is set to prevent API users from trying to create or update nested objects.

While this first approach gets us started, consider a more complex scenario. Suppose our testimonial snippet now includes images, perhaps via a `wagtail.images.blocks.ImageChooserBlock` within the `RichTextField`. We’d need a slightly more nuanced approach to deal with these embedded objects.

```python
# serializers.py (expanded)

from rest_framework import serializers
from wagtail.snippets.models import register_snippet
from wagtail.core.fields import RichTextField
from django.db import models
from wagtail.images.models import Image
from wagtail.core.rich_text import get_text
import re


# Same Organization model as before, omitted for brevity

class Testimonial(models.Model):
    author = models.CharField(max_length=255)
    text = RichTextField()
    organization = models.ForeignKey(
        Organization,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name='testimonials'
    )
    
    def __str__(self):
        return self.author

    class Meta:
        verbose_name = "Testimonial"
        verbose_name_plural = "Testimonials"

register_snippet(Testimonial)

class OrganizationSerializer(serializers.ModelSerializer):
    class Meta:
        model = Organization
        fields = ['name']


class ImageSerializer(serializers.ModelSerializer):
    class Meta:
        model = Image
        fields = ['id', 'title', 'file'] # or any relevant fields


def _serialize_rich_text(rich_text):
    # This is a simple example. For a complete implementation, consider using a package like `wagtail-serializer`.
    if not rich_text:
        return ""
    text_content = get_text(rich_text)
    image_ids = [int(match) for match in re.findall(r'image="(\d+)"', rich_text)]
    images = Image.objects.filter(id__in=image_ids)
    serialized_images = ImageSerializer(images, many=True).data if images else []

    return {"text": text_content, "images": serialized_images}


class TestimonialSerializer(serializers.ModelSerializer):
    text = serializers.SerializerMethodField()
    organization = OrganizationSerializer(read_only=True)

    def get_text(self, obj):
        return _serialize_rich_text(obj.text)


    class Meta:
        model = Testimonial
        fields = ['author', 'text', 'organization']

```

Here, I’ve added an `ImageSerializer` to handle image fields and a `_serialize_rich_text` function that extracts the text content as well as finds image ids embedded in the `RichTextField`. The `TestimonialSerializer` now uses a `SerializerMethodField` to fetch the formatted text content including images from the function. Note, this simple example, while effective, may not cover every possible scenario and may require further extensions.

Finally, for a more robust approach, we can explore using django's `natural key` functionality to identify objects instead of their internal database ids. Using natural keys is very important to prevent the API from breaking when database migrations or re-creation occurs. We can modify our previous serializers to make use of this feature:

```python
# serializers.py (updated)

from rest_framework import serializers
from wagtail.snippets.models import register_snippet
from wagtail.core.fields import RichTextField
from django.db import models
from wagtail.images.models import Image
from wagtail.core.rich_text import get_text
import re

# Same Organization model as before, omitted for brevity

class Testimonial(models.Model):
    author = models.CharField(max_length=255)
    text = RichTextField()
    organization = models.ForeignKey(
        Organization,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name='testimonials'
    )
    
    def __str__(self):
        return self.author

    def natural_key(self):
        return (self.author, ) # author is a safe bet as natural key since the user should choose a non-repeating value
    
    class Meta:
        verbose_name = "Testimonial"
        verbose_name_plural = "Testimonials"

register_snippet(Testimonial)
    
class Organization(models.Model):
    name = models.CharField(max_length=255)
    
    def __str__(self):
        return self.name
    
    def natural_key(self):
        return (self.name,)

    class Meta:
        verbose_name = "Organization"
        verbose_name_plural = "Organizations"

register_snippet(Organization)

class OrganizationSerializer(serializers.ModelSerializer):
    class Meta:
        model = Organization
        fields = ['name'] # the natural key is outputted by default

class ImageSerializer(serializers.ModelSerializer):
    class Meta:
        model = Image
        fields = ['id', 'title', 'file'] # or any relevant fields


def _serialize_rich_text(rich_text):
    # This is a simple example. For a complete implementation, consider using a package like `wagtail-serializer`.
    if not rich_text:
        return ""
    text_content = get_text(rich_text)
    image_ids = [int(match) for match in re.findall(r'image="(\d+)"', rich_text)]
    images = Image.objects.filter(id__in=image_ids)
    serialized_images = ImageSerializer(images, many=True).data if images else []

    return {"text": text_content, "images": serialized_images}

class TestimonialSerializer(serializers.ModelSerializer):
    text = serializers.SerializerMethodField()
    organization = OrganizationSerializer(read_only=True)

    def get_text(self, obj):
        return _serialize_rich_text(obj.text)

    class Meta:
       model = Testimonial
       fields = ['author', 'text', 'organization'] # author is the natural key

```

Here we implement the `natural_key` method on both models, which then enables django to use the natural key instead of internal ids in the serialized output. Notice that the `OrganizationSerializer` does not explicitly define a field for a natural key, since this is the default behavior in django.

For those looking to delve deeper, I'd recommend exploring the official django rest framework documentation, specifically sections on serializers and nested relationships. The wagtail documentation concerning snippets and rich text fields is also crucial. Furthermore, reviewing the code of well-maintained, open-source wagtail plugins like `wagtail-serializer` can offer valuable insights. In addition, I suggest reading the official django documentation on model methods, with special attention to the `natural_key` method. Understanding these resources will equip you to create robust, adaptable apis that handle wagtail snippets effectively. Remember, the key is breaking down the challenge into smaller, manageable pieces and building custom solutions that fit your needs.

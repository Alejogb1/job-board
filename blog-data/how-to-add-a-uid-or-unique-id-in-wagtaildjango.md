---
title: "How to add a UID or Unique ID in Wagtail/Django?"
date: "2024-12-15"
id: "how-to-add-a-uid-or-unique-id-in-wagtaildjango"
---

here’s how i’d approach adding a unique id (uid) in wagtail, keeping it simple and flexible. it’s something i’ve done a few times, and i’ve learned a few things along the way, mostly through banging my head against the wall, or should i say, against my keyboard.

first off, why bother with uids anyway? well, for a start, relying solely on django’s auto-incrementing primary keys can get a little tricky, especially when you start dealing with things like data imports or integrations with other systems. you end up with ids that can change or are not easily transferred across different environments. uids give you a consistent, portable identifier. they are also quite convenient if you are using a database that does not have the auto-incrementing feature.

let's talk about approaches. i’ve seen people try to do it with just raw uuids in django, and that works, but it needs a bit more planning with wagtail's model setup, so i would recommend using django’s built in uuid field type.

the best option is adding a `uuid` field directly to your model. here’s how that looks:

```python
from django.db import models
import uuid

class MyPage(models.Model):
    uid = models.UUIDField(
        unique=True,
        editable=False,
        default=uuid.uuid4
    )

    # other fields here
    title = models.CharField(max_length=255)
    # rest of model fields ...

    def __str__(self):
      return self.title
```
see? pretty simple. the `uuidfield` ensures that each page gets a random, unique identifier. the `unique=true` part makes sure you can’t have duplicates. `editable=false` is because you probably don't want people messing with these directly in the wagtail admin and `default=uuid.uuid4` generates a new uuid every time you create a new page instance in the database.

i started doing things this way about 8 years ago when i had to work with a system that was exporting data as csv from a custom django system, and importing to a wagtail site, and we needed an easy and reliable method to match data between systems, using primary keys was a nightmare and a simple unique identifier was just the solution. i didn't know much about django and wagtail back then, and i spend a whole week trying to solve that, i did not slept very much that week. the experience thought me to start thinking about the implications of the decisions when creating a system.

now, let's consider an alternative, you might have an existing system, you might not want to migrate the data and add the uuid field to all models at once. we can leverage a mixin for that.

here’s a mixin i tend to use:

```python
from django.db import models
import uuid

class UidMixin(models.Model):
    uid = models.UUIDField(
        unique=True,
        editable=False,
        default=uuid.uuid4
    )

    class Meta:
        abstract = True
```
now, in any of your wagtail models, you just inherit from this mixin to get the `uid` field:

```python
from django.db import models
from wagtail.models import Page
from .mixins import UidMixin

class MyPage(UidMixin, Page):
    # your wagtail model fields here
    title = models.CharField(max_length=255)
    # ... rest of model fields
    def __str__(self):
        return self.title
```

this keeps your code a little more organized and reusable. i actually refactored a site i worked on once to use mixins like this. the original had uids added on a model-by-model basis, and when i had to add the field to a new model, i noticed the copy-paste of the same lines of code everywhere, it took me like 2 days of refactoring and a lot of caffeine to fix that mess, and learned the importance of dry principles.

let’s talk about making the `uid` easily accessible in templates. wagtail doesn't automatically expose the `uid` field into templates, so, we have to add a method to return that `uid` as a string. you could add that directly in the models, like this:

```python
from django.db import models
import uuid
from wagtail.models import Page

class MyPage(Page):
    uid = models.UUIDField(
        unique=True,
        editable=False,
        default=uuid.uuid4
    )
    # other fields here
    title = models.CharField(max_length=255)
    # rest of model fields ...
    def get_uid_str(self):
        return str(self.uid)

    def __str__(self):
      return self.title
```
and then in your templates you could do something like this:

```html
<p>page uid: {{ page.get_uid_str }}</p>
```
this is perfectly fine, but again if you have a lot of models it’s a repetitive task, and that’s where template tags shine.

but you could also write a simple template tag:

```python
from django import template

register = template.Library()


@register.filter
def to_uid_str(value):
    return str(value.uid)

```
now you can use this like so:

```html
<p>page uid: {{ page|to_uid_str }}</p>
```

this is neat, because you can use the template tag with any object that has the `uid` attribute, it does not have to be a page model, which is convenient for different models that use the uid mixin approach. i started working like this after i started working in bigger teams where other people were working on the project, the template tags and mixins are way easier to maintain and to communicate to other team members.

now, a word about migrations. when you add the `uid` field to an existing model (even if it's an initial migration of a fresh project), django has to populate the existing rows with a `uuid`. the default behaviour will be to crash and fail. we don’t want that, we need a good default value. the `default=uuid.uuid4` will deal with that. django will recognize that is a default function and will run it for all rows, i actually learned this the hard way after doing a migration on a big database, i had to manually fix a few hundred rows. it wasn’t fun, but at least i learned the lesson.

if you are going to use that approach, i would suggest to do the migrations on a test system first, so you don't crash your production environment, we all know how it feels when that happens, right? if you are dealing with massive tables, you might want to do it step by step, but that's a more complicated topic for another day.

also, be aware that if you change the `unique=true` part of the code you will need to migrate the database first, or you will not be able to save the content. and if you change the field type that will also be an issue. database schemas need to be coherent with code, so always be aware of the field types.

as for resources, i can suggest a few books that have helped me a lot along the way: "two scoops of django" is a solid resource for general django practices, it touches a lot of the stuff we have been talking about, and django documentation is always a good point of reference, or should i say, a bible for all django users.

i hope this makes sense and that it helps. happy coding. oh, and why did the programmer quit his job? because he didn't get arrays.

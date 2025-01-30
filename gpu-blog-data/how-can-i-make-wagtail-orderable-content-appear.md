---
title: "How can I make Wagtail orderable content appear on my homepage?"
date: "2025-01-30"
id: "how-can-i-make-wagtail-orderable-content-appear"
---
The core challenge in displaying Wagtail orderable content on the homepage lies in effectively retrieving and rendering the ordered items within a Wagtail template.  Standard Wagtail page retrieval methods don't inherently account for custom ordering defined within a model.  This requires a custom solution combining model manipulation, template logic, and potentially a custom Wagtail panel for more robust management.  I've encountered this issue numerous times building complex CMS-driven websites, and the optimal solution depends on the scale and complexity of your project.


**1. Clear Explanation:**

Wagtail's strength is its flexibility, but this also means that straightforward solutions aren't always readily apparent. To display orderable content on your homepage, you need to:

a) **Define Ordering within the Model:**  Your content model must include a field that dictates the order.  The simplest approach is using a `models.IntegerField` or `models.PositiveIntegerField`.  This integer will represent the order, with lower values appearing first.  While Wagtail doesn't provide an inherent ordering field, you can manually manage this field, using custom admin panels (detailed in example 3) or through the database directly (not recommended for production).

b) **Retrieve Ordered Content in the View:** Your view (or page controller) needs to retrieve the content from the database, ordering it according to the order field you defined. Django's ORM provides the `order_by()` method for this.

c) **Render Ordered Content in the Template:** The template must then iterate through this ordered queryset and render each item appropriately.


**2. Code Examples with Commentary:**

**Example 1: Basic Implementation using a PositiveIntegerField**

```python
# models.py
from wagtail.admin.edit_handlers import FieldPanel
from wagtail.core.fields import RichTextField
from wagtail.core.models import Page
from django.db import models

class HomePage(Page):
    body = RichTextField(blank=True)

    content_panels = Page.content_panels + [
        FieldPanel('body', classname="full"),
    ]

class OrderableItem(models.Model):
    page = models.ForeignKey(HomePage, on_delete=models.CASCADE, related_name='orderable_items')
    title = models.CharField(max_length=255)
    description = models.TextField(blank=True)
    order = models.PositiveIntegerField(default=0, editable=False) #Note: editable=False prevents accidental manual modification.

    panels = [
        FieldPanel('title'),
        FieldPanel('description'),
    ]

    class Meta:
        ordering = ['order'] # Database-level ordering for efficiency.

    def __str__(self):
        return self.title

# views.py (or in a custom page controller)
from django.shortcuts import render
from .models import HomePage

def home_page(request, *args, **kwargs):
    page = HomePage.objects.first() # Assumes only one homepage.
    if page:
        context = {'page': page}
        return render(request, 'home.html', context)
    else:
        return HttpResponseNotFound("Homepage not found.")

# home.html
{% for item in page.orderable_items.all %}
    <h2>{{ item.title }}</h2>
    <p>{{ item.description }}</p>
{% endfor %}

```

This example utilizes a simple `PositiveIntegerField` for ordering. The `Meta` class within the `OrderableItem` model ensures database-level ordering for efficiency. The homepage template then directly iterates over the ordered items.  This is suitable for relatively small datasets.


**Example 2:  Handling Large Datasets with Pagination**

For larger datasets, pagination is crucial for performance.  We modify the view to incorporate pagination.

```python
# views.py
from django.shortcuts import render
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from .models import HomePage

def home_page(request, *args, **kwargs):
    page = HomePage.objects.first()
    if page:
        items = page.orderable_items.all()
        paginator = Paginator(items, 10) # 10 items per page. Adjust as needed.
        page_number = request.GET.get('page')
        try:
            items_page = paginator.page(page_number)
        except PageNotAnInteger:
            items_page = paginator.page(1)
        except EmptyPage:
            items_page = paginator.page(paginator.num_pages)
        context = {'page': page, 'items': items_page}
        return render(request, 'home.html', context)
    else:
        return HttpResponseNotFound("Homepage not found.")


# home.html
{% for item in items %}
    <h2>{{ item.title }}</h2>
    <p>{{ item.description }}</p>
{% endfor %}

{% include "pagination.html" %} # Separate template for pagination controls.

# pagination.html
{% if items.has_other_pages %}
    <ul class="pagination">
        {% if items.has_previous %}
            <li><a href="?page={{ items.previous_page_number }}">&lt; Previous</a></li>
        {% endif %}
        {% for num in items.paginator.page_range %}
            {% if num == items.number %}
                <li class="active"><a href="#">{{ num }}</a></li>
            {% else %}
                <li><a href="?page={{ num }}">{{ num }}</a></li>
            {% endif %}
        {% endfor %}
        {% if items.has_next %}
            <li><a href="?page={{ items.next_page_number }}">Next &gt;</a></li>
        {% endif %}
    </ul>
{% endif %}

```
This example integrates pagination using Django's `Paginator` class. This significantly improves the user experience and prevents performance bottlenecks when dealing with numerous ordered items.


**Example 3:  Implementing a Custom Wagtail Panel for Order Management**

For more sophisticated order management, a custom Wagtail panel is beneficial. This allows for drag-and-drop ordering within the Wagtail admin interface.

```python
# models.py
from wagtail.admin.edit_handlers import FieldPanel, MultiFieldPanel, Orderable
from wagtail.core.fields import RichTextField
from wagtail.core.models import Page
from django.db import models
from wagtail.snippets.edit_handlers import SnippetChooserPanel

class HomePage(Page):
    body = RichTextField(blank=True)

    content_panels = Page.content_panels + [
        FieldPanel('body', classname="full"),
        MultiFieldPanel([
            Orderable(
                'orderable_items'
            ),
        ], heading="Orderable Items"),
    ]

class OrderableItem(models.Model):
    # ... (as before) ...

    class Meta:
        ordering = ['order']

    panels = [
        FieldPanel('title'),
        FieldPanel('description'),
    ]

```

This leverages Wagtail's `Orderable` functionality within a `MultiFieldPanel` to provide a user-friendly interface for reordering items within the Wagtail admin.  This eliminates the need for manual order field updates and improves workflow significantly.  Remember to adjust the `ordering` meta option in the model based on your implementation choices.


**3. Resource Recommendations:**

* The official Wagtail documentation. Thoroughly review sections covering models, templates, and the admin interface.
* Django's ORM documentation.  Understanding Django's queryset manipulation is essential for effective data retrieval and ordering.
* Wagtail's advanced tutorial section.  Many concepts related to custom panels and model extensions are discussed here.  Pay special attention to the examples on creating custom admin panels and incorporating third-party libraries to extend Wagtail's functionality.


By carefully considering the size of your dataset and the level of administrative control required, you can choose the most appropriate solution from these examples.  Remember to always test thoroughly and utilize Django's debugging tools to identify any potential issues in your implementation.  In my past experiences, combining the elegance of Wagtail's admin with the robustness of Django's ORM provided the most efficient and scalable approach to managing and displaying ordered content.

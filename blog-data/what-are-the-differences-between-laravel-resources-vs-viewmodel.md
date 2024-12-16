---
title: "What are the differences between Laravel Resources VS ViewModel?"
date: "2024-12-16"
id: "what-are-the-differences-between-laravel-resources-vs-viewmodel"
---

Alright, let’s unpack the nuances between Laravel resources and view models. This is a topic I’ve had to navigate extensively, especially back in my days managing a rather complex e-commerce platform. We hit issues scaling the API and rendering views efficiently. That's where a solid understanding of these architectural patterns became crucial. It's less about one being universally "better" and more about choosing the appropriate tool for the job, and honestly, often about the subtle architectural decisions that impact performance.

Let's start with Laravel resources. In their essence, they are primarily about transforming your eloquent models into a specific, often json-based, structure, typically for your API endpoints. Think of it as a well-defined contract: you hand a model instance, and it returns a data representation suitable for transmission. This is hugely beneficial when you want to control exactly what data your API is exposing and how it's formatted. Laravel resources handle serialization very effectively, offering options for including relations, conditional formatting, and custom logic within a resource class.

For example, consider a `Product` model with potentially sensitive or unnecessary information that should not be exposed via an API. You might have fields like the internal cost or inventory details that are crucial for internal operations but should never be part of the JSON response for a public-facing api. Using a resource allows you to define precisely what attributes to include:

```php
<?php

namespace App\Http\Resources;

use Illuminate\Http\Request;
use Illuminate\Http\Resources\Json\JsonResource;

class ProductResource extends JsonResource
{
    public function toArray(Request $request): array
    {
        return [
            'id' => $this->id,
            'name' => $this->name,
            'description' => $this->description,
            'price' => $this->price,
            // Exclude internal data such as cost, inventory and others
        ];
    }
}
```

In this simplified `ProductResource`, we’re explicitly defining which fields of the `Product` model get serialized into the json response. This provides security and performance gains. Furthermore, resources handle collections of models as well. We can use `ProductResource::collection($products)` to transform a list of products seamlessly. This approach minimizes database query load and ensures data integrity by exposing only required information.

Now, let's switch gears to view models. Their purpose is radically different. View models are primarily about preparing data for a specific view, be it a blade template or, in modern context, a frontend framework component. The goal is to move data manipulation logic out of the template itself and into a class responsible for view-specific formatting and presentation. It's about encapsulating the data, transforming it into a format that is convenient for rendering, and reducing boilerplate in the view layer.

Imagine you have a complex view showing aggregated sales data. Rather than performing all that calculation and formatting within the blade file, you'd want a ViewModel to take on the responsibility. Here's a conceptual example:

```php
<?php

namespace App\ViewModels;

use App\Models\Order;
use Carbon\Carbon;

class SalesDashboardViewModel
{
    public function __construct(public Carbon $startDate, public Carbon $endDate)
    {
    }

    public function getSalesData() : array
    {
       // Fetch orders within date range
       $orders = Order::where('created_at','>=', $this->startDate)->where('created_at','<=', $this->endDate)->get();

       // Calculate total sales
       $totalSales = $orders->sum('total');

       // Format sales data for display purposes
       return [
            'totalSales' => number_format($totalSales, 2),
            'numberOfOrders' => $orders->count()
            // add more aggregated data
        ];
    }
}
```

Here, the `SalesDashboardViewModel` receives the start and end date as parameters. It then calculates and formats the relevant data—total sales and number of orders—making that data available to the view via its `getSalesData()` method. The view itself remains simpler and easier to read. This approach significantly reduces complexity within our blade templates and moves responsibility for data transformation to dedicated classes, aligning with the Single Responsibility Principle. This makes our views clearer and easier to maintain.

Now, you might be thinking, "well, what if I need to transform data differently for different clients?" That’s a valid point and a common scenario, especially with APIs serving multiple apps. That is where the flexibility of view models can shine. Think of it this way: resources are good for serializing data into a format (usually json) and view models are about making the view have all the data it needs already formatted.

Let's say your admin dashboard requires very detailed data, including user-specific sales figures whereas your customer portal only needs aggregated data. Here’s how a view model allows you to have that level of control.

```php
<?php

namespace App\ViewModels;

use App\Models\Order;
use App\Models\User;

class AdminSalesDashboardViewModel
{
    public function __construct(public User $user)
    {
    }

    public function getUserSalesData()
    {
        $userSales = Order::where('user_id',$this->user->id)->get();

        $totalUserSales = $userSales->sum('total');

        return [
            'totalUserSales' => number_format($totalUserSales,2),
            'numberOfUserOrders' => $userSales->count(),
            'user' => [
                'name' => $this->user->name,
                'email' => $this->user->email
            ],
            // add more user related data
        ];
    }
}
```

In this example, `AdminSalesDashboardViewModel` retrieves sales data specifically related to the authenticated user which is accessible via its `getUserSalesData()` function. This enables more fine-grained control over the data presented on the admin dashboard, including specific user information that should not be present in a public-facing API. Resources excel at consistent data serialization, but view models offer targeted formatting to accommodate different usage contexts.

So, to summarize, resources are primarily about data serialization, particularly for API responses, providing a structured and controlled way to expose your model data. View models, on the other hand, are concerned with preparing data for specific views, encapsulating complex data transformations to keep our views clean and focused solely on presentation. Neither is a silver bullet, but a clear understanding of their strengths will guide you to making sound architectural decisions that will pay off in the long run, just as I experienced during my days scaling that e-commerce platform.

If you want to really dive deep into these patterns, I’d recommend looking at Martin Fowler's "Patterns of Enterprise Application Architecture" for a great overview of architectural patterns in general. For specifics on data transfer objects and similar concepts, "Domain-Driven Design" by Eric Evans provides an excellent foundation. Finally, you may want to examine any good text on API design practices and patterns which would cover many of the best practices employed within Laravel's resource handling capabilities. These sources, in my experience, offer the foundational knowledge needed to make these architectural choices effectively.

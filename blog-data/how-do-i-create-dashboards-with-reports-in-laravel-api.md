---
title: "How do I create dashboards with reports in Laravel API?"
date: "2024-12-16"
id: "how-do-i-create-dashboards-with-reports-in-laravel-api"
---

, let’s tackle this. Dashboards and reports, when served through a Laravel API, involve a few key elements: data aggregation, efficient data retrieval, and effective presentation. I’ve certainly built my share of these over the years, and the approach I've found most reliable combines careful model design, query optimization, and a structured API endpoint strategy. Let’s dive into how this works in practice.

Fundamentally, the challenge isn't just about pushing data from your database to a frontend client; it’s about doing so in a manner that’s performant and meaningful. I remember one project where we initially just pulled all the records and processed them in the frontend. Needless to say, things ground to a halt as data volume grew. We learned quickly that the heavy lifting needs to happen on the server side.

First, let's consider data aggregation. Direct queries that process raw data in a loop in the controller are a definite anti-pattern. Instead, leveraging the database’s aggregation capabilities is crucial. If you need to calculate, say, the total number of users created per month, you should construct that query to be processed directly by the database. Laravel’s eloquent provides some nice tools for this, making the query more concise.

Here's a snippet to illustrate:

```php
<?php

namespace App\Http\Controllers\Api;

use App\Models\User;
use Illuminate\Http\JsonResponse;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\DB;

class DashboardController extends Controller
{
    public function usersPerMonth(): JsonResponse
    {
        $monthlyUsers = User::select(DB::raw('YEAR(created_at) as year, MONTH(created_at) as month, COUNT(*) as count'))
                ->groupBy('year', 'month')
                ->orderBy('year', 'asc')
                ->orderBy('month', 'asc')
                ->get();

        return response()->json($monthlyUsers);
    }
}
```

In this example, the `usersPerMonth` method demonstrates how to use raw SQL in conjunction with Eloquent to perform a grouping query, directly retrieving the data in the required format. This approach avoids retrieving every single user record and performing the grouping in PHP which can significantly slow performance for large datasets. Notice that the query uses aggregate functions such as `COUNT()` and it groups by the year and month to generate a nice format for plotting time-series data. The result is a json response ready to be consumed by the frontend.

Now, let’s talk about complex reports. It is common to build reports that require data joined from multiple tables, or data filtered using multiple parameters. This is often handled using custom query scopes within your Eloquent models. These scopes can encapsulate common filtering or data joining logic. Instead of repeating this in your controllers, you can just use the named scopes for retrieving data. These named scopes offer a consistent and testable approach to retrieving data.

Here is another example:

```php
<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Builder;

class Order extends Model
{
    use HasFactory;

    protected $fillable = [
        'customer_id',
        'order_date',
        'total_amount',
        'status'
    ];

    public function customer()
    {
        return $this->belongsTo(Customer::class);
    }

    public function scopeFilterByDateRange(Builder $query, $startDate, $endDate)
    {
        return $query->whereBetween('order_date', [$startDate, $endDate]);
    }

    public function scopeFilterByStatus(Builder $query, $status)
    {
        return $query->where('status', $status);
    }
}
```

```php
<?php

namespace App\Http\Controllers\Api;

use App\Models\Order;
use Illuminate\Http\JsonResponse;
use Illuminate\Http\Request;

class ReportController extends Controller
{
   public function orders(Request $request): JsonResponse
   {
        $startDate = $request->input('start_date');
        $endDate = $request->input('end_date');
        $status = $request->input('status');

        $orders = Order::query();

       if ($startDate && $endDate){
           $orders->filterByDateRange($startDate, $endDate);
       }

       if ($status){
           $orders->filterByStatus($status);
       }

        return response()->json($orders->with('customer')->get());
   }
}
```

In this example, we've defined two scopes: `filterByDateRange` and `filterByStatus`. These are reusable and provide a cleaner syntax for querying the orders. In the controller, the `orders` method accepts request parameters to filter the orders. This approach ensures all filtering logic is encapsulated in the model, enhancing the maintainability of your codebase. The `with('customer')` demonstrates how you can avoid the n+1 problem using eager loading when retrieving relationships.

Regarding pagination, large reports are almost always better served in smaller, paginated chunks. This not only improves response times but also reduces the amount of data the client needs to process at once. Laravel’s built-in pagination functionality makes this quite straightforward.

Here is one final example demonstrating pagination:

```php
<?php

namespace App\Http\Controllers\Api;

use App\Models\Product;
use Illuminate\Http\JsonResponse;
use Illuminate\Http\Request;

class ProductController extends Controller
{
    public function products(Request $request): JsonResponse
    {
        $perPage = $request->input('per_page', 10); // Default to 10 items per page
        $products = Product::paginate($perPage);

        return response()->json($products);
    }
}
```

Here, the `products` method shows how easily we can implement pagination with the `paginate` method. The `per_page` parameter allows a user to specify the number of items to include in each response, or use a default of 10 if not provided. The response returned by `paginate()` includes not only the data, but also metadata such as the current page number, the total number of pages, and so on. This simplifies the handling of pagination in the frontend application.

Finally, when designing your endpoints, I suggest using a consistent structure for your responses. This not only makes consuming your API easier for frontend developers but also helps to ensure uniformity across the application. Consider utilizing JSON:API or similar specifications to help guide your API structure, specifically including error handling and pagination information consistently. Furthermore, caching mechanisms, such as Laravel’s built-in cache facade, are worth exploring to improve overall performance for frequently accessed reports, especially when results do not change frequently. For complex caching needs, you may want to use a redis or memcached server.

For further in-depth study, I highly recommend exploring "Eloquent Performance Patterns" from Spatie, which includes many insightful strategies for improving data retrieval. In addition, the official Laravel documentation is the best starting point for every concept we covered, from raw queries to named scopes, relationships, and pagination. For a better understanding of query optimization at a database level, "High Performance MySQL" by Baron Schwartz is an indispensable resource. These resources, along with practical experience, are what I've personally used for years to tackle such challenges.

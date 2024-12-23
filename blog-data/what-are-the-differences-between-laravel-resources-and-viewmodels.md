---
title: "What are the differences between Laravel Resources and ViewModels?"
date: "2024-12-23"
id: "what-are-the-differences-between-laravel-resources-and-viewmodels"
---

, let's dissect this. I've grappled with this particular distinction quite a bit in past projects, especially when scaling applications with increasingly complex data presentation needs. It's a crucial point for any Laravel developer looking to maintain clean and efficient code. Essentially, both Laravel resources and view models aim to transform data before it reaches the client, but they approach this problem from different angles and with distinct purposes. I'd say the core difference lies in their scope and responsibility within the application architecture.

Let's begin with Laravel resources. Think of resources as your *data representation layer*. They're primarily concerned with the *structure* and *format* of your api responses. When you're building an api, you’re very likely dealing with json or other standardized data formats. Laravel resources excel at defining how eloquent models (or any other data source for that matter) are transformed into a consistent and easily consumable form. I've used them extensively when creating api endpoints that need to adhere to specific json schemas, ensuring that only the necessary attributes are exposed and are presented in the desired format. For example, consider you have a `User` model with many columns, but you only need to expose `id`, `name`, and `email` in an api endpoint. Resources handle this elegantly. Moreover, they handle relationships, transformations on nested entities, and even conditional data inclusion smoothly. Think of a resource as a *serialiser*, taking data and producing an output that a consumer knows how to handle. This, in my experience, makes them indispensable for creating predictable and well-defined api interfaces. I had a project where we started with basic json output for api responses. As the api evolved, we had to include more and more conditional data. Resources allowed us to encapsulate these complexities within the resource definition rather than cluttering our controllers.

Now, let's shift to view models. These differ fundamentally in their purpose. A view model is primarily concerned with *preparing* data for *display* in a view. They represent data intended for a specific context within the application's ui. View models do not focus on standardizing or serialising an output, instead they focus on *enriching* data for rendering on the client. Whereas a resource is used to send data to the outside world, a view model’s purpose is to serve data within your application. Consider you have a user profile page, you might need to aggregate data from multiple tables, format dates, calculate derived values (like user activity statistics), or apply certain business logic. This doesn't involve changing the underlying data's structure so much as providing context and relevant information for a specific view. I’ve often used view models to clean up complex logic from blade templates, resulting in far more readable and maintainable views. These are not inherently coupled to a model but are constructed to provide the exact data set a particular view requires. View models often interact with multiple models and services, enriching raw data with display-specific logic. This helped enormously when we had complex reports that needed data combined from diverse systems, formatted for a human-readable output. I’d also point out that whilst Resources are generally a built in Laravel feature, ViewModels can be implemented as required.

To illustrate, here are some code snippets:

First, a typical Laravel resource example:

```php
<?php

namespace App\Http\Resources;

use Illuminate\Http\Request;
use Illuminate\Http\Resources\Json\JsonResource;

class UserResource extends JsonResource
{
    /**
     * Transform the resource into an array.
     *
     * @return array<string, mixed>
     */
    public function toArray(Request $request): array
    {
        return [
            'id' => $this->id,
            'name' => $this->name,
            'email' => $this->email,
            'is_active' => (bool) $this->is_active,
            'created_at' => $this->created_at->toIso8601String(), // formatted date
             'profile' => new UserProfileResource($this->whenLoaded('profile')), // nested resource
        ];
    }
}
```

This `UserResource` serializes the `User` model for an api response, controlling what data is included and formatting the `created_at` attribute. If the `profile` relation has been loaded (using eager loading), it is included as a nested resource. It is very clear what data is being passed to the client, in a standardized format.

Next, here's how a view model could look:

```php
<?php

namespace App\ViewModels;

use App\Models\User;
use Carbon\Carbon;

class UserProfileViewModel
{
    public User $user;

    public function __construct(User $user)
    {
        $this->user = $user;
    }

    public function getFullName(): string
    {
        return $this->user->first_name . ' ' . $this->user->last_name;
    }

    public function getLastLogin(): string
    {
       if($this->user->last_login_at) {
            return  Carbon::parse($this->user->last_login_at)->diffForHumans();
       }
       return 'Never Logged In';
    }


    public function getFormattedJoinDate(): string
    {
        return $this->user->created_at->format('F j, Y');
    }

     public function getProfilePictureUrl(): string
    {
        return $this->user->profile_picture_url ?: '/default-profile.jpg';
    }

    public function getUserStats() {
         //Some additional logic, potentially hitting other services or models to create a derived value
          return [
               'totalPosts' => 123,
               'totalComments' => 345,
           ];
    }
}

```

Here, the `UserProfileViewModel` is built around a `User` model, but it prepares data specific to the user profile view. It formats date output and calculates derived data. Note the absence of any standardisation or serialisation, this class exists to provide enhanced information for a specific view.

Finally, we can see how we might use both in conjunction, a view might return the data from a resource and a view model:

```php
<?php

namespace App\Http\Controllers;
use App\Models\User;
use App\Http\Resources\UserResource;
use App\ViewModels\UserProfileViewModel;
use Illuminate\Http\Request;

class UserController extends Controller
{

     public function show(Request $request, User $user)
    {
        if ($request->wantsJson()) {
            // api response - resource
            return new UserResource($user->load('profile'));
        }

        // view response - view model
        $viewModel = new UserProfileViewModel($user);
        return view('users.profile', ['user' => $viewModel]);
    }

}
```
In this example, a single controller method provides different responses depending on the request type: a json response using a Resource for api calls, and a view response, using a view model, for web responses. This shows the flexibility of using both in conjunction.

In summary, resources focus on the structured output of your models for api consumption. View models enrich data for specific view contexts and often combine data from multiple models and services. Choosing the right tool for the job is paramount for maintainable code. Using Resources where appropriate for api responses and ViewModels where appropriate for views, leads to a cleaner and more maintainable code base.

For further reading, I'd suggest looking at these references:

*   **"Eloquent JavaScript" by Marijn Haverbeke**: Although primarily about javascript, the ideas of data transformations and structuring data are foundational and are well explained. The thinking process is relevant irrespective of your specific language.
*   **"Patterns of Enterprise Application Architecture" by Martin Fowler**: This provides the foundational thinking around the ‘presentation’ patterns of software development and touches upon the reasoning for ViewModels, as well as many other application architecture principles.
*   The **Laravel documentation** itself is excellent. Pay close attention to the sections on “api resources” and “eloquent model events” as this will give you a thorough understanding of the resources and the data they are built from.

I found that deeply understanding their differences has significantly streamlined development on several projects and reduced code duplication. If you get these two concepts cemented in your mind it will significantly enhance your ability to create well defined and organised code in the Laravel ecosystem.

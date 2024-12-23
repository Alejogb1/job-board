---
title: "How to redirect /api/v1 to /api/v2 in Rails 6?"
date: "2024-12-23"
id: "how-to-redirect-apiv1-to-apiv2-in-rails-6"
---

Alright, let's talk about redirecting api endpoints, specifically moving from `/api/v1` to `/api/v2` in a Rails 6 application. This is a common scenario, especially when dealing with api versioning, and I’ve certainly tackled this kind of migration more than a few times over the years. It's a situation that can easily trip you up if not approached methodically.

The key here is to maintain backwards compatibility as much as possible while clearly signaling the shift to the newer version. Ideally, you want a smooth transition for your client applications, minimizing any disruption. So, let's break down the core strategies, and I'll provide some concrete code examples based on my experiences.

The first, and probably most common approach, is to handle redirects directly within your Rails routing configuration. This is very effective when the changes between versions are largely in the controller or model logic, and less in the pathing conventions of the actual API endpoint.

Here's a quick snippet showcasing how you might accomplish this in your `config/routes.rb` file:

```ruby
Rails.application.routes.draw do
  namespace :api, defaults: { format: 'json' } do
    namespace :v1 do
        # legacy v1 routes here, maybe with deprecated code
        resources :users, only: [:index, :show]
    end

    namespace :v2 do
      # all new v2 endpoints here
      resources :users, only: [:index, :show]
    end

    # Redirect v1 requests to v2. Explicitly specify a 301
    # (permanent redirect) to notify clients.  Note the
    # contraint ensures we are only redirecting endpoints
    # within the /api/v1 namespace and not anything else.
    get '/v1/*path', to: redirect('/api/v2/%{path}'), constraints: lambda { |req| req.path.start_with?('/api/v1/') } ,  status: 301
  end
end
```

In this example, any request made to `/api/v1/users` or `/api/v1/users/123` would be redirected to the equivalent endpoint under `/api/v2` with a 301 status code. Notice the constraint block. This is crucial, you don’t want to just redirect any /v1/ anything that gets passed into your application, that will cause a lot of unexpected behavior and possible security issues.

While routing redirects are useful for simple endpoint shifts, they don’t allow for more sophisticated logic or parameter transformations. Sometimes, the structure of the new api version might need slight modifications which are best done in a controller, for example we might want to transform camel case to snake case for query params. That’s where controller-based redirects become valuable.

Here’s an approach where we can implement the redirect directly within a controller, which i’ve had to use in cases where you may not want to send an explicit redirect to the consumer:

```ruby
# app/controllers/api/v1/users_controller.rb
class Api::V1::UsersController < ApplicationController
    before_action :redirect_to_v2, only: [:index, :show]

    def index
      # this block won't be reached if redirected below
    end

    def show
      # this block won't be reached if redirected below
    end

    private

    def redirect_to_v2
        # transform any camel case params to snake case
        v2_params = params.deep_transform_keys { |key| key.to_s.underscore.to_sym }

        redirect_to(controller: '/api/v2/users', action: action_name, params: v2_params, status: :moved_permanently)
    end
end
```

In this scenario, any request coming to the `Api::V1::UsersController` for the `index` or `show` actions triggers the `redirect_to_v2` method. This redirect method transforms camel case keys to snake case, then forwards the request with all parameters to the equivalent action in the `Api::V2::UsersController`. This method is particularly useful when you have params that may have changed their naming conventions between api versions.

Sometimes, you might need even more fine-grained control, especially when some `/v1` endpoints need to be maintained while others are redirected. In those cases, I’ve found a combination of routes and controller logic provides the needed flexibility, particularly with custom logic or header manipulation.

Here’s an example of using a specific header on the request to perform a version check and perform different logic based on that header:

```ruby
# app/controllers/api/v1/users_controller.rb
class Api::V1::UsersController < ApplicationController
    before_action :check_api_version, only: [:index, :show]


    def index
        # Handle logic based on the version, for v1 specific logic if needed
        if @api_version == 'v1'
            @users = User.all
            render json: @users, status: :ok
        else
            redirect_to(controller: '/api/v2/users', action: action_name, params: params.permit!.to_h.deep_transform_keys{ |key| key.to_s.underscore.to_sym }, status: :moved_permanently)
        end
    end

    def show
        if @api_version == 'v1'
             @user = User.find(params[:id])
             render json: @user, status: :ok
         else
             redirect_to(controller: '/api/v2/users', action: action_name, params: params.permit!.to_h.deep_transform_keys { |key| key.to_s.underscore.to_sym }, status: :moved_permanently)
         end
    end

    private

    def check_api_version
        @api_version = request.headers['X-API-Version'] || 'v1'
    end
end
```

In this example we check for an `X-API-Version` header to determine how to route the request. If the header indicates the API version to be `v1` then the specific `v1` logic executes. Otherwise, the client is redirected to `v2`.

A few points to keep in mind when implementing these solutions:

1.  **Choose the Right HTTP Status Code:** When using redirect routes or controller redirects, be sure to pick the correct status code. A `301 Moved Permanently` is the right choice when clients should not attempt to use the old url again. `307 Temporary Redirect` might be suitable during testing or during a phased rollout if you want the client to use the old url for some time.
2.  **Handle Parameter Changes:** When moving between API versions, parameter names, data structures or data types may have changed. The second code example attempts to make sure that your old params are in snake case, this can be expanded to perform many different transformations of the params.
3.  **Test Thoroughly:** Ensure you test your redirects with a variety of clients that access your api. This includes checking responses to not just `GET`, but also to `POST`, `PUT`, `PATCH` and `DELETE` requests.
4.  **API Documentation:** Ensure your API documentation reflects the changes, and especially that `v1` is clearly marked as deprecated with clear instructions to the migration paths.
5.  **Phased Rollouts:** For larger migrations, consider doing a phased rollout. You can use feature flags or environment variables to control which traffic gets routed to `/v2` so you can closely monitor and revert quickly if needed.
6.  **Monitoring:** Monitor the performance and error logs during and after the migration. This will help you identify any issues clients might be facing.

For deepening your understanding of API versioning and related topics, I recommend diving into these resources:

*   **"Building Microservices" by Sam Newman:** This book offers a comprehensive guide to designing and building microservices, with insightful discussions on versioning strategies.
*   **"RESTful Web APIs" by Leonard Richardson and Mike Amundsen:** A classic resource for learning about REST principles and designing good api endpoints, which can be applied to api versioning and migration strategies.
*   **RFC 7231 (Hypertext Transfer Protocol (HTTP/1.1): Semantics and Content):** Understanding HTTP status codes is crucial, and this document is the authoritative source. Pay particular attention to status codes 301, 302, 303, 307 and 308.

Implementing a seamless redirect from `/api/v1` to `/api/v2` in Rails requires careful planning, detailed testing, and an understanding of the different redirect strategies available. By using the examples and best practices outlined above, you can effectively migrate your api to the new version while minimizing disruption to clients. Remember to document your changes, and communicate clearly with your api consumers. Good luck with your migration!

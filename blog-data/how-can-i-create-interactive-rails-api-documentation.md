---
title: "How can I create interactive Rails API documentation?"
date: "2024-12-23"
id: "how-can-i-create-interactive-rails-api-documentation"
---

, let’s tackle interactive api documentation in rails. It’s a question I've seen pop up countless times, and I've certainly felt the sting of poorly documented apis, both as a consumer and as a builder. In my experience, merely providing a static list of endpoints isn't enough; developers need a way to actually interact with the api, to test parameters, and to see responses in real-time. This accelerates adoption and drastically reduces onboarding headaches. I’ve built and maintained several rails apis over the years, and relying on things like hand-written documentation or generic swagger files always fell short when things got complex. We needed something better.

The key, as I see it, lies in combining the power of a robust documentation framework with rails' inherent api-building capabilities. When I say 'robust documentation framework,' I’m typically referring to tools that not only generate the documentation from code annotations but also provide interactive interfaces. And that's exactly what we’ll explore.

There are, in my opinion, a few effective approaches, each with trade-offs. I’ve leaned heavily on using `rswag`, `apipie`, and at times, a more customized solution leveraging `openapi` (formerly swagger) specs. Let me break these down, giving you a practical feel for how they actually work.

First up is `rswag`. This gem beautifully integrates with `rspec`, allowing you to define your api endpoints, parameters, and responses directly within your test suite. This promotes a 'documentation-as-code' approach. The beauty here is that documentation is inherently tied to your tests, reducing the likelihood of the docs becoming out of sync. It’s a very strong case for integration.

Here’s an example of a simple endpoint definition using rswag in an rspec test:

```ruby
require 'swagger_helper'

describe 'Products API', type: :request do
  path '/products' do
    get 'Retrieves a list of products' do
      tags ['Products']
      produces 'application/json'
      parameter name: :page, in: :query, type: :integer, description: 'Page number for pagination', required: false
      parameter name: :per_page, in: :query, type: :integer, description: 'Number of products per page', required: false
      response '200', 'Products Retrieved' do
          schema type: :array,
                  items: {
                  type: :object,
                    properties: {
                      id: { type: :integer },
                      name: { type: :string },
                      price: { type: :number, format: :float }
                    },
                   required: %w[id name price]
                  }
        run_test!
      end
    end
  end
end

```
This code, while part of a test, also generates swagger-compatible json documentation when you use the provided generator tasks. Rswag then provides an interface using its ui gem that allows for executing these api calls and reviewing the responses, complete with parameter inputs. It's genuinely one of the most efficient approaches I've used to link testing and documentation.

Next, let's explore `apipie`. Apipie leans more on explicit annotations within your rails controller methods. Instead of relying on tests, you add documentation blocks directly above your actions. Some developers prefer this as it keeps the documentation more closely coupled to the implementation. The downside, however, is the potential to diverge from actual functionality if the docs aren’t carefully maintained. This can often happen when development speeds up.

Here's how you'd define a similar endpoint using apipie:

```ruby
class ProductsController < ApplicationController
  api :GET, '/products', 'Retrieve a list of products'
  param :page, Integer, 'Page number for pagination', required: false
  param :per_page, Integer, 'Number of products per page', required: false
  formats ['json']

  def index
    @products = Product.page(params[:page]).per(params[:per_page])
    render json: @products
  end
end
```
This approach requires installing the apipie gem, running the generator to create documentation, and navigating to `/apipie` in your rails application. The interface is similar to what we see with rswag—allowing you to input parameters and trigger actual api calls. It gives a good visual, even though the initial setup might be more verbose compared to rswag. I've found that teams who prefer a clearer separation between tests and documentation might find this more appealing.

Finally, let's touch upon a more customized approach using the `openapi` or swagger specification directly, coupled with a gem like `rswag-api`. While rswag provides its own approach to defining the openapi spec, sometimes you may need more direct control over your specification. In my past projects, I have occasionally generated or even manually composed my `openapi.json` or `openapi.yaml` specification, particularly when incorporating external or complex api schemas.

This allows for a more detailed control over the data representation. Here’s how you can integrate a manually written openapi spec with `rswag-api`:

```ruby
# Assuming you have an openapi.json file describing your api

Rswag::Api.configure do |config|
    config.swagger_root = Rails.root.join('path_to_your_openapi_directory').to_s
    config.swagger_docs = {
      'v1/swagger.json' => {
        openapi: '3.0.1',
        info: {
          title: 'My API',
          version: 'v1'
        },
        paths: {
              '/products': {
                    get: {
                        summary: "Retrieves products",
                        produces: ["application/json"],
                        parameters: [
                          {
                            name: :page,
                            in: :query,
                            description: "Page number",
                            required: false,
                            schema: {type: :integer}
                          },
                          {
                            name: :per_page,
                            in: :query,
                            description: "Products per page",
                            required: false,
                            schema: {type: :integer}
                            }
                          ],
                         responses: {
                              "200": {
                                description: "successful operation",
                                content: {
                                  "application/json": {
                                    schema: {
                                     type: :array,
                                      items: {
                                       type: :object,
                                        properties: {
                                              id: { type: :integer },
                                              name: { type: :string },
                                              price: { type: :number, format: :float }
                                                },
                                              required: %w[id name price]
                                              }
                                           }
                                     }
                                  }
                              }
                            }

                        }
                 }
           }
      }
    }

    config.app_root = Rails.root
end
```
This configuration tells rswag to use your existing `openapi.json` spec file for documentation. Rswag's api engine is great here as it still provides that visual and testable layer on top. The advantage here lies in fine-tuning, for example, with tools like the swagger editor. It allows you complete flexibility to define the entire api spec, which can be important when dealing with legacy systems or non-standard api design.

Choosing the right approach depends heavily on your project needs, your team's preferences, and how granular you need to control your documentation. In any case, remember that consistency is key. If you go with rswag or apipie, commit to fully describing your api throughout its lifetime. If you opt for direct openapi definitions, make sure you validate them and keep them up to date. Having a centralized and interactive documentation platform should significantly streamline your api development and adoption processes.

For further reading, I'd recommend delving into the OpenAPI specification documentation directly (often referred to as Swagger) to understand the full potential of the specification, it is maintained by the OpenAPI Initiative. A great resource for `rswag` is its own github repository, as it includes extensive examples and documentation. Additionally, the `apipie` gem provides a comprehensive tutorial on its usage, which will be helpful when considering it as an option. These sources will help you delve deeper into the concepts and practical applications I’ve discussed, and will certainly help you better navigate the complexities of api documentation in rails.

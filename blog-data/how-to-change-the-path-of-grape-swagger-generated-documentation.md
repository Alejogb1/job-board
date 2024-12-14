---
title: "How to Change the path of grape-swagger generated documentation?"
date: "2024-12-14"
id: "how-to-change-the-path-of-grape-swagger-generated-documentation"
---

alright, so you're looking at changing the output path for grape-swagger, got it. i've been there, spent way too long staring at config files and console outputs trying to get that swagger ui to show up where i actually wanted it. it's one of those things that feels like it *should* be simple, but then you hit some weird default and have to go digging.

from the sounds of it, you're using grape, which is cool – i've done a lot of api work with it. and the grape-swagger gem, which is pretty much the standard way to generate swagger docs from your api definitions. the problem usually comes down to grape-swagger, by default, sticking things in a particular spot and if you're not using the default path, things get a bit wonky.

the key here is understanding that grape-swagger, by design, expects you to provide the base path. if you don't define it, it will pick one. sometimes the gem does a pretty good job guessing but there are cases that it gets it wrong, that is just life. it is mostly dependent on the way you use grape in your code.

so here's what i'd do: first, take a look at your `api.rb` (or whatever you've named your grape api file). you'll need to configure grape-swagger to use the path you want. there are two ways to achieve this; the first and more standard one is adding a `mount` statement at the top level of your grape file. the other one is by explicitly defining the base path on the configuration block. the second approach is the one i recommend as it helps debug problems.

let's assume your api has something like this:

```ruby
# api.rb
module MyAPI
  class Base < Grape::API
    format :json
    prefix :api # you may have something similar
    version 'v1', using: :path
    # ... other configurations
  end

  class Users < Base
      # your users api definitions
  end
end
```

if you want the swagger docs to be available under the path `/api/v1/docs`, you'd need to update the `Base` class to tell grape-swagger. there is another way i see most people do which is to `mount` directly a swagger endpoint. that works and sometimes can make it simpler to manage. however, i always recommend to explicitly set the base path as you need to have less dependencies in one block of code.

here is how i usually update the `Base` api file:

```ruby
# api.rb
require 'grape-swagger'

module MyAPI
  class Base < Grape::API
    format :json
    prefix :api
    version 'v1', using: :path
    add_swagger_documentation(
      base_path: '/api', # notice the path is the same as the prefix.
      mount_path: '/docs',
      api_version: 'v1'
    )
      # ... other configurations
  end

  class Users < Base
      # your users api definitions
  end
end
```

notice the `base_path`. it needs to match your api's prefix. the `mount_path` defines the endpoint relative to base. so, in this case, the swagger ui will appear at `/api/v1/docs`.

now, this is what most people do but when the paths get deeper i recommend that you take the time to create some helper method to define the base path automatically. for example if you have an environment variable that defines that, you can create a method and extract the data to a method and then define the base path dynamically. like this:

```ruby
# api.rb
require 'grape-swagger'

module MyAPI
  class Base < Grape::API
    format :json

    def self.api_base_path
      if ENV['API_BASE_PATH'].present?
       ENV['API_BASE_PATH']
      else
        '/api'
      end
    end

    prefix :api
    version 'v1', using: :path
    add_swagger_documentation(
      base_path: api_base_path,
      mount_path: '/docs',
      api_version: 'v1'
    )
      # ... other configurations
  end

  class Users < Base
      # your users api definitions
  end
end
```

now you can set an environment variable called `API_BASE_PATH` to something like `/another-api` and the swagger ui will appear under `/another-api/v1/docs`. pretty handy if you're using docker or you're deploying to different environments.

that's the approach I usually prefer because is very predictable and i can debug in more detailed ways. but as i mentioned previously you could also `mount` a swagger endpoint, which is also valid for simpler use cases or if you have a very large app where you want more flexibility and mount the swagger ui in a different endpoint. here's how it would look. notice this is not within the `add_swagger_documentation` block.

```ruby
# api.rb
require 'grape-swagger'

module MyAPI
  class Base < Grape::API
    format :json
    prefix :api
    version 'v1', using: :path
    # ... other configurations
  end

  class Users < Base
      # your users api definitions
  end

  class Docs < Grape::API
     add_swagger_documentation(
      base_path: '/api',
      api_version: 'v1',
     )
  end
  Base.mount Docs => '/docs' # notice the mounting
end
```

in this example, we're defining another class called `Docs` that inherits `Grape::API` and we use `add_swagger_documentation`. then in our base class, we mount it as `/docs`. it accomplishes the same but it adds a bit of complexity to the project structure.

now, a common mistake (and one that i have done myself way too many times) is forgetting to restart the server or the rack app when you make changes to the api files. i know it sounds basic but it has happened to me and is very easy to overlook. so before going down a rabbit hole triple check you have restarted the server.

another common mistake happens when you have complex routes, in that scenario is important that you have defined the `add_swagger_documentation` at the top level `Base` class, otherwise, grape-swagger gets confused.

when it comes to diving deeper into grape and swagger specifically, i highly suggest you check the source code, is very well written and has all the details for you to better understand the inner workings of the gem. you can find it on github of course. there are no papers, at least none that i know of, but there are amazing books about ruby apis that will give you a great base to build amazing things.

i do remember that one time i spent a whole day fighting with a weird routing issue and the swagger doc was appearing all wrong. turns out, i had a typo in the `base_path` and that i was also using a different version of grape than the one supported by the grape-swagger gem. i felt like i should be working on enterprise software instead of debugging gems. it's amazing how a simple typo can make your life miserable.

and don't forget, when things get really weird, enable debugging in grape. it can give you detailed logging of the routing process which can be helpful when things do not behave as you expect them to. in grape you can do it with: `Grape::API.logger.level = Logger::DEBUG`. you can do it in one of the config files or within your grape api files.

hope this helps you get your swagger docs in order. remember to double-check the `base_path` and `mount_path`, make sure your server is restarted, and check that you're using the proper grape-swagger version for your version of grape. and please, never forget to enable debugging, it will save you a lot of time and frustration.

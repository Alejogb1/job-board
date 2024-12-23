---
title: "How can I seed a Ruby on Rails database using GraphQL and JSON responses?"
date: "2024-12-23"
id: "how-can-i-seed-a-ruby-on-rails-database-using-graphql-and-json-responses"
---

Okay, let's talk database seeding with GraphQL and JSON responses in a Rails application. This is a problem I've encountered a few times, often when setting up demos or needing more realistic test data. The typical Rails `seeds.rb` file just doesn't cut it when you have a complex data model and want to leverage the existing GraphQL API for data creation. Directly creating records via ActiveRecord isn’t as robust because it bypasses all validations, business logic, and can cause inconsistencies with your application’s data. So, using the GraphQL API is crucial for fidelity.

My past experiences taught me that a straightforward approach is best: query the GraphQL endpoint with crafted JSON payloads for mutations, parse the response, and handle any errors gracefully. I’ll show you a few methods to approach this, each with a specific use case. I’ll also include specific code examples so you can see how it translates to a real project. The goal isn’t just to inject data; it’s to mimic the actual application workflow as much as possible to ensure consistency with the production setup.

Let's start with the most fundamental scenario: seeding with a single, well-defined mutation. Imagine a situation where you need to create a user with a specific username and email using a `createUser` mutation. You wouldn't want to bypass your authorization layer or validation logic. Here’s the Ruby code:

```ruby
require 'net/http'
require 'uri'
require 'json'

def execute_graphql_mutation(graphql_url, mutation_query, variables = {})
  uri = URI(graphql_url)
  http = Net::HTTP.new(uri.host, uri.port)
  http.use_ssl = true if uri.scheme == 'https'

  request = Net::HTTP::Post.new(uri.request_uri)
  request['Content-Type'] = 'application/json'
  request.body = { query: mutation_query, variables: variables }.to_json

  response = http.request(request)

  if response.code.to_i >= 400
    raise "GraphQL request failed: #{response.code} - #{response.body}"
  end

  JSON.parse(response.body)
end

graphql_endpoint = 'http://localhost:3000/graphql' # Replace with your graphql endpoint.

mutation = <<~GRAPHQL
  mutation CreateUser($username: String!, $email: String!) {
    createUser(input: { username: $username, email: $email }) {
      user {
        id
        username
        email
      }
      errors {
        message
        field
      }
    }
  }
GRAPHQL

variables = { username: 'testuser', email: 'test@example.com' }

begin
  result = execute_graphql_mutation(graphql_endpoint, mutation, variables)
  if result['data'] && result['data']['createUser'] && result['data']['createUser']['user']
    puts "User created successfully: #{result['data']['createUser']['user']}"
  elsif result['data'] && result['data']['createUser'] && result['data']['createUser']['errors']
    puts "Error creating user: #{result['data']['createUser']['errors']}"
  else
      puts "Unexpected response format: #{result}"
  end
rescue StandardError => e
  puts "An error occurred: #{e.message}"
end
```

In this example, `execute_graphql_mutation` handles sending a POST request with your query, converting the response from json to ruby hashes, and error checking. The mutation string is defined as a heredoc for readability and uses GraphQL variables for dynamic values. We handle both success and failure scenarios, displaying appropriate messages to the console. This basic structure can be adapted to other mutations by simply modifying the mutation query and variables.

Now, let's move onto a more complex situation. You might need to seed a series of related objects—for example, create multiple posts for a user using a `createPost` mutation. It wouldn’t be efficient to call the above function multiple times. For this, we can loop over an array of parameters. Here's a modified example that takes an array of user and post data and creates both in a controlled manner:

```ruby
require 'net/http'
require 'uri'
require 'json'

def execute_graphql_mutation(graphql_url, mutation_query, variables = {})
  uri = URI(graphql_url)
  http = Net::HTTP.new(uri.host, uri.port)
  http.use_ssl = true if uri.scheme == 'https'

  request = Net::HTTP::Post.new(uri.request_uri)
  request['Content-Type'] = 'application/json'
  request.body = { query: mutation_query, variables: variables }.to_json

  response = http.request(request)
  if response.code.to_i >= 400
    raise "GraphQL request failed: #{response.code} - #{response.body}"
  end
  JSON.parse(response.body)
end


graphql_endpoint = 'http://localhost:3000/graphql' # Replace with your GraphQL endpoint

user_mutation = <<~GRAPHQL
  mutation CreateUser($username: String!, $email: String!) {
    createUser(input: { username: $username, email: $email }) {
        user {
            id
            username
        }
        errors {
            message
        }
    }
  }
GRAPHQL

post_mutation = <<~GRAPHQL
    mutation CreatePost($userId: ID!, $title: String!, $body: String!) {
        createPost(input: { userId: $userId, title: $title, body: $body }) {
            post {
                id
                title
            }
            errors {
                message
            }
        }
    }
GRAPHQL

seed_data = [
  {
    user: { username: 'user1', email: 'user1@example.com' },
    posts: [
      { title: 'First Post', body: 'This is the first post.' },
      { title: 'Second Post', body: 'This is the second post.' }
    ]
  },
  {
      user: { username: 'user2', email: 'user2@example.com' },
      posts: [
          { title: 'Another Post', body: 'Here is another one.'}
      ]
  }
]

seed_data.each do |data|
  user_variables = data[:user]
  begin
      user_result = execute_graphql_mutation(graphql_endpoint, user_mutation, user_variables)
      if user_result['data'] && user_result['data']['createUser'] && user_result['data']['createUser']['user']
          user_id = user_result['data']['createUser']['user']['id']
        puts "User created successfully: #{user_id}"
        data[:posts].each do |post_data|
           post_variables = {userId: user_id, title: post_data[:title], body: post_data[:body] }
           begin
               post_result = execute_graphql_mutation(graphql_endpoint, post_mutation, post_variables)
               if post_result['data'] && post_result['data']['createPost'] && post_result['data']['createPost']['post']
                  puts "Post created: #{post_result['data']['createPost']['post']}"
               else
                 puts "Post creation error: #{post_result}"
               end
            rescue StandardError => e
               puts "Error creating post: #{e.message}"
            end
        end
      else
         puts "User creation error: #{user_result}"
      end
   rescue StandardError => e
       puts "Error creating user: #{e.message}"
  end
end
```

In this code, we have two separate mutation queries: one to create a user and another to create posts for a user. We iterate over the `seed_data` array, which specifies users and their associated posts. After creating a user, we retrieve the user's id and then create posts, one at a time. This ensures our database has properly related records. I've seen projects become much more consistent by seeding like this, avoiding data corruption from bypassing normal application flows.

Lastly, let’s discuss error handling. While our examples do provide simple error logging, in real-world situations you often need more detailed error handling and logging. Consider that mutations might have nested errors. Let’s enhance our first example to show how to handle nested errors properly:

```ruby
require 'net/http'
require 'uri'
require 'json'

def execute_graphql_mutation(graphql_url, mutation_query, variables = {})
  uri = URI(graphql_url)
  http = Net::HTTP.new(uri.host, uri.port)
  http.use_ssl = true if uri.scheme == 'https'

  request = Net::HTTP::Post.new(uri.request_uri)
  request['Content-Type'] = 'application/json'
  request.body = { query: mutation_query, variables: variables }.to_json

  response = http.request(request)

  if response.code.to_i >= 400
    raise "GraphQL request failed: #{response.code} - #{response.body}"
  end
  JSON.parse(response.body)
end

graphql_endpoint = 'http://localhost:3000/graphql' # Replace with your GraphQL endpoint

mutation = <<~GRAPHQL
  mutation CreateUser($username: String!, $email: String!) {
    createUser(input: { username: $username, email: $email }) {
      user {
        id
        username
        email
      }
      errors {
        message
        field
      }
    }
  }
GRAPHQL

variables = { username: '', email: 'test@example.com' } # intentionally invalid username

begin
  result = execute_graphql_mutation(graphql_endpoint, mutation, variables)
  if result['data'] && result['data']['createUser']
    create_user_result = result['data']['createUser']
      if create_user_result['user']
        puts "User created successfully: #{create_user_result['user']}"
    elsif create_user_result['errors']
      create_user_result['errors'].each do |error|
          puts "Error creating user: Field '#{error['field']}' - Message: '#{error['message']}'"
      end
    else
        puts "Unexpected response format: #{result}"
    end

  else
    puts "Unexpected response format: #{result}"
  end
rescue StandardError => e
  puts "An error occurred: #{e.message}"
end
```

Here, we are intentionally sending an invalid username (an empty string). Instead of relying on a generic error message, we loop over the errors array within the response and display detailed error information including the field that failed and the error message. In my experience, this type of granular error checking is critical when seeding a database as you get early warning to any issues in your schema.

For further reading, I would highly recommend “GraphQL in Action” by Samer Buna, and the official GraphQL specification documentation. These two sources can really help you solidify your understanding of GraphQL in both practical and theoretical ways. Additionally, “Programming Ruby 1.9 & 2.0: The Pragmatic Programmers' Guide” by Dave Thomas provides a fantastic resource for deepening your understanding of the core Ruby language itself which always helps. These resources can be instrumental as you develop more complex seeding strategies for your Rails applications.

This approach, focusing on sending properly formatted JSON requests, interpreting the responses correctly, handling errors explicitly, and using real GraphQL mutations, gives you a more realistic approach to seeding your application’s data. This has helped me many times, and it's a methodology I encourage others to follow for more consistent and maintainable Rails projects.

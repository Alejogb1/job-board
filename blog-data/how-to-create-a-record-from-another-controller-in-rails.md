---
title: "How to Create a record from another controller in Rails?"
date: "2024-12-14"
id: "how-to-create-a-record-from-another-controller-in-rails"
---

alright, so you're trying to create a record in one rails controller from another, eh? i've been there, more times than i’d like to remember. it's one of those things that seems simple on the surface but can quickly turn into a tangled mess if you’re not careful. i remember back in the day, when i was first learning rails, i tried doing this and ended up with a controller that was basically a swiss army knife – it was doing everything and nothing well. let’s avoid that, shall we?

the core issue here is decoupling. controllers ideally should be focused on handling requests and responses, not directly interacting with the database through active record models unless it’s part of their own crud operation. so, directly creating a record from another controller is generally not the recommended approach because we want to keep our logic nicely separated and our models doing what they should.

let’s think about this. one way is to pass the data from the initiating controller to the other controller, and the second controller can then create the record. however that adds complexity that makes things hard to maintain if you need the operation in many places.

a cleaner and more maintainable way to handle this is by using service objects or, as i often prefer, model callbacks. both methods keep the core functionality where it belongs: in the models, thus reducing the complexity of the controllers. 

let's start with a classic example. let's say you have a `posts_controller` and a `comments_controller`. you want a comment to be created after a post is created. here’s a way to accomplish this using a model callback.

first, in your `post.rb` model file:

```ruby
class Post < ApplicationRecord
  has_many :comments, dependent: :destroy
  after_create :create_initial_comment

  private

  def create_initial_comment
    comments.create(content: "this is the first comment")
  end
end
```

in this snippet, after a new post is created, the `create_initial_comment` method gets automatically called through the `after_create` callback. this method then creates a default comment. it’s neat, compact, and maintains the logical boundary between model creation and comment handling. note: `dependent: :destroy` ensures that when the post is deleted the related comments are also removed.

now you might ask, what if i need more control over how the comment is created? or, what if the logic is more complex? that’s where a service object comes in handy.

let’s imagine a scenario where we have a `user` and an `account` model. when a new user is created, we want to also create a default account for them. this time we are not using callbacks because we need more flexibility, that’s when the service pattern is more useful. in `app/services`, we'll create `account_creator.rb`:

```ruby
class AccountCreator
  def self.create_default_account(user)
    Account.create(user: user, balance: 0.0, account_type: 'default')
  end
end
```

now, in your `users_controller.rb` (where you create users), after creating a new user we can call the service.

```ruby
class UsersController < ApplicationController
  def create
     @user = User.new(user_params)
     if @user.save
       AccountCreator.create_default_account(@user)
       redirect_to @user, notice: 'user created and account created!'
     else
       render :new
     end
  end

  private

  def user_params
    params.require(:user).permit(:name, :email, :password)
  end
end
```

in this approach, the `users_controller` still creates a `user` but it doesn’t handle the complexity of how an account should be created. that logic lives within the `accountcreator` service. this is a cleaner and much more maintainable solution, specially as the complexity grows. it keeps the controller focused on just handling requests. the model and the service contain the business logic.

as for when to use callbacks vs service objects, a good rule of thumb is this: if the logic is very simple and tightly coupled with the model, go for callbacks. if the logic is more complex, involves multiple models, or if you plan to reuse it in multiple places, a service object is the way to go. sometimes i even use a combination of both. for instance a callback may call a service object, or the service may call another service. it’s turtles all the way down.

about resources for learning this, i would not point you to tutorials or blog posts (they may become obsolete quickly). instead, i recommend checking out the "domain driven design" book by eric evans, it is a gem and talks a lot about service patterns and why we need them. also, "refactoring ruby edition" by martin fowler is also a good read on how to move logic between files. the rails guides about active record callbacks are very useful, even if you’re going with the services approach you'll need to know them.

remember, the goal isn't just to get your code working, but to make it maintainable and easy to understand. decoupling your controllers and models will save you time and headaches in the long run. i had a project once where i mixed all the code on the controllers and after a month, i could not even read the code myself. i had to rewrite it, so, i can say i learned this the hard way. avoid that. keep it clean. keep it simple. keep it separated.

oh, and, just a quick joke before i go, why do programmers prefer dark mode? because light attracts bugs! haha, i'm here all week.

if you have any follow-up questions, drop them below. i'm always here to help, and i’m happy to share more of my hard-earned wisdom. good luck!

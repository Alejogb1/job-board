---
title: "How can follower notifications be implemented in Ruby on Rails using the noticed gem?"
date: "2024-12-23"
id: "how-can-follower-notifications-be-implemented-in-ruby-on-rails-using-the-noticed-gem"
---

Alright, let's tackle this. I've definitely been down this road before, having built a fairly complex social feed system for a previous project that relied heavily on notifications. We opted for the `noticed` gem back then, and it proved to be a solid choice for managing those asynchronous updates. Let me walk you through the practicalities of implementing follower notifications, as we did.

First off, `noticed` is not a magic bullet. It's a powerful tool, but you'll still need to structure your models and logic accordingly. The core concept is that some action (say, a user following another user) triggers a notification which gets delivered to the appropriate recipient. In the context of follower notifications, the "recipient" is the user being followed, and the trigger is the act of a different user initiating the follow action.

The initial setup involves ensuring you have `noticed` installed correctly. The standard gem installation process applies here: `gem install noticed` followed by adding `gem 'noticed'` to your `Gemfile` and running `bundle install`. Then you need to generate the migration for noticed and run it with `rails noticed:install`, followed by `rails db:migrate`. This step sets up the necessary tables for tracking your notifications.

Next comes defining your notification. We’ll create a `FollowerNotification` class under `app/notifications`. This class inherits from `Noticed::Base`:

```ruby
# app/notifications/follower_notification.rb
class FollowerNotification < Noticed::Base
  deliver_by :database
  # Optional: deliver_by :email, mailer: "UserMailer", method: "new_follower"

  param :follower_id

  def follower
    User.find(params[:follower_id])
  end

  def message
    "#{follower.username} started following you."
  end
end
```

In this simple example, we specify that we'll deliver via the database (you could also add email delivery if desired), and we define a `param`, which is the `follower_id`, that allows passing of data. The `follower` method retrieves the `User` record based on this passed ID and the `message` constructs the notification text. It's crucial to keep these methods concise and specific to the notification type.

Now, let's look at where we *trigger* the notification, which is typically in the `create` action of your follow relationship. Assume you have a `Follow` model that represents a user following another user:

```ruby
# app/models/follow.rb
class Follow < ApplicationRecord
  belongs_to :follower, class_name: 'User', foreign_key: 'follower_id'
  belongs_to :followed, class_name: 'User', foreign_key: 'followed_id'

  after_create :notify_followed_user

  private

  def notify_followed_user
    FollowerNotification.with(follower_id: follower_id).deliver_later(followed)
  end
end
```

Here, we define a callback `after_create` that calls the `notify_followed_user` method, passing the `follower_id`. We then utilize `FollowerNotification.with` to pass our parameters along with `.deliver_later(followed)`, so the notification is delivered asynchronously, avoiding blocking in the user's interaction, and will be delivered to the user being followed.

Let's say you want to add an action to the controller for creating followers, and your parameters are going to come from the client, here's an example:

```ruby
# app/controllers/follows_controller.rb
class FollowsController < ApplicationController
  def create
     followed = User.find(params[:followed_id])
     follow = current_user.followings.build(followed: followed)

     if follow.save
      render json: { message: "Now Following", status: :created }, status: :created
     else
        render json: { errors: follow.errors.full_messages }, status: :unprocessable_entity
    end

  end
end
```

Now we have a basic setup. The act of creating a `Follow` record generates a notification.

The final piece, and arguably the most crucial for a good user experience, is displaying the notifications. You’ll usually want to fetch unread notifications associated with the current user and display them in a digestible way in your view. Consider the controller logic:

```ruby
# app/controllers/users_controller.rb
class UsersController < ApplicationController
  def show
    @user = User.find(params[:id])
    @notifications = @user.notifications.unread.order(created_at: :desc)
  end
end
```

Here, `current_user.notifications` gets all notifications, and we chain `.unread` to get only the unread ones, ordered by the date they were created, with the most recent first. Then, in your view, you would iterate through the `@notifications` to display the message, ideally with the option to mark them as read:

```erb
# app/views/users/show.html.erb
<% @notifications.each do |notification| %>
  <div class="notification">
    <%= notification.message %>
    <%= link_to "Mark as Read", notification_path(notification), method: :patch %>
  </div>
<% end %>
```

This view iterates through each notification and displays the message along with a link to mark them as read. In your routes you would have set this up with `resources :notifications` along with the needed update patch functionality in your `NotificationsController`.

That's the core implementation. Keep in mind a few things, from my experience:

*   **Batching and Throttling:** If you anticipate a large volume of notifications, consider batching them to reduce database load and optimize for efficient delivery. `noticed` provides mechanisms for this. You will want to read the documentation to make sure your notification process is running efficiently.
*   **Customizing Delivery:** `noticed` also supports custom delivery mechanisms. If you have specific requirements, this is a good place to look, if sending emails via third-party platforms you can configure delivery to use your provider of choice.
*   **Read/Unread Tracking:** Carefully design how you manage the "read" status. `noticed` will help you here, but you will need to create the logic for updating the notification statuses. Think about situations where a notification is read somewhere but needs to be updated elsewhere.
*   **Security:** Always sanitize notification content. Especially if you are taking user inputs as part of generating the message.
*   **Database Indexing:** Ensure that your `notifications` table has appropriate indexes for `recipient_id` and `created_at`. This is crucial for efficient querying as your user base grows.
*   **Testing:** Thoroughly test your notification workflow. Make sure notifications are delivered correctly, that they are displayed appropriately in your views, and that the read status updating is working as expected. A test suite is essential in your development lifecycle, don't skip it.

For further reading on this subject, I highly recommend delving into "Agile Web Development with Rails 7" by Samuel J. Davis. The book has comprehensive material on working with models, controllers, and views and has a dedicated section on active job which is relevant to the asynchronous nature of notifications. Additionally, the official documentation of the `noticed` gem on GitHub should be a primary reference for specific details and features of the gem, and you'll find other excellent sources on asynchronous programming principles.

This is not a complete solution to any one system, but it should provide a solid foundation for how to implement follower notifications in your Rails application using `noticed`. Remember to iterate and refine based on your specific needs and as always, to read the documentation. Good luck with your implementation.

---
title: "Why is Rails: Updating Item creates 2 records, one of the updated item and one of the old item?"
date: "2024-12-14"
id: "why-is-rails-updating-item-creates-2-records-one-of-the-updated-item-and-one-of-the-old-item"
---

here's what i think is happening, and how to fix it, based on my own share of head-scratching moments with rails. i’ve been building rails apps for, well, let’s just say a long time, and this specific issue – the double record creation on what should be a simple update – is a classic gotcha that has tripped me up more than once.

what you're seeing is that rails, particularly when dealing with models and persistence, isn't always as straightforward as we’d like. when you “update” an item, rails doesn't just go into the database and change the values in a single row. instead, it's often doing something more like a 'find,' then ‘modify’, and finally an 'insert' operation. the core of the issue comes down to how rails handles attribute assignment coupled with how some of the model persistence magic is working under the hood when it comes to certain operations like those in controllers or in the model itself.

it sounds like, in your case, the "update" operation isn’t actually updating the existing record directly. instead it's creating a *new* record reflecting the changes, which leaves the original record untouched. what makes this happen so often is usually a misconception of how the update operations on active record instances should be used or how the validations and other callbacks are interfering.

the most common culprit is that you’re probably creating a new object, then trying to "save" that object thinking it will automatically update an existing entry. for example, if in your controller code you're finding an object like this:

```ruby
def update
  @item = Item.find(params[:id])
  @item = Item.new(item_params)
  if @item.save
    redirect_to @item
  else
    render :edit
  end
end
```

this snippet will *always* create a new record, and not update the existing one, since you're reassigning `@item` to be a *new* instance with the new parameters, ignoring the original found one. you need to merge the new parameters into the found instance. this is a frequent beginner mistake. the `Item.new` in the second line is the primary cause of the problem.

another common problem arises when dealing with nested attributes, especially if there are any validations or callbacks tied to these attributes. these validations or callbacks might, under certain conditions, prevent the *update* and trigger the creation of a new record. this is often more subtle and can appear in more complex models or when validations are improperly written. also if you are using a create method instead of an update method in your controller this may happen.

a common example of this with nested attributes can be when you use a form and send parameters through it to the controller.

let’s assume you have a `User` model with a nested `Profile` model. the user controller might have:

```ruby
  def update
    @user = User.find(params[:id])
    if @user.update(user_params)
      redirect_to @user
    else
      render :edit
    end
  end

  private

  def user_params
    params.require(:user).permit(:name, profile_attributes: [:id, :bio])
  end
```

and inside of your model if you are not using the correct setting, an update might create an issue. for instance if we have this model:

```ruby
class User < ApplicationRecord
    has_one :profile, dependent: :destroy
    accepts_nested_attributes_for :profile, allow_destroy: true
end

class Profile < ApplicationRecord
    belongs_to :user
    validates :bio, presence: true
end
```

in this example, if the parameters passed in `user_params` for `profile_attributes` do not contain an `:id` for an existing profile and you try to update this it will create a new profile, since it cannot find the current profile entry, and it will try to create one, it might seem silly, but it happens quite often, and it causes this update behaviour to go sideways. it would also cause problems if the bio attribute was not sent.

the fix depends on what’s causing the problem in your app, but usually it involves these core changes:

*   **ensure you're updating the existing object**. do not re-initialize an object. instead use the `update` method or `assign_attributes` to modify an existing record instance, and *then* save.
*   **review your model validations carefully**. if there are validations failing on your update action, you need to investigate them. this may not be the obvious issue, but a check-up will ensure your model logic is bullet-proof.
*   **inspect your nested attributes behavior closely**. if you are using nested attributes make sure your parameters are set correctly and your `accepts_nested_attributes_for` and associated configurations are working as expected and your form sends the proper parameters.

here’s a revised version of the first snippet that illustrates the correct way:

```ruby
def update
  @item = Item.find(params[:id])
  if @item.update(item_params)
    redirect_to @item
  else
    render :edit
  end
end
```

notice i replaced `Item.new` with `@item.update(item_params)`. this will find the item, then merge the new parameters into that item and update the existing row. this is much more clean and explicit.

also, make sure to keep an eye on any callbacks defined in the model, as these are executed before and after operations like save, create or update. callbacks are useful but can have unintended consequences if not carefully written. for example, let’s say you have a callback in your `item` model to log changes:

```ruby
class Item < ApplicationRecord
  before_update :log_item_changes

  private
    def log_item_changes
      puts "item id: #{self.id}, before updated, parameters: #{self.changes}"
    end
end
```
this callback won't create a new object but it might help you debug your issue because it will print the parameters and the id of the object you are updating so you can make sure you are using the right instance. it will also show the `changes` made so you can use it to debug if parameters were sent correctly.

if you're trying to understand active record and these kinds of behavior, the official rails guides are probably your best resource. they can be found on the rails website. the "active record basics" section is a particularly good place to start and can be used as a reference point. and look into the source code of active record itself, it can be very enlightening. you can find it on the rails github page. another great book that is now a bit old but has some strong foundations about these concepts is “agile web development with rails”. though it is quite old, most of the core principles are still valid to this day. and let me tell you: debugging this sort of thing always feels like a treasure hunt. sometimes, though, the treasure is just correctly placed parameters.

it's one of those rails quirks that gets everyone at some point. but once you’ve worked with it, it all makes sense, or at least, it makes a bit more sense. it happened to me once i had an `after_create` hook that would create another record. and it was super confusing for a week until i realised it, it felt like i was in some kind of rails inception scenario. don't feel too bad about this, we all been there at some point, and probably will face again some variation of this issue, but now at least, we know the problem's face.

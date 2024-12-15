---
title: "How to handle Rails 6 & route for custom post- action for a nested resource & form?"
date: "2024-12-15"
id: "how-to-handle-rails-6--route-for-custom-post--action-for-a-nested-resource--form"
---

alright, so you've got this nested resource situation in rails 6, and you need a custom action for a post request, plus a form to go with it. i've been there, trust me. it's one of those things that seems simple on the surface but can get hairy real quick. i remember back in the rails 4 days, dealing with similar stuff, and let me tell you, the learning curve was... steep. took me a solid week once to fix a routing issue that looked like it was caused by a ghost. just a misconfigured route with a spelling error that i could not see, funny enough after i slept for 8 hours i spot it right away, it is like my brain had to compile the code without my consciousness.

anyway, let's break this down. it's all about the routes, controllers, and form setup. so, first routes.rb. you need to make sure you're explicitly declaring that custom post route within the nested resource. rails isn't always magic, you need to be precise. we need to move past default resourceful routes for this special case.

```ruby
  # config/routes.rb
  resources :authors do
    resources :books do
      post 'custom_action', on: :member # member for /authors/1/books/2/custom_action
    end
  end

```

see how we use 'post 'custom\_action', on: :member'? that's key. `on: :member` means it applies to an individual book under author, not a collection. if you use `on: :collection` it would make the url /authors/1/books/custom\_action without an id for the book which is not what you want. the action is also a `post` not a get, since this action is going to mutate data. there are some cases you may use a `get` for a custom action but i digress.

now, the controller part, you need a corresponding method in your `books_controller.rb`. think of it like the receiving end of that route:

```ruby
  # app/controllers/books_controller.rb
  class BooksController < ApplicationController
    before_action :set_author
    before_action :set_book, only: [:custom_action]

    def custom_action
      if @book.update(custom_params)
        redirect_to author_book_path(@author, @book), notice: 'Book updated successfully'
      else
        render :show, status: :unprocessable_entity
      end
    end

    private

    def set_author
      @author = Author.find(params[:author_id])
    end

    def set_book
      @book = @author.books.find(params[:id])
    end

    def custom_params
      params.require(:book).permit(:title, :other_custom_field) # adapt these to match your needs
    end
  end
```

notice the `before_action`. that helps to set up the `@author` and `@book` objects so you don't have to repeat that in every action. it's about keeping your controller actions lean. that custom\_params method is crucial; it's your permit list of allowed parameters, always good to prevent mass assignments. you need to have the form parameters be nested under the book object. If it does not match your form parameters, rails won't pass it as params[:book] or whatever your named object is so keep an eye on that, it is a common rookie error. also, i'm redirecting back to the book show action after, assuming you have that view available. and if something goes wrong the error messages of the model will automatically be displayed.

next is the view form setup. this is where the form comes in, it needs to point towards the route we just made. think of it as the input side of the route we defined earlier. here is where it gets a little tricky, you need to use the url helper method that rails provide for nested actions.

```erb
  <!-- app/views/books/show.html.erb -->
  <%= form_with url: custom_action_author_book_path(@author, @book), method: :post, model: @book do |form| %>
    <%= form.text_field :title %>
    <%= form.text_field :other_custom_field %>
    <%= form.submit 'Update Custom' %>
  <% end %>

```

take a good look at `custom_action_author_book_path(@author, @book)`. this creates the correct url based on the routes we set up. you don't wanna manually write the url yourself, that's what helpers are for! model: @book helps us keep the form bound with our `@book` object and so the params can be nested correctly. also, notice we're using `form_with` which is the recommended way in rails 6 and above. no more `form_tag` unless you really need it.

now, couple of pointers i have picked up in the past years. if the forms seem not to be working, the rails server log is your friend. it is usually telling you exactly what is going on, even if it is a cryptic message. pay close attention to the parameters. are they being passed correctly? if you're receiving a `params[:book]` hash, that is a good sign, if it is not there then you probably have an issue with your form parameters or an incorrect nested object. if `before_action` is not finding the object it throws an error, so you can start debugging there. also, remember the `custom_params` is your friend, it should match exactly the parameters of your form.

now, for some learning resources. i highly recommend the "agile web development with rails 6" book. it is an excellent reference for most rails stuff. for nested resources and routing specifically i always find myself coming back to rails guides about routing at https://guides.rubyonrails.org/routing.html. although the guides are online documentation it is a good practice to keep a local copy of the document on your local computer for fast offline access.

i've had my share of head-scratching moments with nested resources, but with a bit of practice and attention to the details, you will be handling complex forms in no time. it's like learning a new language, at the beginning is hard, then you master the grammar, and after a while it starts becoming second nature. remember, rails is very opinionated, and following the rails way of doing things simplifies your life.

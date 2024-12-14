---
title: "How can I revert to a fallback in a Rails has_many dependent: :destroy scenario?"
date: "2024-12-14"
id: "how-can-i-revert-to-a-fallback-in-a-rails-hasmany-dependent-destroy-scenario"
---

alright, so you’re dealing with a `has_many` association in rails, and specifically, the dreaded `dependent: :destroy` clause. it’s a common pitfall, believe me, i've been there, staring at my logs at 3 am wondering why the entire application just went kaput. the standard `dependent: :destroy` is a one-way ticket: when the parent record goes, so do all of its children. no take-backs, no second chances. but, it seems you need something more... nuanced. you want a fallback mechanism, some way of preserving those child records when the parent decides to check out.

i get it. i remember once, back in my early days of rails, i was working on an e-commerce application. we had a `user` model `has_many :orders, dependent: :destroy`. seemed straightforward enough, until some product manager came along and told me that, “users should not be able to delete their account if they had orders, we need to keep all the historical data, we need to keep track of our customer's purchases, no matter what.” that's when i learned `dependent: :destroy` wasn't always the silver bullet, we needed something a bit less... *final*.

the core issue is, as you've noticed, that `dependent: :destroy` happens inside the `after_destroy` callback of the parent model. this is a rails magic method that gets invoked after a record is destroyed but before the transaction commits. when it triggers, the children records are gone, poof! no recovery mechanism exists unless you build one.

so, how do we achieve the fallback? well, let’s break this down: you don’t really want to stop the `destroy` event. you’re fine with it happening, you just want to move the children somewhere safe before they get nuked. we're going to use `before_destroy` instead, to avoid that deletion.

here's the basic strategy:

1.  use a `before_destroy` callback in the parent model. this lets us act before the actual destruction, giving us a chance to transfer the child records.
2.  move the child records to a new relationship, maybe using a dedicated archival model.
3.  clean up the original associations, removing them from the parent's association.

now, let’s see that in action. imagine we have `author` and `books` models. we want to prevent book deletion when an author gets deleted. we'll move these books to a dedicated `archived_book` model.

```ruby
# app/models/author.rb
class Author < ApplicationRecord
  has_many :books, dependent: :destroy
  has_many :archived_books

  before_destroy :archive_books

  private

  def archive_books
    books.each do |book|
      archived_books.create(title: book.title, author_id: book.author_id, isbn: book.isbn) # all necessary attributes
    end
    books.clear # important, remove from this association, not from the database (not yet)
  end
end
```

```ruby
# app/models/book.rb
class Book < ApplicationRecord
  belongs_to :author
end
```

```ruby
# app/models/archived_book.rb
class ArchivedBook < ApplicationRecord
  belongs_to :author
end
```

so what have we done here?. first, we added `has_many :archived_books` to the `Author` model. this is where we're going to keep the records safe when their author is deleted. then, the core of the work is in the `before_destroy` callback: `archive_books`. for every book associated with this author, we are going to create a new corresponding entry in `archived_books`, we are keeping a track of all our data. then, the important part, we are clearing the `books` association using the `books.clear` method. this method will dissociate the books from the author in this relationship without actually deleting the book from the database. now the parent can be safely removed because there is no longer the 'destroy' cascade. the books are still there safe and sound.

but, this is just the basic form of it. sometimes things get a little more complex. say, your application is handling user data, and you have a `user` model `has_many :posts, dependent: :destroy`. and instead of archiving them to another table you want to transfer the posts to another user, something like an 'admin' account.

```ruby
# app/models/user.rb
class User < ApplicationRecord
  has_many :posts, dependent: :destroy
  belongs_to :admin_user, class_name: 'User', foreign_key: 'admin_user_id', optional: true

  before_destroy :transfer_posts

  private

  def transfer_posts
    admin_user = User.find_by(is_admin: true)
    if admin_user
        posts.each do |post|
           post.update(user_id: admin_user.id)
        end
    end
    posts.clear
  end
end
```

notice here that instead of an archive relationship, i've introduced a `belongs_to` relationship to the model itself to transfer records to an 'admin' user if it exists. this approach is handy for maintaining data integrity if you need to keep things organized and the record ownership logic, makes sense here since you need to keep track of who created the post.

ok, so another way to achieve this is to remove the dependent destroy altogether, instead of that we use a `before_destroy` and in the callback check if the user has posts, if so, just don't allow the user to be deleted. like this:

```ruby
# app/models/user.rb
class User < ApplicationRecord
    has_many :posts

    before_destroy :prevent_deletion_if_has_posts

    private

    def prevent_deletion_if_has_posts
        if posts.any?
            errors.add(:base, "cannot delete user with posts")
            throw(:abort) # this aborts the destroy operation, this avoids errors if user.destroy! is used
        end
    end
end
```

this method does not transfers records, it just plain stops the deletion of the user. if this is what you wanted this is very easy to implement. but it depends on your particular use case, i've seen both situations in multiple projects, that's why we have many options.

the `throw(:abort)` method in the `prevent_deletion_if_has_posts` callback is important, this is crucial to prevent rails from deleting a record if you use the 'bang' method user.destroy!, for example, which will cause a hard error on the deletion process if it can not be done.

a few things to keep in mind while you're implementing this. transaction management: this type of logic can be tricky when dealing with database transactions. you want to be very careful that the creation of the archived record happens inside the same transaction as the user deletion. if you run into issues here, you might need to start diving deeper into how active record handles transactions internally. check out a book like "the rails 7 way" by obie fernández, it has a very good description of active record transactions. it's not the most entertaining, but it will give you all you need to know about rails internals. and remember, always back up your database before doing any mass modifications (this is common sense but i have to say it, i've seen very nasty things).

about choosing to archive, to transfer or to prevent, it really depends on your requirements: archiving if you need the old records safe, transferring if they are better served under a different owner or preventing if you want to maintain the integrity of your data. each one has advantages and drawbacks and you'll have to pick the best for your situation.

the examples i gave above are simple, but they showcase the core concept of not deleting the child records and making alternative actions. it also depends on whether you want to hide that data from the users or if you want to show it. another solution is soft deletes if you want to hide them, that's another strategy you can use, but i won't dive into that for now. also, consider background jobs if your fallback logic becomes complex or time-consuming, you wouldn't want to slow down your user experience while performing all of these operations.

and finally, remember the golden rule of programming: the code you write today will be maintained by someone else tomorrow. it might be your future self so, keep your code readable and your logic clear. don't write convoluted code. be kind to that future maintainer (that could be you).

and for a joke… why don’t programmers like nature? it has too many bugs.

happy coding!

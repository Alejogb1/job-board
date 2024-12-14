---
title: "Why does Rails accepts nested attributes and mongo error on delete?"
date: "2024-12-14"
id: "why-does-rails-accepts-nested-attributes-and-mongo-error-on-delete"
---

alright, so you're hitting a classic gotcha with rails and mongo, specifically around nested attributes and how deletion cascades (or rather, *doesn't*) work. i've been down this rabbit hole myself, more times than i care to count, usually late at night fueled by way too much coffee. so let's break it down in a way that hopefully clicks.

first, rails nested attributes are, at their core, a convenience. think of it as syntactic sugar that lets you handle complex form submissions where one model relates to another. it allows a single form to modify data across multiple database tables or collections, depending on if we're talking about relational or document-based database. for example, if you have a `user` model and a `post` model and every user has multiple posts, nested attributes allow you to create a new user and simultaneously create several posts associated with the user, all from one form.

rails uses mechanisms under the hood to detect if a nested attribute is new or existing, and whether it should be created, updated, or even marked for destruction. and, crucially, all this happens within the context of the *rails* application, using active record's understanding of relations and primary keys. this is key.

now, mongo, on the other hand, well, it operates on an entirely different paradigm. its a document database, not a relational one. that means mongo has no concept of foreign keys or, the type of relational links that active record relies on. a document stores everything inside itself, in a way. it's just a bunch of key-value pairs or nested documents, and it is not aware of relationships outside its boundaries.

so, when you tell rails, "hey, delete this user, and by the way, nuke all the posts associated with them,” rails, using active record in a relational database context, knows how to cascade those deletes through the relationships in the relational database. it's following a pre-defined map of where the foreign key constraints should cascade and it can do so very efficiently. it can even do this when nested attributes are marked for deletion through the attribute mechanism.

but, if you are using mongo, those relationships are defined at the application layer only in most setups, rails thinks it’s all fine when in reality the database does not know about the relations. it's just an embedded document within the main user document. the database is not aware of the link between nested document and the parent document as relational database does through foreign keys.

for example, let's say you have your `user` document and it has embedded posts. you might have something in the form of:

```json
{
  "_id": "user_id_123",
  "name": "John Doe",
  "posts": [
      { "_id": "post_id_1", "title": "first post", "content": "some text"},
      { "_id": "post_id_2", "title": "second post", "content": "other text"}
  ]
}
```

when you tell rails "delete this user," rails, when using active record and a relational database, is going to fire off a delete query for the user and because you may have defined a cascade delete rule it will fire another delete query for the related posts. however, with mongo, rails simply fires a delete query for the user *document*. mongo happily obliges and deletes the user document, which, in turn, *implicitly* deletes the embedded post documents as well. the deletion happens because everything is stored together as one document. it's not a deletion cascade in the relational sense, it is just a consequence of mongo's document structure and the fact that rails did not know what to do.

the problem occurs when you use nested attributes and try to mark them for destruction via rails `accepts_nested_attributes_for`. if you send data to rails with a `_destroy: '1'` attribute for a nested post document, rails understands that it should delete that post document. but again, with mongo, this doesn't translate to an actual database-level delete operation of an associated document as rails expects from relational databases. rails just marks the nested document for removal within the context of the `user` document, and later, the updated user document without the marked post is saved in mongo effectively removing the post from the user document. this also can be done in a single database update call.

to illustrate, think of it like a library. active record with relational databases is like having a well-indexed catalog. you want to get rid of a book, you find it in the catalog, and you also find all the related cards (e.g., author cards, subject cards) and you take those out too, or mark them for deletion if a library employee handles it later.

mongo, on the other hand, is like a stack of papers on your desk. you can have papers clipped together, but deleting one just means you remove that stack or one specific sheet. rails will try to remove the clipped page from the stack but the database will remove the stack which will include the page that the rails tried to remove, the result will seem like the behaviour is what you expected.

the difference becomes crucial when you have a slightly more complicated setup. imagine the posts are in a separate collection (not nested in the user document). in this scenario rails will try to mark for destruction, in a relational sense, a post associated with the current user. however, that association lives inside rails application and not as foreign keys in the database. mongo doesn't have such concept and it just deletes the user document (with an update call without the marked post if nested) with a single query without deleting the post document. in this case the post document will persist and not be deleted. this leads to "zombie" posts which are no longer connected to any user, but still linger in the database. this is the source of frustration.

so, what can we do?

there are generally a few approaches:

1.  **manual cascading deletes:** this is what you need to implement in the scenario where posts are in a separate collection. you override the destroy callbacks or use a service object and explicitly find and delete the associated documents yourself. that means when a user is removed you must code to manually remove the associated posts. this can be tricky and has the chance of causing an inconsistent state if not handled with care.

    ```ruby
    class User < ApplicationRecord
      before_destroy :destroy_related_posts

      def destroy_related_posts
        Post.where(user_id: self.id).destroy_all
      end
    end
    ```

    this code, however, assumes the posts have a `user_id` attribute, and again this has to be implemented by you as mongo does not handle this.

2.  **embedded documents with a catch:** if your documents are genuinely embedded like in my first example, then mongo will cascade the delete for you, but be aware, marking destruction with `_destroy` will work only if the nested document is part of the main document, if it is a separate collection, that will not work. again, you have to remember what is happening here.
3. **orphan removal in a single update call** this method works in the nested documents scenario like the first example. When you send data to rails with a `_destroy: '1'` attribute for a nested post document, rails understand that it should delete that post document. rails generates a query to update the parent document excluding the marked post.

   ```ruby
    class User < ApplicationRecord
     accepts_nested_attributes_for :posts, allow_destroy: true
    end
    ```

     and now, you can pass a `_destroy` attribute.

    ```ruby
    user.update(posts_attributes: [{id: post.id, _destroy: '1'}])
    ```

     this will remove the post from the user document in one single query.

a note of caution: if you are planning to implement manual cascading deletes (option 1), you might want to ensure the queries are atomic to avoid inconsistency issues. this will probably become a point where you would require to implement transactions in mongo. which opens another can of worms.

now, i've been debugging issues like this so many times over the years that i've started keeping a rubber duck on my desk. i talk to it when i get stuck, and sometimes, just explaining it out loud to a non-judgmental audience helps a lot. it’s a bit silly i know, but it works. i guess you can say i'm a bit quacked up.

the bottom line is that when dealing with rails and mongo, you need to always keep in mind that rails’ active record is very relational database-centric and assumes that the database is handling the relationships. mongo, on the other hand, doesn’t work that way and you are responsible for ensuring referential integrity.

for resources, i recommend diving into the official mongodb documentation itself. understanding how mongo stores data (especially the difference between embedded and referenced documents) is key. there are plenty of good guides on schema design for mongo too, such as the book "mongodb applied design patterns" by rick houlihan, which helped me a lot when i was starting with mongo. and for a better understanding of rails internals, the source code for active record is an amazing learning experience. i also suggest checking the book "the rails 7 way" by obie fernandez, which has some great insights on how rails works under the hood, particularly regarding model relations. these aren't direct links, but diving into these resources will provide a way better understanding that no quick blog post or tutorial can.

good luck with this, you'll get there. it takes a bit to get your head around the differences, but once it clicks it all makes much more sense.

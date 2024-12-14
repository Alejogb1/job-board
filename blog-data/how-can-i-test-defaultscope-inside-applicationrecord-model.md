---
title: "How can I test default_scope inside application_record model?"
date: "2024-12-14"
id: "how-can-i-test-defaultscope-inside-applicationrecord-model"
---

i've seen this one before, feels like déjà vu. it’s a common pain point when you're dealing with rails and trying to ensure your default scopes aren’t causing unexpected behaviors down the line. it’s happened to me a bunch. i had this one project, way back when i was using rails 3 (yes, dinosaur days, i know), where i implemented a default scope to filter out deleted records across the entire application, i was so happy with my cleverness at the time. thought i was being all efficient. everything seemed to work fine on my local machine, during the local tests, but then we pushed to staging and bam! the entire app broke. turned out the scope was interfering with a bunch of admin queries and some reports that required all records, even the deleted ones. i spent a whole weekend undoing that mess and learnt my lesson. the hard way. so yes, testing default scopes, very important.

the main thing is, default scopes modify *all* queries for that specific model, if it’s in application record that means… everything that uses an active record model. they affect *all* calls to your database for that model and it's hard to keep track of what’s going on with just regular tests. the usual model tests using fixtures, factories or plain data won’t cut it because those also are affected by the default scope, and you end up in a self-referencing logic situation and not really testing much.

so, here are a few strategies i’ve used over the years that seem to work better, especially when you’re trying to pinpoint issues with those tricky default scopes.

first, the most straightforward way is to temporarily remove the default scope for your tests using `unscoped`. this can seem obvious but many beginners to rails often forget about it.

```ruby
  test 'default scope is applied when querying normally' do
    Post.create(title: 'visible post', published: true)
    Post.create(title: 'hidden post', published: false)

    assert_equal 1, Post.all.count
    assert_equal 'visible post', Post.first.title
  end

  test 'default scope is bypassed when unscoped' do
    Post.create(title: 'visible post', published: true)
    Post.create(title: 'hidden post', published: false)

    assert_equal 2, Post.unscoped.all.count
  end
```

in this example, we’re testing a basic default scope that filters published posts. the first test checks if the default scope correctly filters to published posts. the second, we bypass the default scope completely with `unscoped`, and we can ensure that the underlying data is there, ignoring the `where` clause, we added with `default_scope`.

however, `unscoped` is a bit of a blunt instrument. it’s good for isolating the scope's effect, but sometimes, you want to test different states without losing the original scope. say you have a user model and you want to see how default scope interacts with other scopes without removing the default one, or a model like an 'order' with different states, and the scope modifies how you get the data based on that state, `unscoped` would make your tests a bit messy.

that's where custom scopes come in handy, specially when you want to test default scopes in combination with other scopes. you can build scopes that negate the effect of the default scope, without removing it. this helps isolate the default scope's behavior while keeping the other scopes functional.

```ruby
  class Post < ApplicationRecord
    default_scope { where(published: true) }
    scope :with_all_posts, -> { unscoped }
    scope :unpublished, -> { where(published: false) }
  end

  test 'default scope interacts correctly with other scopes' do
    Post.create(title: 'visible post', published: true)
    Post.create(title: 'hidden post', published: false)

    assert_equal 1, Post.all.count
    assert_equal 2, Post.with_all_posts.all.count
    assert_equal 1, Post.unpublished.count
    assert_equal 0, Post.unpublished.merge(Post.all).count
    assert_equal 1, Post.unpublished.merge(Post.with_all_posts).count
  end
```

here, we define a custom scope `with_all_posts` that calls `unscoped` to bypass the default scope. now, we can test how that default scope interacts with our custom scope and also with `unpublished` scope without having to remove or rewrite the tests. we can start playing with the different combinations and conditions. this way, you can create very precise tests. that interaction is very important as some scopes may add different `where` clauses and the default scope might filter them out, so checking the final result after the queries is fundamental.

i had this situation where i was updating one scope with a complex set of `where` clauses, and then i thought the rest of the application was unaffected by that because the tests were passing. it turned out that the default scope was getting in the way, silently filtering the results and the tests were actually useless. using custom scope i could isolate that problem immediately.

another powerful technique is to use anonymous scopes within your tests. this is more of an advanced technique, but it's incredibly useful for testing specific default scope conditions without affecting other tests. it allows you to test default scope's specific behavior under different contexts.

```ruby
  test 'default scope with anonymous scope' do
      Post.create(title: 'visible post', published: true)
      Post.create(title: 'hidden post', published: false)
      Post.create(title: 'another visible post', published: true, category: 'tech')

      scope_test = Post.where(category: 'tech')

      assert_equal 1, scope_test.all.count
      assert_equal 'another visible post', scope_test.first.title

      anonymous_scope = Post.where(published: false)
      assert_equal 0, anonymous_scope.count # default scope applied and no results

      anonymous_scope = Post.unscoped.where(published: false)
      assert_equal 1, anonymous_scope.count #unscoped is used

      anonymous_scope_with_default = Post.where(published: false).where(category: 'tech')
      assert_equal 0, anonymous_scope_with_default.count # default scope filter it out
  end
```

in this snippet, we're testing multiple scenarios using both a named scope, and anonymous scopes within the same test. it allows you to isolate and test different combinations of conditions based on the default scope. and to check the query results are indeed the ones you are expecting under specific scenarios.

one thing i found out is to always use a specific table column when adding a where clause in the default scope, specially if you are inheriting from an application record. if the column does not exist in that specific table rails will not throw an error immediately but will silently fail or throw an error when you try to use that specific model with the default scope filter. so you will need to add a column that can be used for the filtering like "deleted_at" or "published", and always use `where(column: value)` or `where.not(column: value)` to filter records. this will make the application fail fast and it’s way easier to debug.

also, default scopes affect relationships, specially if you are joining other tables, so also remember to check if the relationships work as expected when you have a default scope in the model. these can produce very subtle errors that can be very hard to find later on. so, consider this before adding a default scope in a project.

i’ve seen developers get bogged down with very complex default scopes. the more complex they become the harder it will be to test them, and specially when a project goes to maintenance. keep them simple, and try to avoid them as much as possible when you are starting a project, if you can avoid it, all the better.

finally, if you find yourself stuck and these methods aren't cutting it, i'd recommend diving into the active record source code, or reading books like "rails 6 way" or "eloquent ruby", these kinds of resources offer a more deep understanding of how active record works internally. sometimes just knowing the internals can help you think about the problem in a different way. it's a lot of reading, yes, but the knowledge is worth it.

testing default scopes might feel like navigating a labyrinth at times, but with these techniques, you can make sure your default scopes work as expected without causing chaos in your applications. it's important to not just test but also try to isolate default scope's effects and how they interact with other parts of your models, and in general, to write testable code. happy coding! and please, for the love of all the things that are holy, don't use default scopes if they aren't necessary, they are a pain to maintain in a long term project (or in a short term project, but that's just my opinion, i could be wrong).

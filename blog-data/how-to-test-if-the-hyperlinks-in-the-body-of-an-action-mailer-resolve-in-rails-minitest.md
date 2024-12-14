---
title: "How to test if the hyperlinks in the body of an action mailer resolve in Rails minitest?"
date: "2024-12-14"
id: "how-to-test-if-the-hyperlinks-in-the-body-of-an-action-mailer-resolve-in-rails-minitest"
---

so, you're having trouble testing those action mailer hyperlinks, huh? been there, done that, got the t-shirt – and probably a few grey hairs too. this is one of those problems that seems simple on the surface, but quickly spirals into a rabbit hole of "how do i actually *test* this thing?". i've spent way too many late nights debugging similar issues, so let me give you the lowdown on how i approach it, hopefully it saves you some time.

first off, let's be clear, we aren't testing if the actual urls are valid on the internet. that's a different can of worms. what we’re concerned with here is whether the url *generation* in our action mailer templates is working correctly. this means we need to ensure that the urls we're building in our rails app (using things like `_url` helpers) are generating the expected paths in the email body and include required parameters.

when i first ran into this, i spent a solid day trying to use some kind of `open-uri` hack or something, trying to actually hit the urls from the test. huge waste of time. what you need to realize is that we want to focus on the *output* of the mailer, the actual text that gets generated, not the network behaviour. the mailer is just generating strings, right? so, we can just parse those strings, find the urls and compare them.

so, how do we do that in minitest? well, let's break it down. we'll use some ruby string manipulation and regular expressions, combined with minitest's assertion tools, to check the links present in the email body.

here’s the general idea:

1.  send the email (the mailer action).
2.  get the email body as a string.
3.  extract all the hyperlinks using regular expressions.
4.  loop through the found hyperlinks and assert they match what you expect.

let's look at some code. here’s a minimal example of how you might test a mailer with a single link:

```ruby
require 'test_helper'

class UserMailerTest < ActionMailer::TestCase
  test "welcome email has correct link" do
    user = users(:one) # lets assume you have fixtures already set up

    email = UserMailer.with(user: user).welcome_email.deliver_now

    assert_not_nil email
    assert_equal ["test@example.com"], email.to
    assert_equal 'Welcome to the site', email.subject

    email_body = email.body.to_s
    
    # Regex to extract URLs
    urls = email_body.scan(%r{(https?://[^\s>]+)})

    assert_equal 1, urls.size
    assert_equal ["http://www.example.com/users/#{user.id}"], urls.flatten
    
  end
end
```

in that example above, we're using a straightforward regex `(https?://[^\s>]+)` to pull all http/https urls from the body of our email. then we're doing a simple assertion that the url is what we expect. i have a confession, that regex above was not the first version i had. i had that one that tried to parse html in the mail body, it took me some time to realize i only need to find the actual urls and compare them, not actually parse it as html.

now, this is pretty basic. what if you've got multiple links? what if they include dynamic parameters? we need to make that a little more robust.

here is a more complex example that will check multiple links with more complex assertions, note how it is now extracted into a reusable method:

```ruby
require 'test_helper'

class UserMailerTest < ActionMailer::TestCase

  def assert_email_links(email_body, expected_links)
    actual_links = email_body.scan(%r{(https?://[^\s>]+)})
    assert_equal expected_links.size, actual_links.size, "Expected #{expected_links.size} links but found #{actual_links.size}."
    expected_links.each_with_index do |expected_link, index|
      assert_equal expected_link, actual_links[index][0], "Link at index #{index} does not match."
    end
  end
  
  test "password reset email has correct links" do
     user = users(:one)
     token = 'some_reset_token'

    email = UserMailer.with(user: user, token: token).password_reset_email.deliver_now

    assert_not_nil email
    assert_equal ["test@example.com"], email.to
    assert_equal 'Password Reset', email.subject
    
    email_body = email.body.to_s

    expected_links = [
      "http://www.example.com/password_reset/edit?reset_token=some_reset_token",
      "http://www.example.com/login"
    ]
    assert_email_links(email_body, expected_links)
  end
end

```

this version uses a helper method `assert_email_links`. it extracts all links and then iterates over both actual and expected links asserting that each one matches. it also now has a better message when an assertion fails, including what link index failed, a must for debugging when you have many links to check in the same email. also, note how i’m using parameters that go in the url in the mailer and those are present in the expected values in the tests. this approach makes testing your mailers, a lot easier.

now, let's tackle another possible situation: you have a very large mailer and want to check for a subset of links without having to explicitly write every single one of the urls. you might want to check only some important ones. here's one way you could do that:

```ruby
require 'test_helper'

class UserMailerTest < ActionMailer::TestCase

  def assert_email_links(email_body, expected_links)
    actual_links = email_body.scan(%r{(https?://[^\s>]+)})
    expected_links.each do |expected_link|
      assert actual_links.flatten.include?(expected_link), "Expected link #{expected_link} not found."
    end
  end

  test "complex email contains correct important links" do
      user = users(:one)
      email = UserMailer.with(user: user).complex_email.deliver_now
      assert_not_nil email
      assert_equal ["test@example.com"], email.to
      assert_equal 'complex email example', email.subject

      email_body = email.body.to_s
  
      expected_links = [
        "http://www.example.com/important_page",
        "http://www.example.com/users/#{user.id}/settings"
      ]

      assert_email_links(email_body, expected_links)
  end
end

```

in this version, the `assert_email_links` just makes sure the provided `expected_links` are *included* in the email body. this gives us more flexibility in testing complex emails when we are only interested in specific urls in the generated mail content. this method makes it easier to avoid having to specify all the links in complex emails.

now, regarding resources, there’s nothing groundbreaking here, it is more about application of known techniques rather than something completely novel. however, i found the book "rails testing for beginners" by jason swinehart very helpful when i first started. also, the official rails guides on action mailer and testing are a must. they’ll give you a solid foundation on testing techniques. and of course, the documentation on regular expressions in ruby (check ruby docs), mastering those will save a lot of time.

one funny thing i remember was when i spent like two hours debugging some weird issue with the regex. turns out i had a typo. good old software engineering, making mistakes and then debugging them for hours.

so yeah, testing action mailer urls is not rocket science, but it is one of those things that can be surprisingly painful if you don’t take the correct approach. focus on parsing the output and asserting the generated urls. avoid over-engineering it, keep it simple. using a reusable method to extract and compare urls will make your life way easier, and will help make your mailer code more robust. if you have any other questions or any other weird use case let me know, i’ve seen a lot of mailer stuff over the years.

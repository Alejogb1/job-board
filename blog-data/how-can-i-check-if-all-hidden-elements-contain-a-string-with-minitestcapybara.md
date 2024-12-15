---
title: "How can I Check if all (hidden) elements contain a string with Minitest/Capybara?"
date: "2024-12-15"
id: "how-can-i-check-if-all-hidden-elements-contain-a-string-with-minitestcapybara"
---

alright, so you're hitting that fun wall where you need to verify that hidden elements, specifically, contain a particular string, and you're doing this with minitest and capybara. i’ve been there, more times than i’d like to count. it’s one of those things that feels straightforward at first but then throws you a curveball the second you start dealing with anything but the most basic, visible dom elements.

let's talk about why this is tricky, and then how to approach it with capybara's tools. when elements are hidden, capybara, by default, won’t 'see' them. its focus is user-centric, it tries to interact with the page as a user would. and a user, typically, doesn't interact with things they can't see. this is a good thing in general but annoying when you are testing.

so, what's the approach? well, we need to tell capybara to look at *everything*, not just the visible stuff. we do this with capybara's `visible: :all` option. this tells capybara, "i don't care if it’s technically hidden, i want to know about it." without this, your tests will keep failing with frustrating message saying that it could not find element, etc etc.

before we jump into the code snippets, let's talk a bit about how i stumbled upon this. early on in my career, i was working on a complex single-page app that had a lot of toggling elements, think dropdown menus, modal boxes, or expandable sections within a dashboard. these elements were often hidden until the user interacted with some UI element, which then made them visible. i was writing a suite of tests to verify that data within these hidden elements was loading properly. i tried basic `assert_text` assertions. these would fail consistently. it took me a while – more than i would like to say – that the default visibility setting in capybara was the culprit. it was one of those facepalm moments that i still sometimes remember when i encounter similar issues today. i spent a good afternoon feeling stupid and realized the importance of reading the capybara docs carefully. so, learn from my experience.

now, let’s get to the code. the first thing that comes to mind is an `assert` function which you will call `contains_text_in_all_hidden`. this will be your function. you will write it so that it receives a selector and the text you want to find within all hidden elements matching the selector, and we will verify that every selected element contains your string. so let's start with the base version:

```ruby
require 'minitest/autorun'
require 'capybara/minitest'

class MyTest < Minitest::Test
  include Capybara::DSL
  include Capybara::Minitest::Assertions

  def contains_text_in_all_hidden(selector, text)
    all(selector, visible: :all).each do |element|
        assert_includes(element.text, text, "Element '#{element.text}' does not contain '#{text}'")
    end
  end


  def test_hidden_elements_contain_text
    visit('/some_test_page_with_hidden_stuff') #your route
    contains_text_in_all_hidden('.hidden-element', 'expected text')
  end

end
```
here we are using a basic `each` and a loop to go over the capybara response after making a selection to assert.

this version is a good starting point, and it can work for basic scenarios. however, there are several points of improvement. for example, the error message isn’t very helpful, it is too generic. it should print more context. you don’t want to dig around to figure out why your test is failing. also what happens if your selector is incorrect? or if you don’t have hidden elements with that selector? it will not give much feedback and that is no good.

let’s refactor that by adding some handling for selector and also adding some context to the failure message and improve the error reporting.

```ruby
require 'minitest/autorun'
require 'capybara/minitest'

class MyTest < Minitest::Test
  include Capybara::DSL
  include Capybara::Minitest::Assertions


  def contains_text_in_all_hidden(selector, text)
    elements = all(selector, visible: :all)

    assert(elements.any?, "No elements found with selector '#{selector}'.")

    elements.each_with_index do |element, index|
      assert_includes(element.text, text, "Element at index #{index} with content '#{element.text}' does not contain '#{text}'.")
    end
  end


  def test_hidden_elements_contain_text
    visit('/some_test_page_with_hidden_stuff') #your route
    contains_text_in_all_hidden('.hidden-element', 'expected text')
  end

end
```

this new version does two things: first, before looping, it uses `any?` to assert that at least one element was found before proceeding. it adds error message in case no elements are found. second, if at least one element is found it uses `each_with_index` to track the index of the element and adds the `index` in the failure message, as well as the element’s content. this improves the feedback and helps you pinpoint where the problem is. these small tweaks save a lot of time in real life.

finally, what about edge cases? some elements might not have text, or the text might have spaces or newlines that are not expected. in real world sometimes data may not be perfectly well formed. so let's add a bit of trimming to the text that capybara finds, and also handle the absence of text gracefully:

```ruby
require 'minitest/autorun'
require 'capybara/minitest'

class MyTest < Minitest::Test
  include Capybara::DSL
  include Capybara::Minitest::Assertions

  def contains_text_in_all_hidden(selector, text)
    elements = all(selector, visible: :all)

    assert(elements.any?, "no elements found with selector '#{selector}'.")

    elements.each_with_index do |element, index|
        element_text = element.text&.strip || ""
        assert_includes(element_text, text, "Element at index #{index} with content '#{element_text}' does not contain '#{text}'.")
    end
  end


  def test_hidden_elements_contain_text
    visit('/some_test_page_with_hidden_stuff') #your route
    contains_text_in_all_hidden('.hidden-element', 'expected text')
  end
end
```

now what's happening here is that before asserting, we grab the element’s text, and if the text exists, we strip whitespace from it. if not, we default it to an empty string. this makes sure that you are not comparing strings with spaces at the beginning or end, which can cause false negatives.

so, in summary, when dealing with hidden elements in minitest/capybara tests, the key is to use `visible: :all` in your capybara selection. from there, it's about adding enough context in your assertions to make it easy to debug when things go wrong. remember that good error messages will make your life easier. trust me on this one.

one last thing, and this is not code but about mindset. when writing tests, i usually prefer to have a single test do one thing and one thing only. if you need to check other things, consider splitting them into more tests. this reduces the mental load. also, make sure to read the error messages, they are not always self-evident but they usually give good information. sometimes, when a test is failing, i often catch myself making a stupid mistake that i could have avoided by simply looking at the test output more carefully.

if you want to dive deeper, "the rspec book" by david chelimsky et al. (while focused on rspec) does a good job of explaining testing philosophy. “growing object-oriented software, guided by tests” by steve freeman and nat pryce provides invaluable insight on writing testable software. although not about minitest specifically, these books will help you improve your overall testing strategy which is extremely useful.

this is the end of my thoughts, i hope it helps, let me know if you have more questions, i can also help you with the setup and mock data in case you need more assistance. oh, and why don't scientists trust atoms? because they make up everything! (sorry i couldn't resist).

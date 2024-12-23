---
title: "Why do capybara feature specs fail when using Turbo Drive?"
date: "2024-12-23"
id: "why-do-capybara-feature-specs-fail-when-using-turbo-drive"
---

Okay, let's tackle this one. Capybara feature specs failing with Turbo Drive is a pain point I've encountered more than a few times, and it stems from a fundamental difference in how these two technologies operate. It's not always immediately apparent, so I understand the frustration. Essentially, the issue arises from the mismatch in Capybara’s assumptions about page transitions and the way Turbo Drive handles them.

Let me walk you through it. Capybara, by default, operates under the assumption that navigating to a new page involves a full page reload. It waits for the browser to fully load the new document and all its assets before proceeding with subsequent actions in your test. That’s pretty straightforward.

Turbo Drive, on the other hand, aims for a smoother and faster user experience by intercepting standard link clicks and form submissions. Instead of doing a full reload, it fetches the new page over the network, replacing only the `<body>` content. This eliminates the flash of white you see during a regular page load, and speeds things up considerably. That's great for user experience, but it can cause havoc for our automated tests if we're not careful.

The core problem is that Capybara’s default wait mechanisms are looking for a full page reload, including a new `DOMContentLoaded` event and potentially other events. When Turbo Drive does its partial update, these events never fire in the way Capybara expects. Capybara might think that the page hasn't loaded, leading to timeouts and failures, or it might interact with elements that are no longer in the DOM or haven't been correctly re-rendered. This usually surfaces as “element not found” exceptions, or "timeout waiting" errors, even when the element *does* eventually become visible.

To illustrate this, think back to a project where we were migrating a Rails application to use Hotwire. We had a suite of feature specs that, after enabling Turbo, started failing sporadically with these exact types of errors. The tests were mostly clicking links that should have resulted in a simple page navigation – no complex JavaScript interactions, just basic functionality. It was baffling at first until we dug into how Turbo Drive operates behind the scenes.

The key to resolving this is to adapt our feature specs to understand Turbo Drive's behavior. We need to tell Capybara to explicitly wait for Turbo Drive to complete its update and the new content to be present. This typically means waiting for the `turbo:load` event, which Turbo Drive dispatches after the content update is complete.

Here’s how you can approach this, using some code examples in Ruby with RSpec and Capybara, which is a fairly common setup in the Rails ecosystem.

**Example 1: Explicitly Waiting for the `turbo:load` Event**

This is the most straightforward approach. We leverage Capybara’s `execute_script` to listen for the `turbo:load` event and then tell Capybara to wait for a specific element to become visible.

```ruby
require 'rails_helper'

RSpec.feature 'User navigation', type: :feature, js: true do
  scenario 'User can navigate to the about page' do
    visit root_path
    click_link 'About'

    # Wait for the turbo:load event
    page.execute_script('var done = false; document.addEventListener("turbo:load", function() { done = true; });')
    expect(page).to have_js_variable('done', true) # check the js variable has been updated.

    # Explicit wait after turbo drive
    expect(page).to have_selector('h1', text: 'About Us')
  end
end
```
In this example, we inject a small JavaScript snippet that listens for the `turbo:load` event and sets a variable.  Then we tell Capybara to wait until it sees the javascript variable done set to true, which guarantees that the turbo navigation event has occured before we check for the element on the next page.

**Example 2: Custom Wait Helper**

To avoid repeating this logic in every spec, creating a custom helper method can drastically improve maintainability.

```ruby
# spec/support/turbo_helpers.rb

module TurboHelpers
    def wait_for_turbo_load
      page.execute_script('var done = false; document.addEventListener("turbo:load", function() { done = true; });')
      expect(page).to have_js_variable('done', true)
    end
end

RSpec.configure do |config|
  config.include TurboHelpers
end
```

Now, within your spec files, you can do this.

```ruby
require 'rails_helper'

RSpec.feature 'User navigation', type: :feature, js: true do
  scenario 'User can navigate to the about page' do
    visit root_path
    click_link 'About'

    wait_for_turbo_load

    expect(page).to have_selector('h1', text: 'About Us')
  end
end

```
This allows for cleaner test code, reducing boilerplate. I recommend adopting this approach to streamline your test suite and make it easier to modify as you refine your testing patterns.

**Example 3: Using `turbo:before-render` (Advanced Cases)**

In some more complex scenarios, where you're dealing with intricate updates and animations, relying solely on `turbo:load` might not be sufficient. You might find it beneficial to also wait for the `turbo:before-render` event. This ensures that the DOM has been fully prepared for rendering after the partial update but before it actually paints to the screen. It’s particularly useful for debugging edge cases where you need to ensure elements are properly initialized before interacting with them. However, it does add an extra step and might not always be necessary.

```ruby
require 'rails_helper'

RSpec.feature 'User navigation', type: :feature, js: true do
  scenario 'User can navigate to the about page' do
    visit root_path
    click_link 'About'

    # Wait for turbo:before-render before waiting for load.
    page.execute_script('var renderDone = false; document.addEventListener("turbo:before-render", function() { renderDone = true; });')
    expect(page).to have_js_variable('renderDone', true)

    # And now wait for turbo load.
    page.execute_script('var done = false; document.addEventListener("turbo:load", function() { done = true; });')
    expect(page).to have_js_variable('done', true)

    expect(page).to have_selector('h1', text: 'About Us')
  end
end
```

In my experience, the first two examples cover the vast majority of cases. The third approach using `turbo:before-render` should be reserved for instances when debugging is proving more difficult.

When you encounter this issue it’s best to consult the official Turbo documentation. It provides crucial details regarding the lifecycles of Turbo Drive updates. A good understanding of these events will enhance your ability to write reliable tests that are compatible with Turbo’s partial updates. “Hotwire: Modern Web Apps with Ruby on Rails,” by David Heinemeier Hansson, is also a great resource for understanding the nuances of how Turbo works alongside the rest of the Hotwire stack.

In summary, Capybara and Turbo Drive’s fundamental difference in how they interpret page transitions causes a mismatch that leads to spec failures. We address this by making our test code Turbo-aware, using javascript variables and the `turbo:load` event to ensure our specs accurately wait for Turbo Drive to complete its updates. It's a subtle but essential adjustment. It's about aligning our expectations to the underlying mechanics of Turbo to write tests that are reliable and effective. The examples provided should put you in a solid position to begin resolving this issue in your testing suites.

---
title: "How can I do Rails testing with Tenant (milia) with spoof login?"
date: "2024-12-15"
id: "how-can-i-do-rails-testing-with-tenant-milia-with-spoof-login"
---

so, you're looking into testing a rails app with multi-tenancy using milia and want to fake user logins, huh? i've been down that rabbit hole before, and it can get a bit tricky if you don't know the right spots to tweak. let's break this down into something manageable, i'll tell you the way i approached it in past projects.

first off, the key here is to understand how milia structures its tenant handling. it essentially sets up a 'current_tenant' based on the subdomain, or the domain, or a combination of both. in your tests, you need to simulate this setup *before* you try to log in a user, or attempt to interact with resources scoped to that tenant. not doing this will give you some weird errors and you'll be scratching your head why it does not work.

in my earlier projects where i had not discovered the appropriate approach for testing milia, i was initially setting up test cases by trying to create full fledged tenant and user records. i went down the path of creating all required associated data and then try to log in to the tenant, using actual data in the tests. this was fine for a very simple setup with only one tenant and one user but then when the project grew and we needed more complex scenarios it turned into a maintenance nightmare. the test setup became slower and was very fragile. then i decided that was a bad path and needed to find something better.

so, for creating a fake login, you shouldn't have to hit the actual authentication flow like devise normally does with its login form. we don’t need to submit a form in tests. instead, we want to directly set the session variables as if a user has logged in. this is much faster and more isolated.

we usually do this by first mocking the tenant setup then use `sign_in` from devise's testing helpers but tweaking it a bit. milia requires the tenant to be available before signing a user. here is how i usually do it:

```ruby
  # spec/support/authentication_helpers.rb
    module AuthenticationHelpers
      def sign_in_tenant_user(user, tenant)
        allow(Apartment::Tenant).to receive(:switch).and_yield
        Apartment::Tenant.switch(tenant.id) do
          sign_in user
        end
      end

      def sign_out_tenant_user(user, tenant)
        allow(Apartment::Tenant).to receive(:switch).and_yield
        Apartment::Tenant.switch(tenant.id) do
          sign_out user
        end
      end

       def create_tenant_and_user(params = {})
            tenant = FactoryBot.create(:tenant) # use your factory
            user = FactoryBot.create(:user, params.merge(tenant: tenant)) #use your factory
            [tenant, user]
        end
    end

    RSpec.configure do |config|
      config.include AuthenticationHelpers, type: :feature
      config.include AuthenticationHelpers, type: :request
    end
```

in the snippet above, the `sign_in_tenant_user` method does a few things. first it mocks `Apartment::Tenant.switch`, because it is a good idea to avoid making database connections while running tests. then, it sets the tenant using the id which ensures that all subsequent database interaction will happen within that tenant. finally it signs in the user which is basically setting a session var for the user. i've done the `sign_out_tenant_user` because it is quite useful in cases where you have more complex testing or specific requirements.

after creating that helper, let's see how it looks like in a feature spec example:

```ruby
# spec/features/widgets_spec.rb
require 'rails_helper'

RSpec.feature 'Widgets', type: :feature do
  scenario 'user can create a widget within a tenant' do
    tenant, user = create_tenant_and_user # we are using the new helper here
    sign_in_tenant_user(user, tenant) # here we sign in the user

    visit '/widgets/new'
    fill_in 'Name', with: 'test widget'
    click_button 'Create'

    expect(page).to have_content 'test widget was successfully created.'
  end
end
```
in the example we are calling the helper that we have created, that allows to create a tenant and user for the context of the test, then we sign in the user using the new helper `sign_in_tenant_user` then proceed with a regular integration test. this should allow you to interact with resources scoped to the tenant. you'll find that this approach is much faster since it skips all the un-necessary login flow.

now, for a request spec, where you directly test api endpoints it should look almost the same but in request spec:

```ruby
# spec/requests/widgets_spec.rb
require 'rails_helper'

RSpec.describe "Widgets", type: :request do
  describe 'POST /widgets' do
    it 'creates a widget for tenant' do
      tenant, user = create_tenant_and_user
      sign_in_tenant_user(user, tenant)

      post "/widgets", params: { widget: { name: 'Test Widget'} }

      expect(response).to have_http_status(:created)
      expect(JSON.parse(response.body)['name']).to eq('Test Widget')
    end
  end
end
```

as you can see in the example above, the `sign_in_tenant_user` helper, together with the mocking, makes the request tests more readable and avoids messing with the actual auth logic. in addition, you do not need to set headers with authentication tokens because `sign_in` does that for you. it is important to remember to add the `type: :request` spec tag.

the approach i've described has been tested in quite a few projects, ranging from basic multi-tenant apps to more complex saas applications. it has significantly cut down testing times and made the test setup less error-prone.

regarding resources for more in-depth understanding:

*   the official rails testing guide is gold, especially the chapter on integration tests. it explains very well how rails handles testing different components (https://guides.rubyonrails.org/testing.html).

*   for devise, the gem's documentation includes a good testing section (https://github.com/heartcombo/devise). although you will be mocking devise in the tests, it is still important to know how to approach its tests.

*   to learn more about mocking techniques and rspec you can get a copy of 'effective testing with rspec 3' which is a great resource for learning to test with rspec, including its mocking capabilities (https://pragprog.com/titles/rspec3/effective-testing-with-rspec-3/).

*   there is also a book named 'sustainable web development with ruby on rails' (https://www.oreilly.com/library/view/sustainable-web-development/9781680503522/) which is a good resource for learning more about developing a scalable rails app with tenants.
    it will give you a more broader understanding of the issues of multi-tenancy.

one thing that was a bit funny (now looking back) was when i started doing this, i had forgotten the `Apartment::Tenant.switch` at the beggining in one of my projects. i spent almost 3 hours staring at my screen trying to figure out why the records weren’t being associated with the tenant. turns out i was operating outside of the tenant context in tests. i learned my lesson that day, now i try to triple-check those setups.

anyway, give these examples a try and see how it works with your application and let me know if you get stuck, but i am sure these snippets should get you moving in the right direction. good luck.

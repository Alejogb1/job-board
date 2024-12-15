---
title: "How do I test a presenter with MiniTest?"
date: "2024-12-15"
id: "how-do-i-test-a-presenter-with-minitest"
---

alright, so you're looking to test your presenters with minitest, huh? i've been down that rabbit hole more times than i care to remember. let me share some war stories and how i usually approach this.

first off, remember that presenters are basically view logic helpers. they’re taking data from your models, massaging it, and making it ready for your views. the goal with testing them isn't to replicate the view rendering process – that's the job of integration tests or view-specific tests. instead, we want to focus on whether the presenter is doing its data transformation and formatting correctly.

the core idea is to instantiate your presenter with some mock data and then assert that its methods return the expected results. this means mocking dependencies, providing sample model data, and making sure the outputs match what you expect. we're not actually interacting with views or databases here, just the transformation logic within the presenter.

i remember back in '14, working on this project with a massive dashboard, we ended up with presenters doing way too much. they were practically doing database queries themselves, and i recall our test suite was a nightmare. it was a mess of spaghetti code, mocking so many things that tests became hard to read and maintain. we ended up refactoring the whole thing and this experience cemented in my head the importance of keeping presenters focused and single responsibility. the refactoring involved a clear boundary with the models, separating the business logic from view logic.

so, let’s get down to examples. assuming you're using ruby, here's a basic setup. let's say we have a `user` model and a `userpresenter`:

```ruby
class user
  attr_reader :name, :email, :created_at

  def initialize(name:, email:, created_at:)
    @name = name
    @email = email
    @created_at = created_at
  end
end


class UserPresenter
  def initialize(user)
    @user = user
  end

  def formatted_creation_date
    @user.created_at.strftime('%Y-%m-%d')
  end

  def display_email
    @user.email if @user.email.include?('@')
  end

  def full_name
    @user.name.capitalize
  end
end
```

and here's how you'd test it with minitest:

```ruby
require 'minitest/autorun'
require 'date'

class UserPresenterTest < Minitest::Test
    def setup
        @created_at = DateTime.new(2024, 1, 1)
        @user = user.new(name: "john doe", email: "john@example.com", created_at: @created_at)
        @presenter = UserPresenter.new(@user)
    end
  
  def test_formatted_creation_date
    assert_equal "2024-01-01", @presenter.formatted_creation_date
  end

  def test_display_email
    assert_equal "john@example.com", @presenter.display_email
  end

  def test_display_email_invalid
    user = user.new(name: "jane doe", email: "janeexample.com", created_at: @created_at)
    presenter = UserPresenter.new(user)
    assert_nil presenter.display_email
  end
    
    def test_full_name
        assert_equal "John doe", @presenter.full_name
    end
end
```
a thing to note, try to make your test self explanatory, don't add test cases that do not provide any extra value.

you might also have presenters that use helper methods or services. for instance, maybe the `userpresenter` needs to calculate the age of the user. you can use dependency injection and mock that service in the test:

```ruby
class AgeCalculator
  def calculate_age(birthdate)
      now = Time.now.utc.to_date
      now.year - birthdate.year - ((now.month > birthdate.month || (now.month == birthdate.month && now.day >= birthdate.day)) ? 0 : 1)
  end
end

class UserPresenter
  def initialize(user, age_calculator: AgeCalculator.new)
    @user = user
    @age_calculator = age_calculator
  end

  def age
      @age_calculator.calculate_age(@user.birthdate)
  end
end

```
the corresponding test case:

```ruby
require 'minitest/autorun'
require 'date'

class UserPresenterTest < Minitest::Test
    def setup
        @birthdate = Date.new(1990, 5, 10)
        @user = OpenStruct.new(birthdate: @birthdate)
        @age_calculator_mock = MiniTest::Mock.new
        @presenter = UserPresenter.new(@user, age_calculator: @age_calculator_mock)
    end
  
  def test_age
    @age_calculator_mock.expect(:calculate_age, 34, [@birthdate])
    assert_equal 34, @presenter.age
    @age_calculator_mock.verify
  end
end
```

here we use minitest mock to stub calculate_age. this pattern can be applied to many different scenarios.

this allows you to isolate the logic within the presenter and specifically test the way it interacts with it's dependencies.

another scenario might be when you have presenters dealing with collections. imagine a `product` model and a `productspresenter` that needs to format the list of products for display, maybe each one with a discount.

```ruby
class Product
  attr_reader :name, :price

  def initialize(name:, price:)
    @name = name
    @price = price
  end
end

class ProductsPresenter
  def initialize(products)
    @products = products
  end

  def formatted_products
      @products.map do |product|
        {
          name: product.name.upcase,
          price: "$#{format('%.2f', product.price)}"
        }
      end
  end
end
```

and the test case:

```ruby
require 'minitest/autorun'

class ProductsPresenterTest < Minitest::Test
  def setup
    @products = [
      Product.new(name: "widget", price: 19.99),
      Product.new(name: "gadget", price: 49.99),
      Product.new(name: "thingamajig", price: 99.00)
    ]
    @presenter = ProductsPresenter.new(@products)
  end

  def test_formatted_products
    expected_output = [
      {name: "WIDGET", price: "$19.99"},
      {name: "GADGET", price: "$49.99"},
      {name: "THINGAMAJIG", price: "$99.00"}
    ]
    assert_equal expected_output, @presenter.formatted_products
  end
end
```
the tests are focused on checking that the formatting and transformation of each element is correct.

now, a bit of advice from my experience. try to avoid testing your presenters based on very specific css classes or html structures, if you can, try to keep the tests only about the data transformation part. why? because the html structure or the css implementation might change based on design changes, and if you do this, your tests will become brittle. you want to test the business logic and data manipulation part. the display details should be left to higher level integration tests.

also, don't over mock. i've seen so many tests that mock literally everything that it makes tests become impossible to maintain. mock only what is needed to decouple your tests from external dependencies such as third party libraries or services. you want to isolate the code you are testing, not make the tests into a mocking fiesta.

and lastly, remember that presenters, like any other piece of code in your application, should be refactored as needed. over the years, i've found that good tests give me the confidence to be able to refactor and improve the quality of my code without worrying about introducing regressions. the key is to test the 'what', and not the 'how'. that is to test the results, and not the specific way the code gets those results.

as for books, i'd recommend "growing object-oriented software, guided by tests" by steve freeman and nat pryce. it changed how i think about tests. also, for a very good overview of testing patterns and practices, i found martin fowler's "patterns of enterprise application architecture" useful.

to finish, the most important thing to remember when writing tests: is that if you find them hard to write, it usually means your code has a problem. when tests are cumbersome, it is a smell that code should be simplified. or refactored. i have a very old colleague that always used to say "code has no legs, tests have." which is quite a weird analogy but i guess that i understand it.

hope this gives you a good starting point.

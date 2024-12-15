---
title: "Why am I getting a Kaminari Pagination TypeError?"
date: "2024-12-15"
id: "why-am-i-getting-a-kaminari-pagination-typeerror"
---

alright, so you're hitting a `typeerror` with kaminari pagination, huh? been there, done that, got the t-shirt, probably spilled coffee on it too. it's one of those things that can pop up and make you feel like you've suddenly forgotten everything about rails. let's break this down.

first off, a `typeerror` generally means you're trying to do something with a variable that it's not designed for. in the context of kaminari, which is a gem specifically designed for handling pagination of activerecord collections or arrays, it usually boils down to kaminari not receiving the data it expects. i've debugged similar problems for weeks back in my early days, just banging my head against the wall before realizing i was feeding kaminari a potato instead of a proper collection.

the most common culprit, from my experience, is that kaminari is expecting an activerecord relation, or an array, which it then tries to paginate. however, sometimes what you're passing to kaminari's `paginate` method isn't an activerecord relation or a simple array, it could be a single object (instance of a model), or a non-enumerable object, or even `nil`. kaminari is expecting something it can count and slice, and if it gets something it cannot it yells `typeerror`.

for example lets say you have a `user` model and you use this inside the controller.
```ruby
def index
  @users = User.find(1) # finding user 1 and not a collection of users
  @users = Kaminari.paginate_array(@users).page(params[:page]).per(10)
end
```
see the problem? that `user` is an active record object, not an activerecord relation or an array. kaminari cannot paginate a single record, therefore you will get `typeerror` when kaminari tries to paginate.

a similar situation can happen if you use `.find_by` and it doesn't find a record, and returns nil, then kaminari will complain again because it's not a collection to paginate.
```ruby
def index
  @users = User.find_by(username: 'non-existent-user') # finding a user that does not exist
  @users = Kaminari.paginate_array(@users).page(params[:page]).per(10)
end
```

now, what does it mean to pass a correct data type?
first you would need to make sure you are receiving an activerecord relation like in this example
```ruby
def index
  @users = User.all # this returns an activerecord relation (a collection)
  @users = Kaminari.paginate_array(@users).page(params[:page]).per(10)
end
```

or if you are working with an array for example:
```ruby
def index
   @users = [1,2,3,4,5,6,7,8,9,10]
   @users = Kaminari.paginate_array(@users).page(params[:page]).per(3)
 end
```
in that case, `Kaminari.paginate_array` is your friend because it converts a regular array into something kaminari can handle. i once spent an entire evening confused because i was using a custom query that returned an array, forgetting to use `Kaminari.paginate_array`. lesson learned.

another thing you should watch out for is how you are using kaminari with the view. if the controller isn't passing a paginated collection, but your view is still trying to render kaminari's pagination links using `paginate @users` it will also raise `typeerror`. it expects a kaminari paginated collection. it's like trying to use a screwdriver to hammer a nail; they're both tools, but they do different things.

debugging this kind of issue usually involves a few steps.

1.  **inspect the object:** use `puts` or `byebug` to check the data type of the variable before you pass it to `paginate`. make sure that variable you are passing is of type array or activerecord relation, if you are passing something else that's your problem. i once spent hours debugging a query only to find out the result was returning an object instead of an activerecord relation. sometimes these things are as simple as changing `.find` to `.where` or `.all`.
2. **check your queries:** are you using `find_by` or custom queries? you will need to pay close attention to these because they might be your problem as they return an object not a relation or may even return nil.
3. **double-check the view:** if the controller is doing the right thing, make sure your view template is using the kaminari methods correctly. `paginate` should be used with a kaminari paginated collection, not a plain array or an activerecord relation.
4.  **kaminari setup**: check your kaminari configuration file in config/initializers/kaminari.rb if you have one, it could contain something that is interfering with the way it works. it is very rare but it can happen.

in essence, kaminari is very particular about what it receives, it is very demanding and strict about the data type. making sure the variable you are trying to paginate is a proper activerecord relation or array is key. also remember that kaminari has 2 ways of paginating, one which works for activerecord relations (`paginate @users`) and the other which works for arrays (`Kaminari.paginate_array(array).page(params[:page]).per(10)`). use each in the proper context.

this error usually is a common thing, and with time it starts making sense. i remember one time my junior colleague got this type of error and spent the whole day debugging it, and it was as simple as him passing a single object instead of a relation. it's like walking around with your eyes closed, and suddenly you bump into a wall, you then realize you should have just opened your eyes. i felt for him, because i went through this before too.

for resources, i'd suggest checking the kaminari's official documentation (you can find that in its github repository). also the book "rails 7 in action" by ryan davis has an excellent chapter about pagination that explains more about how pagination works. if you want to go deeper, there are plenty of academic papers on database pagination, but those may be overkill unless you want to invent your own pagination method. (not recommended unless you are writing a gem or are researching in the field of computer science).

i hope that gives you some insights, and saves you from hours of debugging, let me know if you need help with anything else, i am always glad to help.

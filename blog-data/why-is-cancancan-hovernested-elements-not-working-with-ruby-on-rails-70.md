---
title: "Why is cancancan hover/nested elements not working with Ruby on Rails 7.0?"
date: "2024-12-15"
id: "why-is-cancancan-hovernested-elements-not-working-with-ruby-on-rails-70"
---

alright, so you're having a tough time with cancancan and nested elements in rails 7.0, right? i've been there, trust me. it’s one of those things that sounds simple on paper, but the reality can be a bit of a headache. let’s get into the nitty-gritty of it.

first off, cancancan is all about authorization. it determines what a user can and cannot do, based on roles, abilities, and whatnot. it’s generally fantastic for single model actions. but when you start nesting things – like a user who owns a blog, and that blog has posts, comments, etc. – things can get dicey very quickly with cancancan.

the problem you're likely experiencing is that cancancan’s ability definitions, when initially created, are generally designed with flat structures in mind. they usually center around a specific model and its actions. the magic happens in your `ability.rb` file where you define rules for each user role and what they can access. but when nested elements come into play, those simple rules don't always cascade down naturally or at least not in the way a lot of developers expect it.

my past experience? oh boy, let me tell you. i once had to build an entire online learning platform with nested courses, modules, and lessons, complete with user roles as students, teachers, and admin. at the beginning, i thought cancancan was gonna solve all my authorization worries, and that everything was going to automatically work just out of the box. i was so wrong. things became a real mess when i tried implementing nested elements like commenting on specific lesson content from a course nested into a module inside of that course.

the issue i found was not only about the user having access to the course to start with, but also that the user had to have access to the module and finally to the specific lesson to add comments. it wasn't working as smoothly as i hoped for. my authorization checks were all over the place, and i had to end up rewriting a good amount of it.

it’s not that cancancan is bad or broken; it’s more about understanding how to structure those nested checks in a way that is both secure and maintainable. a big gotcha is not using the parent’s resource when building the rules. the rule is that when using nested routes, the child controller needs the parent resource id and we have to explicitly use that in our authorization logic.

another important point, is to make sure that your controllers are actually loading the parent model before loading the child. it is very easy to forget, specially if you get used to using `load_and_authorize_resource` without double checking everything, so that’s the first step. the second one, is to write the ability rules using that loaded parent model. so you need to define the abilities not only on the child resources, but also on the parent resources, and that the ability check will also include the parent model check.

let me show you some examples of what i am talking about, and how to avoid getting caught in common errors. let’s start with the `ability.rb` file.

```ruby
# app/models/ability.rb

class Ability
  include CanCan::Ability

  def initialize(user)
    user ||= User.new # guest user (not logged in)

    if user.admin?
        can :manage, :all #admin can do it all
    else
        can :read, Course
        can :read, Module, course: { public: true } #modules that belongs to public courses are public
        can :read, Lesson, module: { course: { public: true } }  #lessons that belongs to public modules are public

        # user who is logged in
        if user.persisted?

          #users can manage their own courses
          can :manage, Course, user_id: user.id
          can :manage, Module, course: { user_id: user.id }  #user can manage modules in their courses
          can :manage, Lesson, module: { course: { user_id: user.id } } #users can manage lessons in their courses

          #users can comment on any lesson
          can :create, Comment
          can :manage, Comment, user_id: user.id
        end
    end
  end
end
```

here, we have a basic set of abilities. admins can do anything. regular users can read public courses, modules, and lessons. a logged-in user can manage their courses, modules, and lessons. a user can create comments and manage its own comments.

now, let's say you have a controller for modules nested under courses. you need to make sure that the parent resource `course` is loaded first, and then used to perform your authorization checks.

```ruby
# app/controllers/modules_controller.rb

class ModulesController < ApplicationController
    load_and_authorize_resource :course
    load_and_authorize_resource :module, through: :course

    def index
        @modules = @course.modules
    end

    def show
        #@module is already loaded because of the load_and_authorize_resource
    end

    #other actions
end
```

and the same goes for the `LessonsController`.

```ruby
# app/controllers/lessons_controller.rb

class LessonsController < ApplicationController
    load_and_authorize_resource :course
    load_and_authorize_resource :module, through: :course
    load_and_authorize_resource :lesson, through: :module

    def index
       @lessons = @module.lessons
    end

    def show
        #@lesson is already loaded
    end

    #other actions
end

```

notice that in both `ModulesController` and `LessonsController` we have the line `load_and_authorize_resource :course` this is very important because, otherwise, when cancancan calls the model the parent object is not available and it will not work properly.

one very common error that i've made before is to only do `load_and_authorize_resource :module, through: :course` and not include the load of the parent resource.

also, if you ever get into weird bugs with complex abilities, sometimes disabling caching on `ability.rb` can help figuring out the issue. `config.cache_classes = false` inside of the `development.rb` environment file can temporarily turn the caching off when developing, to make sure you are seeing your last change in the rules, however remember to turn it back on after you are done as caching helps with performance in production.

a common mistake when defining abilities is also to not use `through:` option when defining children routes, which leads to errors when accessing the children because the authorization logic does not have access to the parent route, and it will fail to authorize the resource. also, remember to include the `course_id` on the `modules` table and the `module_id` in the `lessons` table. i have spent hours wondering what was wrong just to find out that the association between models wasn't created properly.

it's also worth noting that cancancan operates on the principle of "first matching rule wins" – this means the order of your ability definitions in `ability.rb` does matter. if you have a broad rule for all resources followed by a more specific rule, the broader rule might be applied first, negating the effect of the specific one. the best way to avoid that is to keep the broader rule as the last rule.

debugging authorization can also get messy. cancancan throws exceptions if something goes wrong, but that’s not always very informative. i always use the `logger` to see how the model is being loaded and what rules are being called to debug my code, and you should too.

resources wise, i found the official cancancan documentation very helpful to clear some doubts. there's a great book called "rails 7 way" by obie fernandez. its a gold mine of information for rails developers. its a great resource to understand the internal of rails as it doesn't only teach you how to use rails, but also how it works under the hood. and, as a general ruby book, “practical object-oriented design in ruby” by sandi metz can also give a much needed fresh perspective on object-oriented principles.

and finally, remember that in the end, sometimes the problem might not be with cancancan itself, but with the way our models are associated. once i spent a full day on a bug only to find out that one of my `belongs_to` declarations was on the wrong model! you know, like that one time i went to the supermarket for milk and ended up buying a can of sardines.

so, to recap, make sure you are loading parent resources correctly in your controllers. use the `through:` option when defining resources that are nested. be explicit about your rules in `ability.rb` and be aware of the order of rules definitions. use the logger to debug your code. with these tips, hopefully, you should be able to get your nested resources authorized like a pro.

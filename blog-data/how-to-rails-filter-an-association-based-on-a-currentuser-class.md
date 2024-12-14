---
title: "How to Rails: filter an association based on a current_user class?"
date: "2024-12-14"
id: "how-to-rails-filter-an-association-based-on-a-currentuser-class"
---

alright, so you're asking about how to filter an association in rails based on the `current_user`, huh? i’ve been down this road countless times, and i can totally relate to the head-scratching feeling. it sounds straightforward, but it can get a bit tricky depending on the specifics of your setup. let's break it down like we're debugging a gnarly piece of code, and i’ll tell you about some war stories along the way.

first off, let's assume you have a basic user model and another model that's associated with it. maybe it’s a `project` model associated with a `user` model. in your `project.rb` model, you might have something like this:

```ruby
class Project < ApplicationRecord
  belongs_to :user
  # other stuff...
end
```

and in your `user.rb` model:

```ruby
class User < ApplicationRecord
    has_many :projects
  # other user stuff...
end
```

now, the typical scenario where you hit this snag is when you want to fetch only projects that belong to the currently logged-in user. you’ve got a `current_user` method, most probably in your `application_controller.rb` or a similar spot, that gives you access to the current user instance. that's all good and dandy.

the naive way (and trust me, i’ve done this too in my early days) would be something like this in your controller:

```ruby
def index
  @projects = Project.all.select { |project| project.user == current_user }
end
```

this works, *technically*. but it is terrible. why? because we are fetching *all* the projects and then looping through them in ruby. we are not using the database, the thing it is for. we are pulling a lot of records from the database that we don't need to, we are using a lot of memory and that can be really slow. that is a big no-no in the rails world. you don't want to do that in a real application with a lot of records. believe me, i've seen applications crawl at a snail's pace because of similar practices.

the correct way, and the way that will make your code more efficient and your database happier, is to let the database do the filtering with a proper sql query. we'll use where clause to do it. so instead of looping, you can do this in your controller:

```ruby
def index
  @projects = current_user.projects
end
```

or even directly:

```ruby
def index
  @projects = Project.where(user: current_user)
end
```

or also if you only have the current_user's id:

```ruby
def index
  @projects = Project.where(user_id: current_user.id)
end
```

these snippets above use activerecord to do the job in the database. this approach pushes all of the heavy lifting to the database, which is far more optimized for these kinds of filtering operations than ruby is. this means a whole lot faster requests to the database.

i remember one time, ages ago when i was still wet behind the ears, i had a similar filtering issue in a reporting dashboard. we were trying to list all the "reports" belonging to a user, and the original code was a classic case of pulling all the reports into memory, then doing a ruby loop. the application was practically unusable under heavy user load. when i refactored it to use `where` clauses directly in the sql, the dashboard went from a glacial crawl to a snappy experience. it was like day and night. never underestimate the power of a well-crafted sql query. it is a game changer.

now, this method has you covered for simple associations, but let's say you have a situation that is more complex. for example, what if your association isn't a direct `belongs_to` association? let's say you have a `team` model, that can have many `projects`, and the `user` is a member of a `team`. you might have something like:

```ruby
class User < ApplicationRecord
  has_many :user_teams
  has_many :teams, through: :user_teams
end
```

```ruby
class Team < ApplicationRecord
  has_many :user_teams
  has_many :users, through: :user_teams
  has_many :projects
end
```

```ruby
class Project < ApplicationRecord
    belongs_to :team
    # other stuff...
end
```

so, how do you fetch all the projects belonging to the user's teams? again, it might be tempting to fetch *all* the projects, and *all* the teams and then iterate. but we know that this will not do the job efficiently. the answer relies on the power of active record joins.

here we go:

```ruby
def index
  @projects = Project.joins(team: { user_teams: :user }).where(user_teams: { user_id: current_user.id })
end
```

this code can look a little bit dense but what it does is, it will fetch all the projects, do a join operation in sql, and only brings the projects of the user using the join operation. if you do not want to use joins for some reason, you can use `where` with a subquery too, like:

```ruby
def index
  @projects = Project.where(team_id: current_user.teams.select(:id))
end
```
this will fetch the ids of the teams, and then make a where condition in the projects using those ids. either one is fine but usually the `joins` approach is the most efficient.

what i learned is that understanding how the queries are made, and what are the optimal ways, is one of the things that differentiate a good developer from a great one. it is one of the things that always should be in your mind. always try to delegate as much work to the database as possible. don't ever loop in ruby if you can avoid it.

also there's an important point i should mention. sometimes you might encounter situations where you don't have a direct `current_user` method readily available in your model. maybe you have a specific `context` that holds your user information, or maybe you have a particular scope you want to apply, or maybe a tenant based application. the general technique still applies, but you will need to find a way to get the current user id or the user instance. always think in terms of database queries instead of ruby loops when you can. that's the ticket.

for more general rails best practices, i recommend reading the rails documentation thoroughly. but you can also read books like "agile web development with rails" or "the rails 5 way" these are good sources of information and have very good best practices for rails developers. also the "effective sql" book is great for understanding sql.

it took a lot of debugging sessions to get this all in my head. actually a former colleague used to joke i was the kind of developer that, if there was a performance problem, would debug so much that i ended up fixing database engine bugs. i guess you can say i got a little too focused on the details, haha. but at the end of the day, it's about finding the right tools and techniques to solve a problem effectively. it is important to know the tools and how they work, you cannot rely on magic. you need to know what the machine is doing.

that's the gist of filtering associations with a `current_user` in rails. you can see it is not very complicated, but doing it properly is very important. just keep the database in mind, use `where` clauses and `joins` when possible and you'll be on the right track. if you have other questions just ask. i'm always happy to share what i’ve learned along the way.

---
title: "How to handle Rails | Ajax Response, from Controller?"
date: "2024-12-14"
id: "how-to-handle-rails--ajax-response-from-controller"
---

alright, so you're hitting that classic rails and ajax dance, where the controller needs to talk back to the javascript, eh? been there, crashed that server more times than i care to count. let's break it down, from a trenches perspective.

first off, the basic premise is simple enough: the javascript throws an ajax request at your rails controller, the controller does its thing, and then spits back a response. the trick is in formatting that response so the javascript understands it and can actually *do* something with it. i've seen countless folks stumble here, usually because they're not quite thinking about the separation of concerns. your rails controller shouldn’t be pushing html strings; it’s better to send data and let the javascript handle the presentation.

back in the day, when i was first getting into this, i made all sorts of messes. i remember this one project, a simple to-do app. i thought it would be clever to render partial views directly from the controller action and send it back to javascript. it worked, technically, but man, what a pain to maintain. it quickly became spaghetti code. javascript was coupled to the html structure of the view and every time i had to change anything in the view, i ended up having to debug the javascript too. it wasn't pretty. the whole application became a ticking time bomb.

the golden rule i learned from that disaster: embrace json. send it from the rails controller, receive it in the javascript. it's like a universal language both sides understand. json is lean, easy to parse, and keeps things sane. instead of sending back rendered html or text responses, we are going to be sending data as a json response.

let's look at a barebones rails controller example. let’s say you've got a model called `task`, and you want to create a new task via an ajax call:

```ruby
# app/controllers/tasks_controller.rb
class TasksController < ApplicationController
  def create
    @task = Task.new(task_params)

    if @task.save
      render json: {
        status: 'success',
        message: 'task created!',
        task: @task
      }, status: :created
    else
        render json: {
          status: 'error',
          message: 'something went wrong',
          errors: @task.errors.full_messages
        }, status: :unprocessable_entity
    end
  end

  private

  def task_params
    params.require(:task).permit(:title, :description)
  end
end
```

this code is pretty straightforward. when the task saves successfully, we’re rendering a json response containing a success status, a message, and the actual task data. if it fails, we send back an error status, an error message, and the validation errors. note the `status: :created` and `status: :unprocessable_entity`. these are important for your javascript to understand what kind of response it’s getting. if it's a 201, the javascript can trigger the success logic, for example. otherwise, if the response is a 422 it means the validation failed and that can be handled separately.

now, on the javascript side, we'd catch this response using something like the fetch api. here is a snippet:

```javascript
// javascript example using fetch

function createTask(title, description) {
    fetch('/tasks', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'X-CSRF-Token': document.querySelector('meta[name="csrf-token"]').getAttribute('content')
        },
        body: JSON.stringify({ task: { title: title, description: description } })
    })
    .then(response => {
       if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
       return response.json();
    })
    .then(data => {
        if (data.status === 'success') {
            console.log('task created:', data.task);
            //do something like rendering the new task
            //on the page or display a success message
        } else {
          console.error('error creating task:', data.errors);
          //handle errors properly and display validation errors
        }
    })
    .catch(error => {
        console.error('fetch error', error);
        //handle network errors
    });
}

//example usage
document.getElementById('create_button').addEventListener('click', function () {
  createTask("my new task title", "my new task description");
});

```

the javascript code sends a post request with the task data as json. it then processes the response, checking if it was successful based on the `status` key in the json response. we use `.then()` to handle the response, checking if there was an error (`response.ok`). we use `.catch()` to handle any networking errors. i’ve seen some people get tripped up here too, forgetting the `response.json()`. that part’s crucial. the `X-CSRF-Token` header is also necessary in rails applications to protect from csrf attacks.

it’s worth noting we’re doing this with fetch api but there are also other options, such as `axios`. each has its own particularities but at its core, they all handle the same problem.

but what if you need to send a list of tasks? then you would do it similarly, sending an array of tasks. remember, json is your friend:

```ruby
# app/controllers/tasks_controller.rb
class TasksController < ApplicationController
  def index
     @tasks = Task.all
     render json: {
        status: 'success',
        tasks: @tasks
      }, status: :ok
  end
end
```

and here is the javascript snippet:

```javascript
// javascript example using fetch
function fetchTasks() {
    fetch('/tasks', {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'X-CSRF-Token': document.querySelector('meta[name="csrf-token"]').getAttribute('content')
        }
    })
    .then(response => {
       if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
       return response.json();
    })
    .then(data => {
        if (data.status === 'success') {
            console.log('tasks fetched:', data.tasks);
            //render the tasks on the page
        }
    })
    .catch(error => {
        console.error('fetch error', error);
        //handle network errors
    });
}

//example usage
document.getElementById('fetch_tasks_button').addEventListener('click', function () {
  fetchTasks();
});

```

in this case we're using `get` method and we're fetching all tasks from the database. we’re sending the entire list back in the response as json under a `tasks` key. the javascript code then iterates over this array, maybe rendering the task list on the page. the `X-CSRF-Token` header is important again since this is a rails app.

i recall one time, i was debugging a similar system and the problem was i wasn't returning the right content type header on the rails side. turns out the server was sending a plain text response, and the javascript was trying to parse it as json. it’s one of those things that’ll make you want to throw your computer out the window. but hey, we’ve all been there.

a few other quick notes:

-   **status codes matter**: 200, 201, 400, 404, 422, 500, etc. use them correctly to help your javascript code handle errors and success scenarios. it's not a case of randomly choosing, you should use the status code that most closely fits the specific situation.

-   **always sanitize user input**. i’m repeating this. *always sanitize user input*. it’s not just about ajax responses, it’s good practice in general, but it's crucial for security. do not trust any data coming from the client. this is one of those times where you should be paranoid.

-   **think about error handling**. don't just log the error to the console, make the user know there was an error, and let them know what they can do about it.

resources? skip the overly simplified tutorials. dive into the actual documentation. *restful web services* by leonard richardson is a good starting point in thinking about api architecture. the *rails api* docs will be your bread and butter for dealing with the controller side. and the *fetch api* documentation is essential in the javascript front.

that's all. it's a common problem. the key is simple json, correct http codes, proper error handling and don't ever try to render partials from the rails controller and send that to javascript. that last point... trust me on that one. that's how you make code that is, like, technically working but it's all just... *wrong*.

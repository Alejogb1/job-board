---
title: "How to handle Rails | Ajax Response, from the Controller?"
date: "2024-12-15"
id: "how-to-handle-rails--ajax-response-from-the-controller"
---

so, you're hitting that classic rails and ajax response situation, eh? been there, coded that. it's a dance we all learn, sometimes the hard way. i remember back in '09, i was building this awful cms for a local newspaper (don't ask, it was a learning experience). the javascript was basically spaghetti code and the rails responses were even worse. i was sending back entire html fragments and just appending them to the dom with jquery. it was slow, it was fragile, and i cringe just thinking about it.

let's break down what we're dealing with, the core of the problem is that the browser javascript side, typically with the help of something like fetch or the old xhr, sends a request to your rails backend, and your rails controller needs to decide what to send back. it’s not a simple case of just rendering a full page like we do with standard http request. the javascript needs data, often in a structured format like json, to update portions of the page without full refreshes. it’s not enough to just send some html back and call it a day (trust me, i tried it).

first, you've got to understand the basics, we want to use a `respond_to` block in our controllers. this allows you to handle different types of requests differently. it’s your key to tailoring the output of your controller action. instead of just `render`, you specify the format you're responding to.

```ruby
# app/controllers/posts_controller.rb
def create
  @post = Post.new(post_params)

  respond_to do |format|
    if @post.save
      format.html { redirect_to @post, notice: 'post was successfully created.' }
      format.json { render json: @post, status: :created }
    else
      format.html { render :new }
      format.json { render json: @post.errors, status: :unprocessable_entity }
    end
  end
end
```

in this snippet, if the request has a header that accepts html (like a standard browser request), it'll do the standard redirect we expect. but if the request has a header that accepts json, (as typically sent by ajax), it’ll serialize the `@post` object as json. a `status` code will also be returned for http code status management, it helps you know what happened on the server side.

note that i'm using the `render json: @post` convention here. this takes the `@post` object and automatically serializes it to json, it works out of the box. but in real world projects, especially with larger objects, you usually don't want to just dump everything and i suggest you check out json serialization with things like `active_model_serializers` it gives you much finer control over what attributes are included, it can save lots of bandwidth and improve your performance significantly. also, you might consider using view models to structure your data, so your controller is not coupled to the internal object representation. but that's a topic for another day.

let’s say you have a `like` button and when clicked we just want to return the new number of likes. here's how you might handle it with ajax, assuming you're using fetch or something similar in your javascript.

```ruby
# app/controllers/posts_controller.rb
def like
  @post = Post.find(params[:id])
  @post.increment!(:likes_count)
  
  respond_to do |format|
    format.json { render json: { likes_count: @post.likes_count } }
  end
end
```

the `increment!` will atomically increase the like count, preventing race conditions. and then in the json block, we are sending back just the like count to the javascript, in a json object. it’s simple, clean, and does exactly what we want.

now on the client-side you’ll want to use the fetch api or similar. with something like `axios` it will be something like.

```javascript
// some javascript code to handle the like button
const likeButton = document.getElementById('like-button');
const likesCountDisplay = document.getElementById('likes-count');

likeButton.addEventListener('click', async (event) => {
  event.preventDefault(); // prevent form submission if needed
  const postId = likeButton.dataset.postId;

  try {
    const response = await fetch(`/posts/${postId}/like`, {
      method: 'POST', // or patch as your need
      headers: { 'content-type': 'application/json', 'accept': 'application/json' }
    });

    if (!response.ok) {
        throw new Error(`http error! status: ${response.status}`);
    }
      
    const data = await response.json();
    likesCountDisplay.textContent = data.likes_count;

  } catch (error) {
    console.error('failed to update the number of likes', error);
    // handle error
  }

});
```

this is a very simple example, obviously, it would be better to handle loading and errors in a more robust manner, maybe display some loading indicator while fetching the data. however, you get the gist of the process. the javascript sends the request, we fetch and process the json output.

you might also want to send some headers back, this could be necessary for more complex scenarios. for instance, you may want to indicate whether the user is logged in, or the data is fresh, or even trigger events via custom headers. you can achieve that easily.

```ruby
# app/controllers/posts_controller.rb
def show
  @post = Post.find(params[:id])
  
  response.headers['x-custom-header'] = 'some-value'

  respond_to do |format|
    format.html # default render
    format.json { render json: @post }
  end
end
```

here i've added a custom header before the `respond_to` block. it's important to add headers before we send our response. if you forget this simple detail you'll have a hard time diagnosing it. it's always a good idea to inspect the network panel on your browser to see all the headers and data that is going back and forth, this often is a source of many debugging session.

regarding the 'content-type' header, the way rails handles it is interesting. if you send a `format.json`, rails automatically sets the `content-type` as 'application/json'. but if you do a `render json: @post.to_json` rails does not set this for you. also, if you send a simple string you will need to manually set the headers. but usually, you are dealing with structured data and it will be correctly set for you, but again, always good to double-check on the network panel. the content-type of the response is very important. if you’re sending json data, make sure the header is set to `application/json`. otherwise your javascript might have a hard time parsing it. if the content-type is plain text the javascript needs to handle the parsing manually. it always better to be specific.

a note on `status` codes: use them wisely. the http status codes are there to signal what happened. a `200` means everything is fine, `201` means something was created, `400` or `422` indicates bad requests and validation errors, `500` indicates an error on the server. always use the correct status code. it will help you debugging and the client code to handle errors appropriately. remember, a happy response is a `2xx`, not just that everything went through the pipes.

also, be mindful of security when handling ajax request. csrf protection is something that you should always handle. rails does it for you on standard forms and requests, but for ajax calls, you usually have to pass the csrf token in the headers.

i've seen code where people try to use `render :partial` to send snippets of html, it might work for simple cases but it quickly becomes a mess when your app grows. json responses are more flexible and easier to maintain. remember the mantra: json for data, html for full-page loads. you could also try graphql if you need more specific needs for your data.

as for further reading, i recommend "rails 7 by example" by kevin skoglund, if you’re just starting. it’s hands-on, and the examples are very good. for deeper dives into http, "http: the definitive guide" by david gourley and brian totty is the bible. and if you really want to geek out about serializers, i suggest reading "effective java" by joshua bloch, even though it’s java specific, it has good ideas on how to structure code for maintainability. and remember, the best learning comes from practice. try building simple crud apps and focus on handling ajax properly. if you are a full stack javascript person, i suggest you look into fastify and nestjs as those ecosystems, in general, handle ajax calls much better from the get-go.

oh, and a small tech joke: why did the javascript developer break up with the html developer? because they had no chemistry. :)

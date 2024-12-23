---
title: "Build a Simple Web Server with Python and Flask"
date: "2024-11-16"
id: "build-a-simple-web-server-with-python-and-flask"
---

dude so i watched this totally rad video about building a simple web server using python and flask it was like a total rollercoaster of emotions and code and i'm gonna spill all the tea right here for you because you know i love to geek out about this stuff


setup/context:

 so the whole point of the video was to show us how to whip up a basic web server from scratch using python which is my jam and flask a python framework that makes building web apps a million times easier think of it as a supercharged toolbox for web developers the dude in the video was a total pro he walked us through the whole process from installing stuff to running our first ever hello world webpage like a total boss


key moments:

1. setting up the environment:  this part was like watching a cooking show except instead of chopping veggies we were installing packages using pip which is basically the python package installer you know the deal pip install flask then you install virtualenv and create your project environment so we don't mess up our system python its like having a super clean workspace it was pretty straightforward but that virtualenv is key because you dont wanna mess up your system python if that happens it's almost a whole lotta pain to fix so be super careful in that step


2. creating the flask app:  this is where things got interesting because we were basically creating the skeleton of our web server using flask i mean the code was short and sweet  from flask import flask app = flask__name__  it was almost too easy at first i was kinda wondering "is that it"  then we started adding routes which are basically like addresses for our web pages  @app.route("/") def hello_world()  return "hello world" this part was pretty wild  it was magic to me seeing how the server was building up  the guy even showed how to handle different routes like /about or /contact he used decorators which are like fancy ways of adding extra functionality to functions think of them as little helper elves


3. handling requests: this part was where we got into the guts of how a web server actually interacts with clients basically when someone visits your website your server receives a request the video showed how to process those requests using the request object in flask it's like a package that arrives with all the information from the client like the url and other parameters the guy demonstrated how to extract that information and use it to customize the response  we learned to deal with get and post requests which are the main ways people communicate with your server  post requests for sending info get requests for looking up stuff so now you can send info to your server


4. running the server: finally the moment of truth we ran our beautiful creation  app.run(debug=true) and bam our little server sprang to life  and it even had debugging turned on so we could see what was happening which was helpful   and the guy showed us the address that we can go to in our browser to see it and there it was a webpage we built showing "hello world" it sounds super simple but i was so stoked i literally celebrated for like 5 minutes it was like "oh my god i just built a website"


5. adding some flair: the video didn't stop there it added a tiny bit of html to our app to add a few customizations like changing the text color  it wasn't super complex but it showed that we could easily expand the server with more complicated html js and css or frameworks like react or vue


code snippets:


snippet 1: setting up the flask app:

```python
from flask import Flask

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<h1 style='color:blue;'>Hello, World!</h1>"

if __name__ == "__main__":
    app.run(debug=True)
```

this is the basic structure of a flask app  it imports flask creates an app instance defines a route for "/" and returns a simple html message  that debug true is crucial for development it gives us useful error messages which saved me from so much frustration plus it reloads automatically which is great


snippet 2: handling a get request with parameters:


```python
from flask import Flask, request

app = Flask(__name__)

@app.route("/greet/<name>")
def greet(name):
    return f"<p>hello {name}!</p>"

if __name__ == "__main__":
    app.run(debug=True)
```


this one shows how to handle dynamic routes the  /greet/ part is a variable that is passed in the url  when you visit /greet/john you get a personalized message  this is powerful and can handle pretty complex stuff  we can read other things from the request object if we needed to


snippet 3:  a more complex example with templates:


```python
from flask import Flask, render_template

app = Flask(__name__)

@app.route("/profile/<username>")
def profile(username):
  return render_template('profile.html', username=username)

if __name__ == "__main__":
    app.run(debug=True)

```

this one introduces templates that separate html from python this way it's easier to maintain it and build up your web application it would also be a good place to start integrating javascript or a framework


resolution:

so the big takeaway was that building a web server is way easier than i thought and not as mysterious as it may seem   flask is super beginner-friendly and it gives us some awesome tools to develop a full web application with relative ease  i went from zero to a simple functioning server in like a few hours  and the best part is i can now build upon this foundation to create something more complex and more useful  it's kind of addictive  i'm already thinking about my next project which is like a full web application with a database  yeah i'm really excited


visual and spoken cues:

the video had a pretty chill vibe the guy's voice was like super relaxed he used lots of visual aids like code snippets and screenshots  and there were cute little animations popping up whenever he installed a package which was a nice touch  one thing that stuck out was when he made a mistake in typing code and he just laughed it off  it was super relatable  made me realize i'm not alone in my struggles


overall:

seriously i'm so pumped about the video it sparked my interest and i learned a whole bunch of new things i can now start building out some of my own stuff and really start to feel like a proper web developer even though i just started  building my own server was like a real rite of passage  it was awesome

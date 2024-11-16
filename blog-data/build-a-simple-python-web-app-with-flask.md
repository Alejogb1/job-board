---
title: "Build a Simple Python Web App with Flask"
date: "2024-11-16"
id: "build-a-simple-python-web-app-with-flask"
---

dude so i just watched this totally rad video about building a simple python web app using flask and honestly my brain is still kinda fried but in a good way like that satisfying feeling after a really intense workout  the whole thing was about showing how ridiculously easy it is to whip up a basic website that actually *does* stuff not just some static html page your grandma could make  they started with the absolute basics assuming you knew less than jack about python or web dev which was super helpful because honestly i'm still pretty much a noob  they aimed to show you the core components of a web app building a little to do list  so basically imagine your brain but as a website listing all the chores you need to do and how to get them done


the setup was amazing  they started by laying out the whole landscape of web development  you know the usual client-server thing  the client sends requests the server responds and then boom you've got a webpage  they really stressed how flask acts as this awesome bridge between python (which you can think of as the powerful engine in the back) and the internet (the crazy highway everyone is driving on) they showed this nifty diagram with all the arrows flying around showing how data flows back and forth and man i'm still thinking about those arrows


one thing that totally blew my mind was how they talked about request objects in flask  it's like this magical little package that contains all the juicy information the client sends  like what page they want to see what data they're submitting  everything  they even showed a snippet of code where they accessed a specific field from that package  it was like peeking into the mind of the user  seriously  imagine this:


```python
from flask import Flask, request, render_template

app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        task = request.form['task'] #this line is the magic grabbing user input
        # add task to a database or list or whatever
        return "Task added: {}".format(task)
    return render_template('index.html') #renders a user interface

if __name__ == "__main__":
    app.run(debug=True)
```

this little piece of heaven is where the user input is captured   `request.form['task']` grabs whatever the user typed into the form named "task" and you can then do whatever you want with it  add it to a database show a confirmation message send it to a space station  it's all up to you


another killer moment was when they explained routing  it's basically telling flask which part of your code should handle specific urls  it's like setting up a switchboard for your website  if someone goes to `/about` they'll get the about page  if they go to `/contact` they get the contact page you get the idea  they used this super simple example:


```python
from flask import Flask

app = Flask(__name__)

@app.route("/about")
def about():
    return "this is the about page"

@app.route("/contact")
def contact():
    return "this is the contact page"


if __name__ == "__main__":
    app.run(debug=True)
```


look at that beauty  super straightforward right you define a function for each page and decorate it with `@app.route` and boom it works its magic and the browser displays it  it's like creating little portals to different parts of your app


and then there was the part where they showed how to create dynamic content using templates  oh my god  this was a game changer  no more manually writing html for every little change  they used jinja2 which is flask's templating engine and that's where things got really interesting  they showed how to pass data from your python code into the html template  imagine like making a personalized letter to each user where you grab their name from a database and put it into the letter  this is what jinja2 allows for  basically the template is the skeleton and your python code adds the meat


```python
from flask import Flask, render_template

app = Flask(__name__)

@app.route("/")
def index():
    name = "john doe" #this is a variable  imagine getting this from a database
    tasks = ["walk the dog", "buy groceries", "code more"]
    return render_template("index.html", name=name, tasks=tasks)  #passing variables into template


if __name__ == "__main__":
    app.run(debug=True)
```

and to create index.html(a separate file)


```html
<h1>hello {{ name }}</h1>
<ul>
    {% for task in tasks %}  #looping over the tasks list
        <li>{{ task }}</li>
    {% endfor %}
</ul>
```

see how we use `{{ name }}` and `{% for task in tasks %}` this is jinja magic this lets you inject data from your python program into your html page  no more tedious manual updates   it's like creating a super cool customizable form which is really useful when you are creating forms for users to submit things



so the resolution was pretty straightforward  they built this awesome little to-do list app from scratch showing all the essential building blocks of a web app with flask  they demonstrated the power of request objects routing and templating making it clear that building a basic web app isn’t as scary as it seems  it's like assembling a really awesome lego castle  each piece has its function and when you put it all together you get this really cool thing you can show off to your friends


basically the whole video was a crash course in building a simple python web application using flask  it showed how easy it is to create a dynamic website without getting bogged down in overly complex stuff  it was really inspiring and motivated me to dive deeper into flask  plus i got to see some seriously cool visualizations and diagrams  if you're looking to build a simple web app i highly recommend checking out that video – it's a seriously fun ride  i mean seriously  it's like getting a secret decoder ring for understanding the internet  and let's be real who doesn't want a secret decoder ring right?  except maybe my grandma she thinks they're for spies

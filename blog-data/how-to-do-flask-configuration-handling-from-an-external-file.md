---
title: "How to do Flask configuration handling from an external file?"
date: "2024-12-14"
id: "how-to-do-flask-configuration-handling-from-an-external-file"
---

alright, so you're looking to move your flask configuration out of your main app file and into something external, makes sense. i've been there, believe me. i've seen some pretty tangled up flask apps that had all their settings hardcoded into the python file. not pretty.

first off, why do this? well, for starters, it cleans up your main `app.py` or whatever you've named it. keeps the logic separated from your settings. but even more importantly, it lets you configure your app differently for different environments - think development, testing, production. you might want debug mode on during development, but definitely not when it's live.

i remember one project, way back, when we were first using flask for a big data processing pipeline. everything was in one massive `app.py`, including all the database credentials, api keys, even which data directory to use. it was a nightmare to maintain, and each time we wanted to deploy, we had to manually edit that file to change the settings. we had a 'deploy' process that involved `sed` and, i'm not even kidding, lots of praying. the thought of accidentally pushing the dev settings to production kept me awake at night. after that, i vowed to never repeat that mistake again.

so, how to do this properly? flask gives you several options, but i've always preferred using a config file, usually a `config.py` or a `settings.py` which then gets loaded by your main application. this keeps it nicely contained.

here’s the gist of it:

1. create an external configuration file, like `config.py`.
2. in that file, define your configuration as variables.
3. in your main flask application, load in these configurations.
4. use the configurations across your application.

let's walk through some actual examples, so you can see how it's done.

**example 1: simple `config.py`**

here's a basic `config.py` file:

```python
# config.py
DEBUG = True
SECRET_KEY = 'this-is-a-secret-key' # obviously, don't use this in production
DATABASE_URI = 'sqlite:///database.db'
```

here we are setting `debug` flag, which is very useful for local development, setting the `secret_key`, which is something flask needs, and the `database_uri` something that my apps use often.

**example 2: loading config in `app.py`**

now, in your main flask application file, say `app.py`:

```python
# app.py
from flask import Flask
import os

app = Flask(__name__)

# load the configuration from an external file
app.config.from_pyfile('config.py')

# example of using the configuration
@app.route('/')
def home():
    debug_mode = app.config.get('DEBUG')
    secret_key = app.config.get('SECRET_KEY')
    db_uri = app.config.get('DATABASE_URI')

    return f"debug mode is: {debug_mode}, db uri is: {db_uri}"


if __name__ == '__main__':
    app.run()

```

in this case, we use `app.config.from_pyfile('config.py')`. this loads the variables from `config.py` directly into flask’s configuration dictionary, you can access settings with `app.config.get('variable')`. i am also returning to the user the current state of debug mode, and database uri.

**example 3: using environment variables as fallback**

sometimes, you don't want to hardcode sensitive data, even in a config file. that's where environment variables come in handy. you can set those outside your code, typically in your operating system. let's modify the example a bit:

```python
# app.py
from flask import Flask
import os

app = Flask(__name__)

# load config from file
app.config.from_pyfile('config.py')

# override with environment variables if present
app.config.update(
    DEBUG=os.environ.get('FLASK_DEBUG', app.config.get('DEBUG')),
    SECRET_KEY=os.environ.get('FLASK_SECRET_KEY', app.config.get('SECRET_KEY')),
    DATABASE_URI=os.environ.get('FLASK_DATABASE_URI', app.config.get('DATABASE_URI'))
)

@app.route('/')
def home():
    debug_mode = app.config.get('DEBUG')
    secret_key = app.config.get('SECRET_KEY')
    db_uri = app.config.get('DATABASE_URI')

    return f"debug mode is: {debug_mode}, db uri is: {db_uri}"

if __name__ == '__main__':
    app.run()
```

here, we use `os.environ.get()` to look for environment variables such as `flask_debug`, `flask_secret_key`, and `flask_database_uri` , if those variables are set, they will overwrite the values in your config.py, it's important to understand that this happens after the loading of `config.py`. it provides a fallback value if the environment variable isn't set using `app.config.get()`. this is the exact trick i used to keep database credentials out of the code when deploying, by then creating a systemd script that exported all the needed configurations. we never had to modify our configuration files directly again. we also used git pre push hooks that checked if the configuration variables were included on our commits, if they were, the commit was blocked, which saved some developers from a lot of headache.

now, a bit of a joke i heard the other day in an office, a bug walked into a programmer's house. the programmer shooed it away, then opened the terminal, created a docker file, a docker compose file, set an nginx reverse proxy and started the application. when the bug finally flew into his house again he called the exterminator.

**some resources i recommend:**

*   **"flask web development" by miguel grinberg:** this is an excellent book. it's very practical, easy to read, and gives you great patterns for structuring your app, and it covers configuration management in detail. it's a must-read if you're serious about flask development.

*   **flask documentation:** as always, the official documentation is your friend. specifically, check out the configuration section. it lays out all the possible configuration options and methods for loading settings. it should have the `from_json`, and `from_mapping` options and ways to use command line arguments to influence configurations.

*   **the twelve-factor app methodology:** this isn’t flask-specific, but it’s an excellent set of practices for building web applications. one of the main points is configuring your application using environment variables, i've found this to be critical for any modern web development process.

*   **the dotenv library:** if you use the environment variable approach, consider using a dotenv file (`.env`) for development, the `dotenv` library will help you to load those variables in a simple way. it is a simpler approach than using a `config.py` file.

*   **various online tutorials:** websites like realpython, and digitalocean have some great flask guides. these are often more focused on specific tasks, so they may have even more specific examples related to your particular needs. you will find many examples of people dealing with configuration management there.

remember, the goal is to keep your configuration separate from your code, make it flexible, and make it easy to deploy your flask apps in different environments without the need of manual changes. it's something you set once, and it will save you a lot of time and frustration in the long run. this approach with a mix of external files and environment variables has worked for me in production across many projects, and i really hope it helps you too. let me know if you get stuck or want more specifics.

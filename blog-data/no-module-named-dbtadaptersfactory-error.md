---
title: "no module named 'dbt.adapters.factory' error?"
date: "2024-12-13"
id: "no-module-named-dbtadaptersfactory-error"
---

Alright so you're hitting that good old "no module named 'dbt.adapters.factory'" error huh I feel you Been there done that bought the t-shirt and even accidentally compiled the wrong version of the adapter and spent an hour figuring out what went wrong classic stuff

This error usually means your dbt installation specifically your dbt-core setup isn't finding the adapter library that it needs It's like trying to make a specific type of sandwich without the right kind of bread It's not a core dbt thing but it needs a specific adapter for your database say postgres or snowflake to make the magic happen

Here's how I’d approach debugging this and how I've banged my head against this exact issue before

First off lets confirm the obvious Did you install the correct dbt adapter for the database you're working with I mean this sounds basic but its the root of like 80 percent of these kinds of issues I swear it was a Friday afternoon and i was trying to deploy to production I kept getting this error after changing my deployment scripts to a more "modern" approach and forgot to add the dependency to the docker image So i was running with vanilla dbt and no way to connect to postgres my own stupidity caused a minor outage

Assuming you're working with Postgres it should be something like `pip install dbt-postgres` for Snowflake it would be `pip install dbt-snowflake` and so on You get the idea if you want to connect to say big query it would be `pip install dbt-bigquery`

Here's a quick sanity check command that I use all the time

```bash
pip list | grep dbt
```

That will give you a list of all dbt related packages you have installed If you don’t see the adapter you need listed you’ve got your answer If it is there double check the name it can be quite common to install the wrong package like accidentally installing `dbt-postgres-compat` instead of `dbt-postgres` as an example Been there done that and the compat one is not a real dbt adapter it just a backward compatibility library that i once installed for no reason just because it was in the autocompletion list I learned the hard way to not trust the auto completion fully

Now if you confirm you have the right adapter installed but are still getting that 'dbt.adapters.factory' error there’s a good chance it might be a version mismatch scenario between dbt-core and the adapter You can use the same command to check the version of your dbt core package and the adapter package

```bash
pip list | grep dbt
```

I was working on this huge dbt project once and we decided to upgrade the dbt version a minor version upgrade they said not too big of a deal They were so wrong We started seeing the no module error and after some head scratching we realised that we didn’t upgrade the adapters at the same time The adapter was a few minor versions behind dbt-core and it was creating the issue that we saw This stuff happens all the time when you're managing dependencies in a big project it’s like a real-world jigsaw puzzle with different pieces that need to be an exact match

Also a good practice is to use a virtual environment For instance virtualenv or pipenv or conda for your dbt projects because this helps avoid the version conflict and messy installs if you install packages in your main system environment you can find yourself with a dependency conflict disaster a situation like i found myself in once when installing packages randomly using pip that broke my python installations so now i always use a virtual environment a must do

To check your virtual environment you can check the location of your python executable within the virtual environment I’m sure you know this if you're asking the question but always good to mention to avoid simple mistakes

```bash
which python
```

You would see something like /path/to/your/venv/bin/python not /usr/bin/python or similar system wide paths you should also see the specific version of python you are running in that specific environment its quite useful when debugging these kinds of issues especially when your team have different python versions and some are using incompatible package versions

Now if the versions all look good double check your `dbt_project.yml` file make sure the `profile:` option is configured correctly. This is the profile that the dbt project uses it maps to the configuration found in `~/.dbt/profiles.yml` and usually this configures which database to connect to and uses an adapter to communicate with that database if the profile is incorrect you might get this error because dbt will not be able to load the adapter to connect to your intended database

Here’s an example of what your `profiles.yml` might look like

```yaml
your_profile_name:
  outputs:
    dev:
      type: postgres
      threads: 4
      host: your_host
      port: 5432
      user: your_user
      pass: your_password
      dbname: your_database_name
      schema: your_schema_name
  target: dev
```

Here is another example using a snowflake connection

```yaml
your_profile_name:
  outputs:
    dev:
      type: snowflake
      threads: 4
      account: your_account
      user: your_user
      password: your_password
      role: your_role
      database: your_database_name
      warehouse: your_warehouse
      schema: your_schema_name
  target: dev
```

And finally a bigquery connection config example

```yaml
your_profile_name:
  outputs:
    dev:
      type: bigquery
      threads: 4
      method: oauth
      project: your_project
      dataset: your_dataset
      location: your_location
  target: dev

```

Remember to replace the placeholders with your actual connection details Sometimes there are extra configurations that you might need based on your database settings such as the `client_id` `client_secret` etc It all depends on how you configure your cloud database credentials

If after all this you're still hitting a wall it might be something with the way your dbt environment is set up or maybe some hidden environment variables are messing things up Sometimes I even found a rogue pip install that i made ages ago that was creating havoc and causing different versions of the libraries to exist and make me go crazy when i was trying to figure out what was happening

If you're still struggling after all of this well sometimes the python package manager pip is not always perfect and there can be corrupted or conflicting package versions and it may be a good idea to fully remove dbt and all the adapters and install everything from scratch in a new virtual environment using

```bash
pip uninstall dbt-core dbt-postgres dbt-snowflake dbt-bigquery -y
```

And then install them again with

```bash
pip install dbt-core dbt-postgres dbt-snowflake dbt-bigquery
```

It's extreme but it's my last resort when the situation is dire A clean install is often the best way to solve some annoying issues like this that have no clear reason

For further reading and resources beyond my personal experience I’d recommend checking out the official dbt documentation they have sections on installation and troubleshooting which are invaluable There's also the "Data Modeling with dbt" book by Claire Carroll which explains many core concepts of dbt and how to use it effectively especially useful when trying to troubleshoot issues Also checking the release notes of dbt-core and the adapters for any breaking changes or compatibility issues is a good practice and if you really want to get into the details the dbt’s source code which is available on GitHub. It's surprisingly readable and you can learn a lot by diving in and seeing how things are actually implemented if you are into that kind of stuff

Oh and one last thing I spent like 3 hours last week trying to figure out a dbt issue it turned out my cat decided the power cord was a good chew toy dbt was not the problem it was the power cord funny right

Let me know if you still need any help or if you have any more details about your setup I’m always happy to try and help out with dbt issues I've been there done that more than once

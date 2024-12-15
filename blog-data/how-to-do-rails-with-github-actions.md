---
title: "How to do Rails with github actions?"
date: "2024-12-15"
id: "how-to-do-rails-with-github-actions"
---

alright, so you’re asking about setting up rails with github actions, eh? it’s something i've spent a decent chunk of time on, and it's definitely a workflow booster once you get it nailed down. i've seen my share of deployment headaches, believe me. back in the day, we were ssh-ing into servers, pushing code, running migrations, praying to the gods of ruby. github actions, when used well, takes a lot of that mess out of the equation.

first off, it's good to have some basic understanding of yaml, because that's what github actions files use. you'll be writing a `.github/workflows/your_workflow_name.yml` file. this file basically describes a set of instructions – a workflow – for github to execute. let's walk through the key parts you'll need.

a basic workflow file will look like this:

```yaml
name: rails ci/cd
on:
  push:
    branches:
      - main
  pull_request:
    branches:
      - main

jobs:
  test:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:13
        ports:
          - "5432:5432"
        env:
          POSTGRES_USER: user
          POSTGRES_PASSWORD: password
          POSTGRES_DB: test_database
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
    steps:
      - uses: actions/checkout@v3
      - name: set up ruby
        uses: ruby/setup-ruby@v1
        with:
          ruby-version: '3.2'
          bundler-cache: true
      - name: setup database
        run: |
          bundle exec rails db:setup
      - name: run tests
        run: |
          bundle exec rspec
  deploy:
    runs-on: ubuntu-latest
    needs: test
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v3
      - name: set up ruby
        uses: ruby/setup-ruby@v1
        with:
          ruby-version: '3.2'
          bundler-cache: true
      - name: deploy to production
        run: |
          bundle exec cap production deploy
```

this yml file covers a common ci/cd (continuous integration/continuous deployment) workflow. let me break it down piece by piece:

*   `name: rails ci/cd`: this line just gives your workflow a name. you can name it whatever you want. it’s just for your reference.
*   `on:`: this section defines the triggers for this workflow to start running. in this case, it will run when you push commits to the `main` branch, or when you make a pull request targeted at the `main` branch.
*   `jobs:`: this is where the actual work happens. workflows are broken down into jobs. each job runs independently and on a fresh vm. we've defined two jobs here, `test` and `deploy`.
*   `test:`: this is the first job, the one that runs tests.
    *   `runs-on: ubuntu-latest`: specifies that this job will run on an ubuntu virtual machine.
    *   `services:`: this section defines external services that need to be running for the tests. here, we're running a postgres database.
        *   `image: postgres:13`: this specifies which docker image to use.
        *   `ports: "5432:5432"`: this maps the postgres port 5432 within the docker container to port 5432 in the github actions environment.
        *   `env: ...`: this sets environment variables that postgres will use, such as the database username, password and the default test db name.
    *   `steps:`: each job is made up of steps. the steps specify individual actions to take.
        *   `uses: actions/checkout@v3`: this step checks out the code from your repository. it's usually the first step in any workflow.
        *   `uses: ruby/setup-ruby@v1`: uses the ruby action to setup the ruby version that you use in development.
        *   `with: ...`: this specifies the version of ruby to use and also tells bundler to cache gems in between workflow runs.
        *   `run: bundle exec rails db:setup`: this command sets up the test database.
        *   `run: bundle exec rspec`: finally, this command runs the tests using rspec.
*   `deploy:`: this is the second job, the one that does the deployment.
    *   `needs: test`: this tells github actions that the deploy job should only run after the test job has completed successfully.
    *   `if: github.ref == 'refs/heads/main'`: this ensures that the deploy job is only run if the push is on the `main` branch. you can add additional branches here if needed.
    *   the deploy job also checks out code and sets up ruby, just like the test job.
    *   `run: bundle exec cap production deploy`: this line is where we actually deploy to production using capistrano. this line might look a little short if you haven’t worked with capistrano before, but it essentially takes the current state of code and pushes it to production servers as well as running database migrations.

this example assumes you have a capistrano setup. if not, your deployment step will be different, perhaps using docker, heroku cli or some other tool.

here’s another common workflow, this one is for deploying to heroku using the heroku cli:

```yaml
name: deploy to heroku
on:
  push:
    branches:
      - main
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: ruby/setup-ruby@v1
        with:
          ruby-version: '3.2'
          bundler-cache: true
      - name: deploy to heroku
        env:
          HEROKU_API_KEY: ${{ secrets.HEROKU_API_KEY }}
        run: |
          bundle install
          git remote add heroku https://git.heroku.com/${{ secrets.HEROKU_APP_NAME }}.git
          git push heroku main:main
```

this workflow triggers when you push to the `main` branch and pushes your code to heroku. it uses environment variables as well. in github actions environment variables are stored as 'secrets' within the repository. so before running this you would need to add the `HEROKU_API_KEY` and `HEROKU_APP_NAME` secrets in your repository settings.

and finally, here’s another small example of a workflow that uses docker compose for the test environment:

```yaml
name: rails ci with docker compose
on:
  push:
    branches:
      - main
  pull_request:
    branches:
      - main
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: set up ruby
        uses: ruby/setup-ruby@v1
        with:
          ruby-version: '3.2'
          bundler-cache: true
      - name: build docker compose
        run: docker-compose build
      - name: run docker compose up
        run: docker-compose up -d
      - name: setup database
        run: docker-compose run --rm web bundle exec rails db:setup
      - name: run tests
        run: docker-compose run --rm web bundle exec rspec
      - name: teardown docker
        run: docker-compose down
```

this workflow uses `docker-compose` to build and run the test environment. as you can see this is slightly more involved but in some circumstances, using docker can be a lot more predictable.

a few things to note. you'll want to manage your rails application's secrets safely. github actions provides an area to manage these in the repository's settings. avoid hardcoding secrets into your yml files. it's the easiest way to get a hacked repository. also, github actions workflows can quickly grow in complexity, so try to keep them clean and well-commented.

i remember once trying to debug a deployment issue where a github actions workflow was deploying an older version of the app. spent hours tracking it down until i found it was a typo in the branch name in the yml. so yeah, double-check all the little things. debugging yaml files can be a special sort of hell i wouldn't wish on anyone, sometimes it makes you ask yourself if you are the problem.

if you are looking for some more detailed documentation or concepts on ci/cd, a good book is *continuous delivery: reliable software releases through build, test, and deployment automation* by jess humble and david farley. and for a good background on ruby and testing i would recommend *effective testing with rspec 3* by myron marston and ian dein. and also the official rails docs are always a great starting point of course.

anyway, this should get you going. let me know if there is anything else.

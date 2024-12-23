---
title: "dbt run target parameter definition?"
date: "2024-12-13"
id: "dbt-run-target-parameter-definition"
---

 so dbt run target parameter yeah I've wrestled with that beast more times than I care to admit Let's break it down simple and techy like we do around here

So you're asking about the `--target` parameter when you're firing off `dbt run` or any other dbt command right It's fundamental but it can be a real pain if you don't grasp its function so let's clear the confusion

Essentially `--target` tells dbt which profile to use When you set up dbt you configure profiles in your `profiles.yml` file These profiles contain connection details for different databases or environments Think of it as different ways to talk to your data warehouse You might have a `dev` profile pointing to your local development database a `staging` profile pointing to your pre-production environment and a `prod` profile for the real McCoy the actual production database

Now without `--target` dbt defaults to a target named `dev` If you haven't changed the default target name during your setup that is what is used This can lead to a world of pain if you try to run a production dbt job on a development environment profile which I have done trust me that resulted in some data loss that made my heart skip a beat I am being very very serious it was like a scene from a bad hacker movie where I am frantically trying to revert the production data I even had to involve several database engineers and a very angry project manager because I didn't pay attention to the freaking target

So `--target` is like saying "hey dbt use *this* connection profile not the default one" You need to specify the correct target to run your models in the right place that's it it's really not rocket science but forgetting to set it right is something that we all experienced at some point in our careers right

Here's the thing the structure of the profile configuration matters too that is like the building blocks of your setup and where the connection details live In `profiles.yml` you define named targets and their associated connection details like username password database schema and so on A simple example can be seen below:

```yaml
dev:
  outputs:
    default:
      type: bigquery
      method: service-account
      project: your-dev-gcp-project
      dataset: your_dev_dataset
      threads: 4
      keyfile: /path/to/your/dev/keyfile.json
      retries: 2
  target: default
staging:
  outputs:
    default:
      type: bigquery
      method: service-account
      project: your-staging-gcp-project
      dataset: your_staging_dataset
      threads: 4
      keyfile: /path/to/your/staging/keyfile.json
      retries: 2
  target: default
prod:
  outputs:
    default:
      type: bigquery
      method: service-account
      project: your-prod-gcp-project
      dataset: your_prod_dataset
      threads: 4
      keyfile: /path/to/your/prod/keyfile.json
      retries: 2
  target: default

```

This is a basic `profiles.yml` for a bigquery setup. Each target (`dev`, `staging`, `prod`) has its own output configuration with the relevant credentials project id and dataset.

Now when you run `dbt run` if you don't specify `--target` dbt will use the target specified in your profile which defaults to `dev` you can run `dbt debug` to confirm which profile and target are being used. To run your dbt models in production you would do `dbt run --target prod` that simple and if you forget to add `--target` before firing your job then you have the potential to re live my hacker movie moment from earlier so don't forget it.

It's all about making sure dbt is using the correct credentials for the right place I had a very funny incident where my teammate ran dbt run without a target and he ended up creating all the staging tables in his personal dev database he forgot he had set in his local dbt setup it was a long day of clean up that day.

Now let's talk about some more advanced use cases you might want to have different profiles depending on the user or machine This is where environment variables come in handy You can dynamically set the target using an environment variable like this

```bash
export DBT_TARGET=staging
dbt run
```

Or to have multiple people working on the same project but in different places on the database you can have your `profiles.yml` setup with different target names for each environment and then use an environment variable like:

```yaml
dev:
  outputs:
    my_local_dev:
      type: bigquery
      method: service-account
      project: your-dev-gcp-project
      dataset: your_local_dev_dataset
      threads: 4
      keyfile: /path/to/your/dev/keyfile.json
      retries: 2
  target: my_local_dev
staging:
  outputs:
    my_local_staging:
      type: bigquery
      method: service-account
      project: your-staging-gcp-project
      dataset: your_local_staging_dataset
      threads: 4
      keyfile: /path/to/your/staging/keyfile.json
      retries: 2
  target: my_local_staging
prod:
  outputs:
    default:
      type: bigquery
      method: service-account
      project: your-prod-gcp-project
      dataset: your_prod_dataset
      threads: 4
      keyfile: /path/to/your/prod/keyfile.json
      retries: 2
  target: default
```

Here in this example we have the same environments but we can choose different targets inside our `dev` environment which can allow multiple engineers work on the same project without stepping on each others toes You can then set your environment target with something like `export DBT_TARGET=my_local_dev` and then every time you run `dbt run` it will use that setup. This method helps with setting it up per user in a multi-user environment.

You might also ask what if I want to use other parameters on run based on the target that is also possible check this out:

```bash
if [ "$DBT_TARGET" = "prod" ]; then
    dbt run --full-refresh --vars '{my_var: "prod_value"}'
elif [ "$DBT_TARGET" = "staging" ]; then
    dbt run  --vars '{my_var: "staging_value"}'
else
  dbt run  --vars '{my_var: "dev_value"}'
fi
```

This is just basic shell scripting but the idea is there you can have different variables or different parameters of dbt run depending on which `DBT_TARGET` you set that's pretty handy if you need different setups per environment.

Now remember all of this is very specific to your setup. You need to make sure your `profiles.yml` is set up right and your environment variables are correctly configured You can also create different profiles files if you need to. Don't try to cut corners here or you may end up with the same data mess I had when I first messed up my dbt targets seriously it's a bad place to be if you end up creating all your development tables in a production database.

In terms of resources I recommend you check the dbt official documentation it's really good I also recommend the book "Data Pipelines Pocket Reference" by James Densmore It is a pretty good book about different data pipeline patterns and some good tips and practices it has sections on dbt as well The "Fundamentals of Data Engineering" book by Joe Reis and Matt Housley also provides a solid foundation for understanding dbt in context of broader data engineering practices they are both good books on data engineering that can help you put the dbt target setup in context. I avoid mentioning specific online sources because links can change or disappear over time so stick to the books and documentation you should be good to go

So there you have it all about `--target` it's crucial for managing environments in dbt and if you're not using it correctly you're gonna have a bad time Always make sure you're pointing dbt to the right place this is something I learned the hard way So set up your profiles properly your environment variables and remember to specify the right `--target` or you're going to have a hard time cleaning up after yourself.

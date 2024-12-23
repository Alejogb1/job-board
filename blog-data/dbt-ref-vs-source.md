---
title: "dbt ref vs source?"
date: "2024-12-13"
id: "dbt-ref-vs-source"
---

 so you're asking about `ref` versus `source` in dbt right Been there done that like a million times. Let me break it down in a way that's probably more real than the dbt docs might.

First off let's get real. I've been wrestling with data pipelines since before dbt was even a thing. Back in my day we were writing SQL by hand copy pasting like madmen and praying to the database gods everything would work. We had this huge monolithic redshift cluster our whole company dumped data into and debugging those pipelines was like trying to find a specific grain of sand on a beach in the dark using only a potato. Then dbt came along like a breath of fresh air. It wasn't perfect but it sure as hell beat writing stored procedures that’s for sure.

Anyway to the point `ref` and `source` are really just different ways to declare dependencies in your dbt project. They're not interchangeable they're for fundamentally different things.

`source` is your hook into raw data. It’s how dbt knows where your data actually *starts* from. This might be tables in your data warehouse ingested by Fivetran or Stitch or even data dropped from a csv file by some random script. You’d declare these in your `sources.yml` file.

Think of `source` like the foundation of your house you wouldn’t build your walls straight on the mud would you You need a proper footing and `source` is that footing for your dbt project. You don't control this data this is the wild west of external sources.

Here's a simple example of how a source definition would look like in `sources.yml`:

```yaml
version: 2

sources:
  - name: raw_data
    database: my_raw_data_db
    schema: public
    tables:
      - name: users
        description: raw users data from CRM
        columns:
          - name: user_id
            description: The unique identifier for the user
            tests:
              - unique
              - not_null
          - name: first_name
            description: The users first name
          - name: last_name
            description: The user's last name
          - name: created_at
            description: The timestamp the user was created

      - name: products
        description: raw products table from the ecommerce database
        columns:
            - name: product_id
              description: The unique identifier for the product
              tests:
                - unique
                - not_null
            - name: product_name
              description: The name of the product
            - name: price
              description: the price of the product
```

Now let's look at `ref`. `ref` is used within dbt models to refer to *other* dbt models. When you write `{{ ref('some_model') }}` you are telling dbt "hey this current model depends on the output of 'some_model'". dbt uses these references to create a dependency graph that it uses to execute your models in the correct order. So `ref` is all about data *within* your dbt project itself the data you are massaging you are creating.

Its like once you have your foundation built the `ref` are the walls the rooms the roof all the things you build on your foundation. And the beauty is dbt manages the construction order. You just define the materials to build the construction based on the materials available with the foundation.

Lets say you wanted to create a users model based on the `raw_data.users` source above using a simple select statement this is how this model would look like using `source` and how the model depends on the source for data.

```sql
-- models/users.sql

select
  user_id,
  first_name,
  last_name,
  created_at
from
  {{ source('raw_data', 'users') }}
```

Now say you create another model to aggregate product information `product_agg`. This model would need to depend on the `products` source table and perhaps also have it's own transformation logic and cleaning done to it. Lets say it performs some basic average price calculation based on a single table. Here is how that would look like:

```sql
-- models/product_agg.sql

select
  product_id,
  product_name,
  avg(price) as average_price
from
  {{ source('raw_data', 'products') }}
group by product_id, product_name
```
And now we can build on that. lets say you now wanted a customer lifetime value based on our `users` model and also the average product prices on our `product_agg` model. We would create a new model to calculate this and our new `customer_lv` model will depend on the other models using `ref` to build the new model. This is where the magic starts.

```sql
-- models/customer_lv.sql

select
  u.user_id,
  u.first_name,
  u.last_name,
  avg(pa.average_price) as average_purchase_value
from
  {{ ref('users') }} u
join
  {{ ref('product_agg') }} pa on 1 = 1
group by u.user_id, u.first_name, u.last_name
```
Here the models are built like lego blocks and using ref the relationships are declared and dbt knows the construction order. dbt knows to create the `users` model and the `product_agg` models before building the `customer_lv` model. dbt has a smart way to solve the order based on the relationships declared by you using `ref` and `source`.

The big difference is this: `source` points to *outside* dbt. `ref` points to *inside* dbt. `source` is your foundation. `ref` is your framework built on that foundation. You can't `ref` a `source` that doesn't exist just like you can't build a house without a foundation. You also can't `source` a dbt model. dbt will throw a fit. I know I have tried that like trying to find the end of a roll of duct tape it's pointless.

Debugging problems involving `ref` and `source` is surprisingly common specially when you're working with a large dbt project. You’ll have to check to see if a source is declared correctly usually checking for typos in `sources.yml` or the model name or if you didn’t add a model to your `dbt_project.yml` or didn't run the command `dbt run` this thing will bite you. You must have made that mistake yourself. Remember this your models are only as good as their dependencies.

So to wrap it all up `source` are your raw external data entries. `ref` is how you build dependencies with other dbt models. Use them wisely understand the difference or you'll be debugging things until next year. And hey while you are at it read "The Data Warehouse Toolkit" by Ralph Kimball if you want to get some real data engineering knowledge in you. Also for dbt specific info go straight to the dbt docs. They have improved lately and you won't find better source of truth.

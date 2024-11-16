---
title: "How to Build AI Apps with PostgreSQL and pgvector"
date: "2024-11-16"
id: "how-to-build-ai-apps-with-postgresql-and-pgvector"
---

dude so i just watched this superbase talk it was wild  the whole thing was like a rollercoaster of  open source awesomeness  a little bit of drama and a whole lotta postgress magic lemme break it down for ya

the setup was pretty straightforward  copple the ceo of superbase  basically laid out why they’re all about postgress and pgvector  think of it as a supercharged explanation of why they use postgress as their foundation  for all their back end as a service offerings and how pgvector makes it a ridiculously powerful vector database they also wanted to show off how they built this awesome thing and collaborated with the community

key moment one: the pgvector origin story

so copple started by telling this crazy story about getting an email from some dude greg  greg basically said hey i built this cool postgress extension called pgvector for vector operations wanna use it  copple’s response was basically  “yeah dude lets merge it” next thing you know they've integrated it into their platform like it was nothing

key moment two:  clippy the doc search ai

i almost choked on my coffee  they built an ai-powered doc search thing  called clippy a direct throwback to microsoft's old clippy  it was initially internal for superbase's docs but then other dev teams like mozilla (mdn docs) saw it and  built their own versions it's this amazing example of how a simple idea can snowball into something huge just like that

key moment three: the pgvector performance drama

then things got real  copple showed this benchmark tweet  some dude was basically saying pgvector was slow and inaccurate compared to other vector databases like quadrant i mean that’s a pretty bold claim to make on twitter right?  copple didn’t take it personally  they just hit the gas on optimization they collaborated with andrew kane the original pgvector developer  and boom within a month they had implemented hnsw indexing which drastically improved query speed and accuracy copple even showed this awesome apples to apples benchmark comparing pgvector with pine cone which proves that postgress can really compete with dedicated vector databases

key moment four: the postgress partitioning trick

this is where things got really magical  copple showed this insane example using postgress partitioning to handle a huge number of image embeddings  imagine an app where users upload photos  the problem  storing tons of embeddings is slow and expensive  and users may not need to see all of it  copple’s solution was partitioning the tables  he used a simple function to check if an image is  a "good cat" or a "bad cat" based on similarity with a reference vector   all the "good cats" go to one partition and “bad cats” to another  this let them only index the “good cat” partition for faster search without touching the bad stuff genius

key moment five: postgress as the ultimate ai platform

copple finished by stressing how postgress is incredibly flexible and extensible its age shows but in a good way it’s a battle-tested system with all the tools you’d need for building AI apps  he emphasized its extensibility  it already has primitives  features and functions pgvector being a perfect example plus things like row level security  which gives you granular control over data access and makes your app super secure

code snippet 1: is_cat function

this is the function copple used to determine whether a cat image is "good" or "bad"  pure postgress magic

```sql
CREATE OR REPLACE FUNCTION is_cat(embedding float[])
RETURNS float AS $$
DECLARE
  canonical_cat float[] := '{0.1,0.2,0.3,0.4,0.5}'; --replace with your canonical cat vector
  similarity float;
BEGIN
  similarity := 1 - distance(embedding, canonical_cat); --distance function needs to be defined appropriately, check documentation
  RETURN similarity;
END;
$$ LANGUAGE plpgsql;
```

this function takes an embedding array as input  compares it to a predefined "canonical cat" vector (you'd replace that placeholder with your actual vector) and returns a similarity score the higher the score the better the match

code snippet 2:  trigger function for partitioning

this one's a bit more advanced  it's a trigger that runs before inserting a new cat image into the database  it uses the `is_cat` function to determine the partition

```sql
CREATE OR REPLACE FUNCTION partition_cats()
RETURNS TRIGGER AS $$
BEGIN
  IF is_cat(NEW.embedding) > 0.8 THEN
    NEW.cat_type := 'good';
  ELSE
    NEW.cat_type := 'bad';
  END IF;
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER partition_trigger
BEFORE INSERT ON cats
FOR EACH ROW
EXECUTE PROCEDURE partition_cats();
```

this trigger checks the similarity score computed by `is_cat` and assigns the cat to the "good" or "bad" partition accordingly  pretty neat right?

code snippet 3: creating the partitioned table

this code creates the partitioned table itself  it shows how easy it is to manage large volumes of data in postgress

```sql
CREATE TABLE cats (
  id SERIAL PRIMARY KEY,
  embedding float[],
  url TEXT,
  cat_type TEXT
) PARTITION BY RANGE (cat_type);

CREATE TABLE cats_good PARTITION OF cats FOR VALUES IN ('good');
CREATE TABLE cats_bad PARTITION OF cats FOR VALUES IN ('bad');
```

here we define a main table `cats` and then create individual partitions `cats_good` and `cats_bad`  data is automatically routed to the correct partition based on the `cat_type` column  you can add indexes to each partition for optimized queries

the resolution: postgress is awesome for ai

basically copple’s message was simple  postgress and pgvector are seriously powerful tools for building ai apps  the talk showed how they scaled from a single email to thousands of ai apps launched weekly  they highlighted  the value of community collaboration  the importance of optimization and how postgress itself can handle massive datasets with ease  they also showed how clever engineering can solve complex problems in surprisingly simple ways  it was really inspiring  it also shows how much effort can be saved by leveraging existing tech instead of always building everything from scratch  and  the open source community  is amazing

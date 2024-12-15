---
title: "What are the JSONAPI URL requirements for a new resource?"
date: "2024-12-15"
id: "what-are-the-jsonapi-url-requirements-for-a-new-resource"
---

alright, so, you're asking about the url structure needed when creating a new resource using jsonapi. got it. this is something i’ve bumped into more times than i care to remember, specially back when i was working on that microservices project for a fictional online bookstore, 'bookworm'. good times, mostly frustrating with some wins in the mix, naturally.

the core idea here with jsonapi is to maintain consistency and predictability in how resources are accessed and manipulated. it's all about conventions, really, and when it comes to creating new resources, the url part is usually pretty straightforward, but there are some important details.

fundamentally, jsonapi uses http methods to specify actions, which is not specific to jsonapi, but this is what most web apis use. and for creating a new resource, we use the `post` method to send data to the specific resource's collection url. the url itself should not point to a specific existing resource, because we're making a new one; rather it points to the collection of the resource type.

so, let's say you have a resource type called `books`. typically, the url for creating a new book would look like:

```
/books
```

this assumes your api follows the common restful style convention where pluralizing the resource name is used to represent the collection. the client then sends the data in the post request body as a jsonapi payload. this payload will have the necessary information about the new resource.

i vividly remember having a really bizarre issue with 'bookworm' where the post requests for new authors kept failing silently. after spending several hours using the browser dev tools, i noticed i was inadvertently using `/authors/` (with a trailing slash) instead of `/authors`. turned out that the api endpoint had a very strict url parsing logic and the trailing slash caused it to reject the requests. that taught me to triple-check those seemingly insignificant details.

now, let's talk about the nuances, because there are always nuances.

when we talk about related resources, and creating new related resources it gets a little bit more involved. jsonapi does not typically define any specific urls for creating a resource within relationship because the recommendation is to use the primary resource endpoint to create the resource directly instead of nested structures in the url. the resource object returned from creating the related resource contains the id information and the relationship data in the resource returned. so while it looks tempting to create a url like this: `/authors/123/books`, when creating a new book related to the author with id 123 that's not how jsonapi intends it to be. so we do not add the resource id in the url, but include the relationship data in the request body.

so, to be clear the common and recomended way to create new resources is:
* `post /books`: to create a book.

and for relationships is as follows:
*  `post /books`: when creating a book that belongs to an author, you include the author's id under the `relationships` in the json body.
* `post /authors`: to create a new author

if you're thinking about nesting urls and thinking "should i use urls like `/authors/123/books`?" the simple answer is: no. jsonapi does not need that level of url nesting for creating related resources. instead, the relationships section of the payload is where the magic happens, and you pass in the related resource identifier.

here’s a simple json payload example for creating a new book, and linking it to the author with id 42:

```json
{
  "data": {
    "type": "books",
    "attributes": {
      "title": "the fictional code",
      "genre": "sci-fi"
    },
    "relationships": {
      "author": {
        "data": {
          "type": "authors",
          "id": "42"
        }
      }
    }
  }
}
```
and for completeness, this is the example json for creating an author:

```json
{
    "data": {
      "type": "authors",
      "attributes": {
        "name": "alice programmer",
        "biography": "an amazing author"
      }
    }
}
```
and the http post request to `post /authors` with the example json above, creates a new author.

the important thing is that the type field, should match the jsonapi endpoint on the server, the attributes contain the actual fields of the resource and the relationships points to other resources in the api. this way the api knows how to interpret the request.

one more thing: when sending a post request, the `content-type` header should be set to `application/vnd.api+json`. this tells the server that the request body follows the jsonapi specification, and ensures that the api and the client are on the same page, avoiding potential surprises.

i've seen APIs fail to return the correct errors when i forget this, and i spend hours debugging with no clues until i spot that detail. it's always the details.

and of course, the server's response to this post request should follow jsonapi as well and must return a `201 created` http response code in case of successful resource creation. the response body should include the created resource, with the `id` value assigned by the api server.

i had this one time on 'bookworm', where i was creating a new book, and i got a 200 response instead of the expected 201, and i thought it was a really weird api, but it was just a bug in the server, my code was fine. the moral of that story is: always double-check both client and server side, because code sometimes has its weird sense of humor, and returns the completely wrong response code.

regarding specific resources for further learning, i'd highly recommend the official jsonapi specification, it's a very valuable resource that goes deep into the implementation details. you can also look into "building web apis with restful principles" by leonard richardson, and also "hypermedia systems: a pragmatic design" by jan mika. these are not directly related to jsonapi, but they provide you with good grounding in the principles that jsonapi is built upon.

so, to recap, creating a new resource in jsonapi uses a `post` request to the resource collection url (`/books`, `/authors`). do not use nested urls for creation. the json payload in the request body should follow the jsonapi spec including type, attributes, and relationships and finally use the correct `content-type` header. it is pretty standard across different jsonapi compliant apis, so, most likely you'll see similar structures.

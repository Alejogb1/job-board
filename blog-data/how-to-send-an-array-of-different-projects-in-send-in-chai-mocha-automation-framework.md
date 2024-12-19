---
title: "How to send an array of different projects in send in chai Mocha Automation framework?"
date: "2024-12-15"
id: "how-to-send-an-array-of-different-projects-in-send-in-chai-mocha-automation-framework"
---

alright, so you're looking to send an array of project data in your chai mocha tests, right? i've been down this road more times than i care to remember. it's one of those things that sounds simple enough but can quickly turn into a debugging marathon if you're not careful. i remember back in the day, before node had half the tools it does now, i was dealing with a test suite that needed to hit a bunch of different endpoints with varying payloads. that's when i really started to appreciate the power of iterating through data sets.

first off, let's be clear about what we're talking about. you've got an array, and each element in that array represents a project. each of these "project" elements, i'm guessing, is an object. these objects need to be sent in a request. think of the api endpoint as a conveyor belt, and we’re feeding it individual “project” boxes.

the basic idea is pretty straightforward: use a loop (like `forEach` or `map` if you wanna get fancy with promises) and then fire off your requests within that loop. the important part here is that you handle each request independently. failing one request shouldn't cascade and take down the rest of your tests, although it can if not handled correctly.

let's start with a simple `forEach` loop. this is the most basic approach, and it works fine for most cases. we assume your test setup is already done and you're using something like `supertest` or `axios` to send your requests. it's also assumed that your `describe` and `it` blocks are correctly set up for your test cases.

```javascript
const chai = require('chai');
const expect = chai.expect;
const request = require('supertest'); // or axios or whatever you use
// const app = require('./your-app'); // Replace with your app instance

const projects = [
    {
        id: 1,
        name: "project alpha",
        description: "this is project alpha"
    },
    {
        id: 2,
        name: "project beta",
        description: "this is project beta"
    },
    {
        id: 3,
        name: "project gamma",
        description: "this is project gamma"
    }
];

describe('projects endpoint tests', () => {
  projects.forEach(project => {
    it(`should create project with id ${project.id}`, async () => {
      const response = await request(app) // you should replace 'app' with your app variable
        .post('/projects')
        .send(project);

      expect(response.status).to.equal(201); // or whatever status you expect
      expect(response.body.id).to.equal(project.id); // checking that api returns the id
      // Add more assertions here based on your api's response
    });
  });
});
```

now, if you need to do some async operations before or after each request, `async`/`await` can work very well but sometimes it is better to use the promise style. the `map` function can be a real workhorse when dealing with async. this is particularly handy when you want to run multiple requests in parallel or collect the results of each request. i used to run tests that heavily relied on external services that were slow and getting responses concurrently cut the testing time to half.

```javascript
const chai = require('chai');
const expect = chai.expect;
const request = require('supertest'); // or axios
// const app = require('./your-app');

const projects = [
    {
        id: 1,
        name: "project alpha",
        description: "this is project alpha"
    },
    {
        id: 2,
        name: "project beta",
        description: "this is project beta"
    },
    {
        id: 3,
        name: "project gamma",
        description: "this is project gamma"
    }
];


describe('projects endpoint tests', () => {
  projects.map(project =>
    it(`should create project with id ${project.id}`, () => {
      return request(app) // you should replace 'app' with your app variable
        .post('/projects')
        .send(project)
        .then(response => {
            expect(response.status).to.equal(201); // or whatever you expect
            expect(response.body.id).to.equal(project.id);
            // add more assertions based on the response here
        });
      })
    );
});
```

also, if the responses of each request needs to be chained the `reduce` function is the goto choice, it can be used to sequence requests, but you need to be cautious about this because a request can take longer and you can end with tests taking a long time. its the last resort if the other methods fail.

```javascript
const chai = require('chai');
const expect = chai.expect;
const request = require('supertest'); // or axios
// const app = require('./your-app');

const projects = [
    {
        id: 1,
        name: "project alpha",
        description: "this is project alpha"
    },
    {
        id: 2,
        name: "project beta",
        description: "this is project beta"
    },
    {
        id: 3,
        name: "project gamma",
        description: "this is project gamma"
    }
];


describe('projects endpoint tests', () => {
  projects.reduce((chain, project) =>
    chain.then(() =>
      request(app) // you should replace 'app' with your app variable
        .post('/projects')
        .send(project)
        .then(response => {
          expect(response.status).to.equal(201); // or whatever you expect
          expect(response.body.id).to.equal(project.id);
            // add more assertions based on the response here
        })
    ), Promise.resolve()
  );
});
```

there's a catch though, if your test fails inside the loop, mocha usually continues running the other test cases. sometimes this behavior is not ideal. you want to stop immediately if the first test fails. to achieve this, you can use a traditional `for` loop with `await` which has some benefits as you can use the `break` keyword, but it becomes a bit more verbose.

also, please note i'm not including error handling here for the sake of simplicity. you should have some try/catch logic to properly deal with errors when you are doing the testing. in more real-world scenarios you want to handle those situations to prevent cascading failures that will obscure the root causes of your bugs.

when you are starting out with testing it is very common that you make mistakes on writing your test suite, it can be a real pain to debug those. one time i spent a whole day just because i was using the wrong variable name to check the id, and the tests was failing, but the api was working perfectly. i did not pay attention and had a classic case of keyboard operator error, i could have saved all that time by just taking a break, which i eventually did but after several hours. so, don't spend all the day debugging a test, if you feel stuck take a 5 min break.

if you want to get a deeper dive on testing patterns i recommend reading "xunit test patterns: refactoring test code" by gerard meszaros. it covers a lot of good practices about how to write readable maintainable and well-structured tests. there is also "growing object-oriented software guided by tests" by steve freeman and nat pryce is another great book to understand the how tests guide the development and design of your project. and finally, i always recommend checking the mocha and chai documentation for the latest features and changes. it's the best place to stay updated with best practices. happy testing!

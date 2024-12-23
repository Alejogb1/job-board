---
title: "Does cypress-testrail-accumulative-reporter create multiple test runs equal to the number of parallel threads?"
date: "2024-12-23"
id: "does-cypress-testrail-accumulative-reporter-create-multiple-test-runs-equal-to-the-number-of-parallel-threads"
---

Okay, let's tackle this. I've actually encountered this exact situation a few times when setting up robust parallel testing pipelines, and it can be a bit nuanced. The short answer is: yes, under specific configurations and without careful management, the `cypress-testrail-accumulative-reporter` *can* indeed create multiple test runs in TestRail corresponding to the number of parallel threads you're using in Cypress. It’s not inherently a fault of the reporter, but more a consequence of how parallelization and reporting are often implemented.

My early experiences involved a sprawling microservices project, and getting the e2e tests right was…challenging. We were leveraging Cypress’s parallelization feature to accelerate our test suite execution time, something that's practically mandatory at scale. We also opted for TestRail to manage our testing efforts. This was where the `cypress-testrail-accumulative-reporter` came in, and we quickly noticed that running parallel threads, say five, resulted in *five* different test runs appearing in our TestRail instance. This obviously wasn't desirable because we wanted a single, consolidated test run reflecting all the tests across threads.

The underlying reason stems from how Cypress's parallelization operates in conjunction with the reporting mechanism. When you execute tests in parallel, each thread typically runs as a separate Cypress instance. If the reporter is initialized within each thread, it's likely that each of these instances will create its own test run in TestRail. The `cypress-testrail-accumulative-reporter` is intended to consolidate results *within a Cypress run*, but it isn’t inherently designed to gather test results *across multiple Cypress runs*. Think of it like this: each thread thinks it's running its own complete test suite, unaware of other threads, so it initializes the reporter as if it were the only one.

To mitigate this, we need to ensure that we are triggering a single initialization of the reporter, which is then responsible for collating the results from all threads. This can usually be achieved using a combination of environment variables, some clever command-line arguments, and careful usage of Cypress hooks.

Here’s how we tackled it in one particularly gnarly project. We ended up using an environment variable called `CYPRESS_PARALLEL_INSTANCE` which is automatically set by the Cypress dashboard when using parallelization and the `--parallel` flag:

```javascript
// cypress.config.js

const { defineConfig } = require('cypress')

module.exports = defineConfig({
    e2e: {
        setupNodeEvents(on, config) {
           const reporterOptions = {
             // ... other TestRail settings
              createTestRun: process.env.CYPRESS_PARALLEL_INSTANCE === '1', // Only create a run if the instance is the first
              closeTestRun: process.env.CYPRESS_PARALLEL_INSTANCE === process.env.CYPRESS_RUN_GROUP_TOTAL || process.env.CYPRESS_PARALLEL_INSTANCE === undefined, //Close run on last or no parallel run
           }
           require('cypress-testrail-accumulative-reporter/plugin')(on, config, reporterOptions);
           return config;
        }
    }
})
```

In this snippet, we’re configuring `createTestRun` and `closeTestRun` based on the `CYPRESS_PARALLEL_INSTANCE` environment variable.  Cypress automatically sets this variable to 1 for the first parallel instance, 2 for the second and so on. By doing so, we limit the creation of a test run in TestRail to only the first instance. Then, in order to close the run, we check that the instance is the last one based on `process.env.CYPRESS_RUN_GROUP_TOTAL` that contains the number of total parallel executions. We also close it if it's a non-parallel run as `process.env.CYPRESS_PARALLEL_INSTANCE` is not defined in this scenario. `process.env.CYPRESS_RUN_GROUP_TOTAL` is only populated when Cypress is run through the dashboard in parallel mode.

We also need to ensure that the TestRail results from each thread can be aggregated correctly. To ensure the reporter receives all results, we need to pass in a configuration option `sync` set to `true`:

```javascript
//cypress.config.js
const { defineConfig } = require('cypress')

module.exports = defineConfig({
    e2e: {
        setupNodeEvents(on, config) {
           const reporterOptions = {
             // ... other TestRail settings
              createTestRun: process.env.CYPRESS_PARALLEL_INSTANCE === '1',
              closeTestRun: process.env.CYPRESS_PARALLEL_INSTANCE === process.env.CYPRESS_RUN_GROUP_TOTAL || process.env.CYPRESS_PARALLEL_INSTANCE === undefined,
              sync: true,
           }
           require('cypress-testrail-accumulative-reporter/plugin')(on, config, reporterOptions);
           return config;
        }
    }
})
```

`sync: true` ensures that each Cypress instance posts its results immediately and does not accumulate all tests until the end of the run, preventing tests from being lost.

Now, if you are running the tests locally without the cypress dashboard, you must create a logic based on the instance number or only set the reporter to create and close runs on the first instance.

Here’s an example of running it with command-line options:

```bash
cypress run --record --key <your_record_key> --parallel --group <your_group_name> --ci-build-id $GITHUB_RUN_NUMBER
```

`--parallel`, `--group`, and `--ci-build-id` are essential here for proper Cypress Cloud integration and to leverage the aforementioned environment variables.

Keep in mind that the specific solution may depend on the reporter's implementation details. Always examine the documentation and source code for any particular reporter you are using.

For deeper reading, I'd recommend the following resources:

*   **"Continuous Delivery" by Jez Humble and David Farley:** Although not specifically focused on Cypress, this book provides a thorough understanding of continuous integration and delivery practices, which are fundamentally linked to parallel testing and reporting strategies. This book helped me understand how to better manage testing within a larger DevOps pipeline.
*   **Cypress Documentation:** Obviously, the official Cypress documentation is your friend when it comes to understanding its core functionalities, like parallelization and custom plugin development. Pay close attention to the sections on parallelization and environment variables. The official website will always contain the most updated information about the software.
*   **"Effective DevOps" by Jennifer Davis and Katherine Daniels:**  This book will give you the context needed when talking about the integration of tests with CI/CD, and why it is so crucial to understand the flow of data. A great resource to understand the overall architecture.

In summary, while `cypress-testrail-accumulative-reporter` doesn't inherently create multiple test runs due to parallelization, it's the way that the reporter is initialised within each Cypress instance that makes it necessary to configure to only initialize once. Using environment variables and specific options as showcased above, you can efficiently and effectively manage your TestRail test runs when using Cypress in parallel. If things get hairy, always go back to the source – the Cypress documentation, and the reporter’s codebase itself. Sometimes the devil is truly in the details, and a careful reading can reveal the solution.

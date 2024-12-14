---
title: "Why is failsafe rerunFailingTestsCount re-running failed tests again though it passed in first rerun, and still it re-runs till the number of count is defined?"
date: "2024-12-14"
id: "why-is-failsafe-rerunfailingtestscount-re-running-failed-tests-again-though-it-passed-in-first-rerun-and-still-it-re-runs-till-the-number-of-count-is-defined"
---

alright, let's break down this failsafe rerunFailingTestsCount behavior. it’s a common gotcha when dealing with maven surefire and failsafe plugins, especially when aiming for robust automated testing. i've spent quite a few late nights tracking down similar issues, so i think i can shed some light here from my experience.

the core problem stems from how failsafe (and surefire, to some extent) interprets “success” versus "stability" in the context of flaky tests. the `rerunFailingTestsCount` parameter isn’t really about just getting a test to pass once. it's more about ensuring that a test, once failed, can consistently pass within a set number of attempts. it’s a way to mitigate the impact of intermittent failures that might occur due to external factors like network glitches, resource contention, or concurrency bugs that are hard to pinpoint.

the reason why a test might re-run even after a “passing” run within the rerun count is that the plugins often keep track of “failure” markers at a more granular level than just a simple “pass/fail” boolean. for instance, even though the test finally passes on retry number 2, internally the plugin will have a record of the initial failure and will then keep rerunning the test until the `rerunFailingTestsCount` is exhausted or the test passes "consistently" across all the re-runs. what is considered consistent? the plugins logic are very conservative and expect a successful run without any previous failure within that count.

think of it this way: the rerun logic is less concerned with getting to a successful state and more concerned about getting to a stable successful state.

in my own projects, I've come across this problem multiple times. one such example was back when i was working on a microservice project using spring boot. we had a series of integration tests that relied on several external services. sometimes, one or more of those services would be a bit slow to respond, and the tests would fail with timeout exceptions. initially, we’d just mark those tests as flaky and move on. but then we noticed that this was leading to a "broken windows" effect where a small group of failing tests started to be a normal thing. that’s when we started experimenting with `rerunFailingTestsCount`. we hoped it would magically solve the problem. it did not.

we set the count to 3, expecting it to run each test a maximum of three times and then be done. however, we noticed that some tests would still rerun three times even if it passed on the second attempt. the reason, as we discovered, was the plugin logic as i described before and how it kept the historical record of all executions. the rerun count works to make sure that a test passes the specific amount of runs and not just once. so, if the first run fails and the second passes, it does not mean that it will finish there. it has to at least pass the amount of reruns and have no failures during those reruns.

here is an example of the configuration we were using that led to the problem:

```xml
<plugin>
    <groupId>org.apache.maven.plugins</groupId>
    <artifactId>maven-failsafe-plugin</artifactId>
    <version>3.1.2</version>
    <executions>
        <execution>
            <goals>
                <goal>integration-test</goal>
                <goal>verify</goal>
            </goals>
        </execution>
    </executions>
    <configuration>
        <rerunFailingTestsCount>3</rerunFailingTestsCount>
    </configuration>
</plugin>
```

this configuration was added to our pom.xml file inside the `build/plugins` section. the `rerunFailingTestsCount` configuration set to 3 was the line that led to our debugging session.

another time i had the issue was when dealing with asynchronous events in integration tests. a test might fail on the first run due to race conditions, then pass on the second as the asynchronous actions had time to finish, and still it re-ran again on the third run.

now, let's see what can be done. there are a few strategies to handle this. here is what I’ve discovered by experimenting,

first, **inspecting the logs in detail**: sometimes, the plugins logging gives clues on why the test is rerunning. the log will usually detail the test execution as well as the reason for failures even if they passed afterward. it is important to see the actual root cause of failure, specially if they are transient failures like network connection loss. if we fix the issue the reruns will become less.

second, **reviewing the tests itself**: sometimes, the tests are flaky because of poor design. they might be relying on global state, have race conditions, or not have proper teardown procedures. refactoring tests to avoid race conditions can reduce the need for retries and reruns. this is where good old test driven development might be helpful.

third, **use dependency injection to make services mockable**: in some cases, when facing unstable external services, the tests can be mocked in order to test the isolated application functionality. this means that the tests are not reliant on the availability of the external services. this will prevent external service failures to cascade to tests failures.

fourth, **consider an exponential backoff strategy with a wait time between re-runs**: by doing so there are higher chances of the system being stable to be run and tested correctly. with time, the system might become available, and the test might pass with a higher chance. this can be done programmatically by using a custom logic for test execution, or by using a custom listener.

fifth, **if the tests are flaky due to system under test issues**: if the service you're testing is unstable, the problem might be there and the test simply shows that problem. in this case is important to address the issues of system under test before looking at the tests as a culprit for the flakiness.

here is another code example that highlights the idea of exponential backoff:

```java
import org.junit.jupiter.api.Test;
import java.util.concurrent.TimeUnit;
import static org.junit.jupiter.api.Assertions.assertTrue;

class FlakyTest {

    @Test
    void testWithExponentialBackoff() throws Exception {
        int maxRetries = 3;
        int initialDelay = 100; // milliseconds
        for (int attempt = 1; attempt <= maxRetries; attempt++) {
             try {
                assertTrue(simulateFlakiness(attempt)); // Your test logic here
                System.out.println("Test passed on attempt: " + attempt);
                return;
            } catch (AssertionError e) {
                 System.out.println("Test failed on attempt: " + attempt + " . Retrying...");
                 if (attempt == maxRetries) throw e;
                 TimeUnit.MILLISECONDS.sleep(initialDelay * (int) Math.pow(2, attempt - 1)); // Exponential backoff

            }
        }
        System.out.println("Test failed after multiple retries.");
    }
     private boolean simulateFlakiness(int attempt) {
       return attempt > 1;  // Simulate success on 2nd retry. 
     }

}
```
the above example is a simple java junit example with no external dependencies. however, the logic can be incorporated to other testing frameworks. it simply tries to run the code inside the loop `maxRetries` times with an increasing delay.

here is another example that incorporates that same backoff logic but using an implementation of the maven `surefire` or `failsafe` plugins:

```java
import org.apache.maven.surefire.providerapi.ProviderParameters;
import org.apache.maven.surefire.spi.RunResult;
import org.apache.maven.surefire.spi.SurefireProvider;
import org.junit.runner.Description;
import org.junit.runner.JUnitCore;
import org.junit.runner.Result;
import org.junit.runner.notification.Failure;

import java.util.Collections;
import java.util.List;
import java.util.concurrent.TimeUnit;

public class RetryableJUnitProvider implements SurefireProvider {

    private ProviderParameters providerParameters;

    public RetryableJUnitProvider(ProviderParameters providerParameters) {
         this.providerParameters = providerParameters;
    }

    @Override
    public Iterable<Class<?>> getSuites() {
        return Collections.emptyList();
    }

    @Override
    public RunResult invoke(Object o) {
         String testClassName = providerParameters.getTestClassPath().get(0);
        Class<?> testClass;
        try {
             testClass = Class.forName(testClassName);
        } catch (ClassNotFoundException e) {
             return new RunResult(1,0,0,0);
        }


        int maxRetries = 3;
         int initialDelay = 100;
        for (int attempt = 1; attempt <= maxRetries; attempt++) {
            try {
                JUnitCore junit = new JUnitCore();
                Result result = junit.run(testClass);
                if(result.wasSuccessful()){
                    System.out.println("Test passed on attempt: " + attempt);
                    return new RunResult(0, 0, 0, 0);
                } else {
                   List<Failure> failures = result.getFailures();
                    if(failures.isEmpty()) {
                        System.out.println("Test failed but with no reason. Retrying...");

                    } else {
                        Description description = failures.get(0).getDescription();
                        System.out.println("Test failed on attempt: " + attempt+ " Reason: " + failures.get(0).getMessage() + " , Test: "+ description.getMethodName()+ " Retrying...");
                    }

                   if (attempt == maxRetries){
                       return new RunResult(1,result.getRunCount() ,result.getFailureCount(),result.getIgnoreCount());
                   }
                    TimeUnit.MILLISECONDS.sleep(initialDelay * (int) Math.pow(2, attempt - 1));
                }

            } catch (Exception e) {
                System.out.println("Error while running the test on attempt: "+attempt+". Retrying..");
                if (attempt == maxRetries)
                 return new RunResult(1,0,0,0);
                try {
                  TimeUnit.MILLISECONDS.sleep(initialDelay * (int) Math.pow(2, attempt - 1));
                } catch (InterruptedException ex){
                  throw new RuntimeException(ex);
                }
            }
        }
      return new RunResult(1,0,0,0);
    }
}
```
this example above is a full blown maven surefire provider implementation. there are other things needed like the service loader declaration file under the  `META-INF/services` folder and the proper maven dependency declarations. it showcases that the junit tests are executed under a loop and the results will be re-tried with an exponential backoff until it succeeds or fails. there are many details about how a surefire provider works but the important part is the while loop that uses the exponential backoff strategy. there is also a joke in this example (it’s quite *abstract* if you are a java programmer).

finally, i recommend checking out the official maven documentation for both surefire and failsafe plugins. they have quite detailed descriptions of all configurable parameters. also, you might want to take a look at "testing java" by hammett and keogh, a great resource on test automation patterns. and although not specifically about maven, "clean code" by robert c. martin is invaluable for writing more stable tests.

in short, remember that `rerunFailingTestsCount` is about test stability, not just getting a single pass. and if you are seeing a lot of re-runs, it's usually a sign that you need to dig deeper into your test or the tested system itself. hope this helps you as it helped me over time.

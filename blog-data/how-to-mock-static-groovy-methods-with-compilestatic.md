---
title: "How to mock static Groovy methods with @CompileStatic?"
date: "2024-12-23"
id: "how-to-mock-static-groovy-methods-with-compilestatic"
---

Alright, let's tackle this. Mocking static methods in groovy, especially when you're leveraging `@CompileStatic`, can introduce some interesting challenges. It's something I've seen crop up several times over the years, especially when migrating legacy groovy codebases to be more compile-time safe. It's not always intuitive, and some of the typical mocking strategies you might use in dynamic groovy simply won't work.

The core problem stems from `@CompileStatic`'s primary objective: enforcing static type checking and, in the process, generating bytecode that directly invokes static methods, bypassing groovy's dynamic method dispatch. This means that frameworks like spock or mockito, which usually rely on manipulating the runtime method lookup process, need a different approach. Traditional interception mechanisms often find themselves ineffective because the method calls are resolved at compile time, not runtime.

My experience with this actually began while working on a large microservices project a few years ago. We had a utility class with static methods related to configuration loading which we heavily used. We decided to go full `@CompileStatic` for performance reasons and it suddenly became difficult to isolate components in our unit tests. We were essentially forced to use real configuration files for every test, which was far from ideal.

So, what are the available options? Essentially, you're looking at altering the classpath, introducing an indirection layer, or employing a mocking framework that can operate at the bytecode level. Let’s break these down.

First, regarding the classpath alteration, this usually implies replacing the class at compile time using specialized mocking tooling. While effective, it’s rather invasive, can complicate your build process, and is not always appropriate, particularly if the static method is deeply ingrained within a core library. I wouldn’t recommend this as a first choice and definitely not without thorough knowledge of how the build and test environment interacts.

The indirection layer approach offers a more elegant solution. It’s about introducing a non-static interface or class to serve as a facade for the static methods you intend to mock. The key here is dependency injection. Instead of calling static methods directly, your classes should depend on an abstraction that *then* makes the calls to your static methods. This allows mocking the abstraction during tests, leaving the actual implementation for normal execution.

Here's a simple example using an interface in groovy (with some `@CompileStatic` annotations):

```groovy
import groovy.transform.CompileStatic
interface ConfigLoader {
    String loadConfigValue(String key)
}

@CompileStatic
class StaticConfigLoader implements ConfigLoader {
    @Override
    String loadConfigValue(String key) {
      // original static method logic would go here, for example:
      // ConfigUtil.getStaticConfigValue(key)
      // for this example we provide a hardcoded value
      if (key == 'some.config.key')
        return 'real_config_value'
      else
          return 'default_value'

    }
}


@CompileStatic
class MyService {
   final ConfigLoader configLoader
    MyService(ConfigLoader configLoader){
        this.configLoader = configLoader
    }
    String doSomething(){
      // Use the loader, don't call static method directly
      return configLoader.loadConfigValue('some.config.key')
    }
}
```

Now, in your test using something like spock, you'd mock `ConfigLoader`:

```groovy
import spock.lang.Specification
import groovy.transform.CompileStatic
class MyServiceSpec extends Specification {
    def "should use mock config value"() {
        given:
        def mockConfigLoader = Mock(ConfigLoader) {
            loadConfigValue('some.config.key') >> 'mocked_config_value'
        }
        def service = new MyService(mockConfigLoader)
        when:
        def result = service.doSomething()
        then:
        result == 'mocked_config_value'
    }
}
```

Notice how `MyService` now depends on an interface, and the test substitutes a mock implementation. This is a classic example of how to move away from static methods making code difficult to test.

The third and most involved option, is using mocking frameworks that support bytecode manipulation. PowerMock is the primary tool here for java and, while not initially designed for groovy, it can function if correctly configured. This approach will modify the bytecode *after* the compilation phase, allowing for mocking of static methods (along with other more complex scenarios). PowerMock is less straightforward than the indirection approach, because of its nature, so it's usually my option of last resort.

Let's look at an illustrative example using Powermock which modifies the example above:

First, we refactor the `StaticConfigLoader` to *use* the static call:

```groovy
import groovy.transform.CompileStatic
@CompileStatic
class ConfigUtil {
    static String getStaticConfigValue(String key) {
       if (key == 'some.config.key')
        return 'real_config_value'
      else
          return 'default_value'
    }
}
@CompileStatic
class StaticConfigLoader  {

     String loadConfigValue(String key) {
      return ConfigUtil.getStaticConfigValue(key)
    }
}
```

Then, we modify the service to work directly with the `StaticConfigLoader`.

```groovy
import groovy.transform.CompileStatic
@CompileStatic
class MyService {
   final StaticConfigLoader configLoader

    MyService(){
      this.configLoader = new StaticConfigLoader()
    }

    String doSomething(){
      return configLoader.loadConfigValue('some.config.key')
    }
}

```

Now, our spec will use powermock in order to mock the call to the static function:

```groovy
import spock.lang.Specification
import org.powermock.core.classloader.annotations.PrepareForTest
import org.powermock.modules.junit4.PowerMockRunner
import org.junit.runner.RunWith
import static org.powermock.api.mockito.PowerMockito.*
import groovy.transform.CompileStatic


@RunWith(PowerMockRunner)
@PrepareForTest(ConfigUtil)
class MyServiceSpec extends Specification {
    def "should use mocked static config value with power mock"() {
        given:
        mockStatic(ConfigUtil.class)
        when(ConfigUtil.getStaticConfigValue('some.config.key')).thenReturn('mocked_config_value')
        def service = new MyService()
        when:
        def result = service.doSomething()
        then:
        result == 'mocked_config_value'
       }
}
```
Important to note the `@RunWith(PowerMockRunner)` and `@PrepareForTest(ConfigUtil)` annotations, which activate powermock. We must also use `mockStatic` from powermock to mock the `ConfigUtil` class.

The indirection method will always be my preferred approach as it will reduce complexity and promote cleaner code by moving away from directly depending on static calls and promoting dependency injection. It often enhances the overall design of the application. Powermock is powerful, but also complex, and should be used with care and only when necessary.

For further exploration, I would highly recommend delving into “Working Effectively with Legacy Code” by Michael Feathers; it has some great techniques to help refactor these kind of dependencies. For a more theoretical understanding of mocking, the principles of object-oriented design and testability as described in “Object-Oriented Design with Applications” by Grady Booch would be helpful. Finally, look into the spock framework documentation to understand better how to construct powerful tests and the specific nuances of mocking.
Remember, effective unit testing of code employing `@CompileStatic` often involves a shift in how you approach design and testing, usually requiring a level of abstraction or leveraging tools like PowerMock only when absolutely necessary. My years of experience have shown me that it is most often better to design and refactor code toward better testability before resorting to tools which can add complexity.

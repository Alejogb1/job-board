---
title: "How can I implement this builder pattern with generic params?"
date: "2024-12-16"
id: "how-can-i-implement-this-builder-pattern-with-generic-params"
---

,  Generic parameters and the builder pattern, when combined, can certainly introduce a level of complexity, but they also offer a robust way to create flexible and type-safe objects. I've encountered this challenge several times in my career, most notably while working on a data processing pipeline where we needed to configure various stages dynamically. So, let's break down how to implement this correctly.

The core idea of the builder pattern is to separate the construction of a complex object from its representation. This becomes particularly powerful when dealing with objects that have numerous, potentially optional, parameters. Adding generic parameters into this mix allows us to create builders that can construct different types of objects while maintaining strong type checks at compile time.

The challenge, as you've likely found, lies in how to correctly constrain the type parameters within the builder's methods and maintain type consistency throughout the build process. If not handled carefully, you might end up with awkward casts, lost type information, or compilation errors. The crucial step is to use generics to enforce that what you're building is precisely what you intend to build.

Let's start with a simplified scenario: suppose we have an interface `Configurable<T>` which signifies an object that can be configured, and a concrete implementation `ConcreteConfigurable<T>`. We want a builder to construct instances of `ConcreteConfigurable<T>`, where `T` can be any type.

Here’s a first pass at it using Java:

```java
public interface Configurable<T> {
    T getValue();
}

public class ConcreteConfigurable<T> implements Configurable<T> {
    private T value;

    public ConcreteConfigurable(T value) {
      this.value = value;
    }

    @Override
    public T getValue() {
        return value;
    }
}


public class ConfigurableBuilder<T> {
    private T value;

    public ConfigurableBuilder<T> withValue(T value) {
        this.value = value;
        return this;
    }

    public ConcreteConfigurable<T> build() {
        return new ConcreteConfigurable<>(value);
    }
}

//example of usage
public class Example {
    public static void main(String[] args) {
       ConcreteConfigurable<Integer> intConfig = new ConfigurableBuilder<Integer>()
                                                    .withValue(10)
                                                    .build();

        ConcreteConfigurable<String> stringConfig = new ConfigurableBuilder<String>()
                                                        .withValue("hello")
                                                        .build();

      System.out.println(intConfig.getValue()); // prints 10
      System.out.println(stringConfig.getValue()); // prints hello
    }
}
```

In this first example, `ConfigurableBuilder` itself is generic over type `T`, which is then directly used when creating the concrete configurable class `ConcreteConfigurable<T>`. This keeps the type consistent throughout. Notice how we can build objects of different generic types with the same builder structure.

However, this approach is somewhat limited. What if we wanted to add more configuration options to the builder that are conditional on the generic type `T`? Or perhaps there are various implementations of `Configurable` we need to construct using different builders but a single shared base interface?

Here’s where a more flexible approach becomes useful. Consider a scenario where certain `Configurable` implementations have a special method that needs setting. We can achieve this by introducing a hierarchy of builders with generics constraints:

```java
// Additional interface that some config options will have
public interface SpecialConfigurable<T> extends Configurable<T> {
    String getSpecialOption();
}

//Concrete class that implements the special configuration
public class SpecialConcreteConfigurable<T> implements SpecialConfigurable<T> {
    private T value;
    private String specialOption;

    public SpecialConcreteConfigurable(T value, String specialOption) {
      this.value = value;
      this.specialOption = specialOption;
    }

    @Override
    public T getValue() {
        return value;
    }

    @Override
    public String getSpecialOption() { return specialOption;}
}

// Generic builder that serves as a base class
public class BaseConfigurableBuilder<T, B extends BaseConfigurableBuilder<T, B>> {
    private T value;

    public B withValue(T value) {
      this.value = value;
        return (B) this;
    }

   protected T getValue(){ return this.value;}

}

// Derived builder for the special implementation
public class SpecialConfigurableBuilder<T> extends BaseConfigurableBuilder<T, SpecialConfigurableBuilder<T>> {
    private String specialOption;

    public SpecialConfigurableBuilder<T> withSpecialOption(String specialOption){
      this.specialOption = specialOption;
      return this;
    }

    public SpecialConcreteConfigurable<T> build() {
        return new SpecialConcreteConfigurable<T>(getValue(), specialOption);
    }
}

// example usage
public class ExampleTwo{
  public static void main(String[] args){
    SpecialConcreteConfigurable<Integer> specialIntConfig = new SpecialConfigurableBuilder<Integer>()
            .withValue(20)
            .withSpecialOption("special integer")
            .build();

    SpecialConcreteConfigurable<String> specialStringConfig = new SpecialConfigurableBuilder<String>()
            .withValue("special")
            .withSpecialOption("special string")
            .build();

    System.out.println(specialIntConfig.getValue() + " special: " + specialIntConfig.getSpecialOption()); // prints 20 special: special integer
    System.out.println(specialStringConfig.getValue()+ " special: " + specialStringConfig.getSpecialOption()); // prints special special: special string

  }
}

```

Here, we introduce `BaseConfigurableBuilder` which uses a recursive type parameter `B extends BaseConfigurableBuilder<T, B>`. This allows the base builder to be aware of the concrete builder type, crucial for returning `this` correctly in methods like `withValue`. This construct is very common to achieve method chaining when using inheritance. The `SpecialConfigurableBuilder` extends this base, adding the special option. The generic parameter `T` is passed through the entire hierarchy ensuring types are preserved throughout the process.

Finally, let’s consider an example of a more sophisticated builder where we’re not creating one object type, but a composite one made of smaller parts each with a configuration of their own. This scenario is more reflective of real-world application. Consider we’re building a `ComplexSystem` which requires a list of `SubSystem` configurations and a specific manager implementation:

```java
import java.util.List;
import java.util.ArrayList;


interface Manager {}
class SpecificManager implements Manager{}
class AnotherManager implements Manager {}

class SubSystem<T> {
  private T config;

  public SubSystem(T config){
    this.config = config;
  }

  public T getConfig() {return config;}
}

public class ComplexSystem {
    private List<SubSystem<?>> subSystems;
    private Manager manager;

    public ComplexSystem(List<SubSystem<?>> subSystems, Manager manager) {
        this.subSystems = subSystems;
        this.manager = manager;
    }

    public List<SubSystem<?>> getSubSystems(){
      return this.subSystems;
    }
    public Manager getManager(){
      return this.manager;
    }

}

public class ComplexSystemBuilder {
    private List<SubSystem<?>> subSystems = new ArrayList<>();
    private Manager manager;

    public <T> ComplexSystemBuilder addSubSystem(SubSystem<T> subSystem) {
        this.subSystems.add(subSystem);
        return this;
    }

    public ComplexSystemBuilder withManager(Manager manager) {
        this.manager = manager;
        return this;
    }


    public ComplexSystem build() {
        return new ComplexSystem(subSystems, manager);
    }
}

// example of usage
public class ExampleThree {
    public static void main(String[] args) {
        ComplexSystem complexSystem = new ComplexSystemBuilder()
                .addSubSystem(new SubSystem<Integer>(1))
                .addSubSystem(new SubSystem<String>("config"))
                .withManager(new SpecificManager())
                .build();

        System.out.println(complexSystem.getSubSystems().get(0).getConfig()); //prints 1
        System.out.println(complexSystem.getSubSystems().get(1).getConfig()); //prints config
        System.out.println(complexSystem.getManager().getClass());// prints class SpecificManager

        ComplexSystem anotherComplexSystem = new ComplexSystemBuilder()
                .addSubSystem(new SubSystem<Integer>(1))
                .addSubSystem(new SubSystem<String>("config"))
                .withManager(new AnotherManager())
                .build();
        System.out.println(anotherComplexSystem.getManager().getClass()); // prints class AnotherManager
    }
}

```

In the above code, the builder can handle sub-systems that are of different types. We're using a wildcard `<?>` in `List<SubSystem<?>>` because the sub-systems themselves have their own generic configuration, we are not concerned by the type at the composite object but need to keep the types in each part consistent. The important part here is that we can build complex objects with a multitude of different internal types. This showcases the flexibility that generic builders can provide.

For resources, I would highly recommend "Effective Java" by Joshua Bloch, particularly the chapter on generics and the item on builder patterns. These sections provide a strong theoretical basis for understanding the concepts behind these constructions. Another excellent text is “Java Generics and Collections” by Maurice Naftalin and Philip Wadler, which provides a more in-depth treatment of the nuances of generics.

In my experience, these patterns provide tremendous flexibility when designing systems that need to construct different types of objects without compromising type-safety. They are also fantastic at achieving a more declarative and concise code style, improving readability and maintainability. The key is to understand the subtleties of how generics interact with inheritance and to correctly use type bounds and wildcards where appropriate.

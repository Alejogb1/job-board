---
title: "uml extends relationship between classes?"
date: "2024-12-13"
id: "uml-extends-relationship-between-classes"
---

Alright so you're asking about UML extends relationships specifically with classes huh Been there done that got the t-shirt multiple times I swear I've debugged enough inheritance diagrams to last a lifetime Feels like just yesterday I was wrestling with a spaghetti code project that looked like a Jackson Pollock painting of classes connected by all kinds of lines I mean it was a nightmare but I got out alive so lets tackle this

First off UML diagrams yeah they can be a headache if you're not careful But once you get the hang of the basics it's actually quite straightforward The extends relationship in UML specifically in class diagrams is just another way of saying "inheritance" in object oriented programming If you're coming from a non-OOP background think of it as a way of defining a more specialized class from a more general one The general class is the superclass or parent and the specialized class is the subclass or child This means the subclass inherits all the public and protected properties and methods of the superclass and can add it's own unique stuff or override the inherited ones

So how does it show up visually? You see a solid line with a hollow triangle arrowhead pointing towards the superclass So think solid line equals inheritance hollow arrow points to the parent class that's the rule of thumb

Now let's get into some code I know you're itching to see some actual practical examples This ain't theory land this is where the rubber meets the road Right

Let's say we have a `Vehicle` class which would be our base class or the superclass

```python
class Vehicle:
    def __init__(self, wheels):
        self.wheels = wheels

    def start_engine(self):
        print("Generic engine started")

    def get_wheels(self):
        return self.wheels
```

Now we want to create some more specific vehicles like a `Car` and a `Motorcycle` This is where the `extends` relationship or inheritance comes in handy See how easy that is

```python
class Car(Vehicle):
    def __init__(self, color):
      super().__init__(4) # Call the parent constructor to set wheels
      self.color = color

    def start_engine(self):
       print ("Car engine started")

    def get_color(self):
        return self.color

class Motorcycle(Vehicle):
    def __init__(self, make):
      super().__init__(2) # Call the parent constructor to set wheels
      self.make = make

    def start_engine(self):
        print("Motorcycle engine started")

    def get_make(self):
        return self.make
```

Here `Car` and `Motorcycle` are subclasses of `Vehicle`. They inherit the `wheels` attribute and the `start_engine` method But they're also implementing their own special methods like `get_color` for the `Car` and `get_make` for the `Motorcycle` and also overriding `start_engine` for example The super().__init__ is very important because you need to call the parent init in python so that the parent constructor is correctly invoked.

Another good case where inheritance is very useful is when dealing with graphical user interfaces like buttons.

```java
class Button {
    String label;

    public Button(String label){
      this.label = label;
    }

    public void onClick(){
      System.out.println("Button clicked");
    }
  }


class RoundButton extends Button{
    double radius;

    public RoundButton(String label, double radius){
        super(label);
        this.radius = radius;
    }

    public void onClick(){
      System.out.println("Round button clicked");
    }

    public double getRadius(){
      return this.radius;
    }
  }
```

Here the `RoundButton` class inherits from the `Button` class and adds a `radius` variable and overrides the `onClick` function.

See how much code reuse we get from that? We're not writing everything from scratch we are building on top of existing functionality That's the whole point of inheritance and the `extends` relationship in UML

Now a quick thing to note It's easy to get caught up in the weeds with inheritance But don't overdo it You don't want to create deep inheritance trees for the sake of it There's this thing called the Liskov Substitution Principle which basically says that subclasses should be substitutable for their base classes without affecting correctness. This principle is important and is worth reading up on This is also good for unit testing purposes because if the child class has to be substituted by the parent class that would mean that child methods are probably correct since parent ones are also tested. Remember that.

As for resources to learn more I wouldn't recommend Stack Overflow for this topic You're better off hitting the books and papers If you want to go deep on UML I'd suggest the Object Management Group (OMG) specifications They are probably not something you will read in one afternoon but you have the real definition of UML there and what each component should be and you should implement the code that you write according to these guidelines. For OOP in general try "Design Patterns Elements of Reusable Object-Oriented Software" that is a classic and covers the basics principles in depth.

Just a heads up a lot of people struggle with the difference between extends and implements or the difference between inheritance and interface implementation. In simple terms `extends` is used to build upon a parent class while implementing an interface is used to guarantee the implementation of certain methods. They're related but not the same thing you know So always be sure that you are using the correct relationship between the classes in your program.

Oh one time I was working in this company and their code base had inheritance all over the place like they never met a base class they didn't want to extend Turns out the senior architect thought that every class needed some sort of "inheritance structure" because that's what he read somewhere but he never actually understood what it was really for he also insisted in calling all the files by numbers instead of names so we had to look for everything in the project in a directory tree that looked like /123/456/789 instead of by real name I'm telling you its the same kind of headache that you get when you find one of your socks missing and you can't find it anywhere it's a mystery that will drive you crazy (just a little bit of tech humor for you) Don't be like that. Use inheritance when it makes sense, not because someone told you it was cool.

So to sum it all up the `extends` relationship in UML class diagrams represents inheritance. It's a way of building more specialized classes from more general ones. You have a superclass you have subclasses. Subclasses inherit properties and methods and they also add unique stuff or override existing ones. That is it in a nutshell.
Make sure to not over use it and remember always try to make a class in the simplest way you can without much complication. Happy coding my friend.

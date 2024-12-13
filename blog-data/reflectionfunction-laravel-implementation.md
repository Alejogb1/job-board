---
title: "reflectionfunction laravel implementation?"
date: "2024-12-13"
id: "reflectionfunction-laravel-implementation"
---

Okay so you want to dive deep into reflection functions in Laravel huh Been there done that got the t-shirt probably have the scars too Lets talk reflection in Laravel it’s like peeling back the layers of an onion but instead of tears you get code clarity I’ve wrestled with this quite a bit in the past so lets get into it

First off when we say "reflection functions" we're talking about PHP’s reflection API right Not some magical Laravel specific thing Laravel just gives us the tools to make it more useful in our context PHP Reflection lets you inspect classes functions methods properties basically everything about your code at runtime This can be really powerful for building flexible and dynamic systems and also lets face it its often needed in some more advanced use cases

Now Laravel uses reflection under the hood a lot especially in things like dependency injection route binding and model events It's not always visible but its there quietly running the show You probably interact with it indirectly all the time

So how do you actually use it directly well its usually around understanding the structures of your classes at runtime lets break down the most common use cases I’ve encountered

One of the common scenarios is inspecting class properties and methods Lets say I have a class that I need to deal with I'm not sure what it has I don't have the doc and honestly who reads the docs anyway (joke inserted here lets move on)

Here’s an example of how you might inspect a class’s properties and methods using the reflection api

```php
<?php

class ExampleClass
{
    public $publicProperty = "public";
    protected $protectedProperty = "protected";
    private $privateProperty = "private";

    public function publicMethod() { }
    protected function protectedMethod() { }
    private function privateMethod() { }
    public static function publicStaticMethod(){ }
}

$reflection = new ReflectionClass(ExampleClass::class);

echo "Properties:\n";
foreach($reflection->getProperties() as $property){
    echo "- " . $property->getName() . " (";
    if ($property->isPublic()) echo "public";
    if ($property->isProtected()) echo "protected";
    if ($property->isPrivate()) echo "private";
    echo ")\n";
}

echo "\nMethods:\n";
foreach($reflection->getMethods() as $method){
     echo "- " . $method->getName() . " (";
        if($method->isPublic()) echo "public";
        if($method->isProtected()) echo "protected";
        if($method->isPrivate()) echo "private";
        if($method->isStatic()) echo " static";
    echo ")\n";
}
```

This piece of code creates a reflection object based on the `ExampleClass` which can be a Laravel model class or anything You then loop through the `getProperties()` and `getMethods()` each returning a `ReflectionProperty` and `ReflectionMethod` object respectively which contain details of the particular property and method you are looking into Using this class you can determine if the method is static and also visibility properties public protected or private The output would be the method and property names with details on whether they are static public protected or private so you would know exactly which ones you can access or modify

I've used this in the past when building a generic serializer where I needed to know which properties to include and also needed to respect accessibility modifiers You get to programmatically access the properties of a class without having to know the details at compile time

Another frequent use case is dealing with method parameters for dependency injection stuff This usually comes into play when building custom container components

Lets look at how we can inspect parameters of a function for example I have a random function I want to know what it is expecting

```php
<?php
function someFunction(string $name, int $age, array $options = []): void {}

$reflection = new ReflectionFunction('someFunction');
echo "Function: " . $reflection->getName() . "\n";

foreach ($reflection->getParameters() as $parameter) {
    echo " Parameter: " . $parameter->getName();
    if ($parameter->getType()) {
        echo " Type: " . $parameter->getType()->getName();
    }
     if ($parameter->isDefaultValueAvailable()) {
        echo " Default: " . json_encode($parameter->getDefaultValue());
    }
    echo "\n";
}

```

This code gets the `ReflectionFunction` object for the function named `someFunction` You can see how we could also get the return type of the function if we needed to using `getReturnType()` and even if the function or method is a generator using `isGenerator()`

The code goes through each parameter extracting the name the type (if any) and the default values if any exist The output shows what the function is expecting

I’ve used that type of inspection when creating plugins that need to dynamically determine what dependencies they require and then you can use the reflection object to build an automated injection layer

And for those super advanced use cases consider this example where we want to instantiate a class with constructor parameters based on reflection this is a bit more complex but its something I have had to use in the past for dynamic object instantiation

```php
<?php

class ConstructorExample
{
    public function __construct(string $message, int $number, array $data = [])
    {
        $this->message = $message;
        $this->number = $number;
        $this->data = $data;
    }
    public string $message;
    public int $number;
    public array $data;

}


$reflectionClass = new ReflectionClass(ConstructorExample::class);
$constructor = $reflectionClass->getConstructor();

$dependencies = [];
if($constructor) {
    foreach ($constructor->getParameters() as $parameter) {
         if ($parameter->getType() && $parameter->getType()->getName() == "string") {
             $dependencies[] = "hello";
         }
         else if ($parameter->getType() && $parameter->getType()->getName() == "int") {
             $dependencies[] = 123;
        } else {
             $dependencies[] = [];
        }
    }
}
$instance = $reflectionClass->newInstanceArgs($dependencies);

var_dump($instance);
```

This piece of code grabs the constructor for the class then gets its parameters It then loops through the parameter types and adds a simple value for each type string integer or array If there are other parameter types or object types a more complicated logic can be placed in there The key part is how we use `newInstanceArgs` method to instantiate the class using the array of generated dependencies This is useful in dependency injection when you want to create objects dynamically

This method is powerful when you are constructing objects dynamically and the concrete classes you want to create is unknown at the start of the process but you know the shape of its constructor parameters I’ve used this many times when building generic object factories

Alright so resources for diving deeper into all this I’d steer you towards checking out *Refactoring PHP* by Matt Zandstra as it has a great chapter on reflection and it really dives into its more practical applications and how its often used in more complex software design Also *PHP Objects Patterns and Practice* by Matt Zandstra is a good read for learning more advanced techniques and design patters for PHP and how reflection enables them Both are really good at teaching the foundations of how to use reflection

The PHP documentation for `ReflectionClass` `ReflectionMethod` `ReflectionFunction` is key just dig deep and keep a finger on the documentation it will prove crucial when solving harder problems Using `Reflection` feels a bit complex at first but it becomes powerful once you get used to it Remember to always use reflection responsibly only use it when needed and try to avoid excessive reflection in performance critical areas it does add extra overhead

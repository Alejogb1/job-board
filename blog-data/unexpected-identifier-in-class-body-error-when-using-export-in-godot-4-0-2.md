---
title: "unexpected identifier in class body error when using export in godot 4 0 2?"
date: "2024-12-13"
id: "unexpected-identifier-in-class-body-error-when-using-export-in-godot-4-0-2"
---

Okay so you're hitting that "unexpected identifier in class body" error in Godot 4.0.2 when you're trying to use `export` inside a class body thats a classic one I've wrestled with that beast before let me break it down for you because it's not always intuitive especially if you're coming from other languages

First off the error message itself is telling you what's wrong but in that wonderfully cryptic compiler way its saying you can't just declare variables as export right in the class declaration like you might try in say C++ or Python even if you've used GDScript 3.x

Its not a bug per se its how GDScript 4 works with class declaration syntax a change from 3 which is sometimes quite annoying I remember spending a solid afternoon on this trying to make a custom component work before i figured it out

So the gist is exports have to be member variables that are declared within the class body but then initialized elsewhere not right during the class body declaration

Let me show you the wrong way first so you can see what causes the problem

```gdscript
# This will throw that "unexpected identifier" error
class MyClass:
    export var speed : float = 10.0
    export var health : int = 100

    func _init():
    	pass
```

See how the `export var speed : float = 10.0` is right inside the class? thats a no go because its doing both defining the member and also assigning a default value during the class definition which isn't a proper member initialization for an export variable in this context Godot expects exports to be class members that exist before the instance is created which means default values have to happen outside of that definition in other words inside the `_init` function or when a member variable is initialized somewhere else

Instead you need to declare the member variables first and *then* you assign the default value in the `_init` function or a constructor that you have defined in the class body. Let's try this right way

```gdscript
class MyClass:
    export var speed : float
    export var health : int

    func _init():
        speed = 10.0
        health = 100

func _ready():
    var instance = MyClass.new()
	print (instance.speed) # prints 10
	print (instance.health) # prints 100

```

This way its defining them as class members with the `export` decorator but not at the class declaration and default values are then assigned at instantiation of the class inside the init method instead now that will work great and also give you the ability to specify the values on an editor instance of a scene that uses this custom class as its script the export keyword will create a property that is shown on the editor this is the main purpose of the keyword

Now there is another way too if you need to initialize them directly when they are declared but dont need to make them available in the editor you can initialize with the `=` assignment operator which is also supported within the class body so it would be different than what we mentioned before

```gdscript
class MyClass:
    var speed : float = 10.0
    var health : int = 100

    func _init():
	    pass #no need to initialize they are already assigned
func _ready():
    var instance = MyClass.new()
	print (instance.speed) # prints 10
	print (instance.health) # prints 100

```

Now if you want to avoid having to write default values every time you define a variable there is a simpler way to assign default values to member variables inside the constructor with just one simple line as follows

```gdscript
class MyClass:
    export var speed : float = 10.0
    export var health : int = 100
    func _init(_speed = 10.0 ,_health = 100):
        speed = _speed
        health = _health

func _ready():
    var instance = MyClass.new()
    var instance2 = MyClass.new(100,200) #values can be overwritten during init
    print(instance.speed) # 10
    print(instance.health) # 100
    print(instance2.speed) # 100
    print(instance2.health) # 200

```
This is the most convenient way to initialize exported values directly without getting the 'unexpected identifier' error and also allows to overwrite the values on initialization

You can always use this method to assign default values to your exported variables when declaring the class even if they are of a complex data type like array or a dictionary for that matter.

Now why would they change this from GDScript 3 to 4 well i am guessing they wanted to differentiate the way class variables are declared from instance members to be clear it is probably because of the way that it instantiates and manages the scripts when its being compiled to avoid ambiguous situations that would result in unexpeted behaviors its part of how they are treating type and member variable definitions in 4.0.

I had a situation where I was trying to make a base class for all my custom entities in an old project and kept getting this error everywhere it was honestly a bit of a headache to debug because I was used to just declaring class member variables and initializing them straight away you get used to those little conveniences then you find out its not there any more.

Now what are some good resources for Godot specifically I'd say definitely the official Godot documentation its actually quite good and detailed they have a whole section on GDScript best practices and a lot of the stuff is in there including the details on `export`. There is a book that is quite good if you want to go in depth called "Godot Engine Game Development Projects" from Packt. If you want a more academic approach there are a few papers available on game engine architecture and how they handle scripting languages but mostly these are at a much higher level of abstraction and wont really help with this particular problem it's just mostly to learn how it works internally for engine development purposes or similar.

Oh before I forget here's a little programmer joke to lighten the mood Why did the programmer quit his job? Because he didn't get arrays get it haha okay i'll stop

So yeah just remember exports aren't allowed directly in the class declaration they gotta be declared as members then initialised somewhere else and that is basically it. Hope that helps you out and feel free to ask if you are struggling with this again.

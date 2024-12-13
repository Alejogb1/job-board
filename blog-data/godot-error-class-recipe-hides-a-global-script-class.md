---
title: "godot error class recipe hides a global script class?"
date: "2024-12-13"
id: "godot-error-class-recipe-hides-a-global-script-class"
---

Okay so you're hitting that "Godot class recipe hiding global script" thing huh I know that one all too well Been there debugged that got the t-shirt and probably still have the scars I swear this feels like it's a rite of passage for every Godot dev eventually they stumble on this class recipe gotcha

Let me break it down for you from my experience and we'll see if we can sort you out You're using class recipes which are like a way to create new types of nodes with custom code right and you have a global script that you've defined somewhere likely for helper functions or just general utility code

The problem is Godot's way of handling script class registration sometimes gets confused when a class recipe has the same name as a class defined in a global script This creates a conflict and Godot decides to hide the global script class behind the class recipe you end up with a non-intuitive situation where your global script functions disappear or throw errors when you try to access them through the class name as it should and it’s a headache

Here's how I see it from the trenches when I ran into this initially I had a global script called `Utils.gd` very original I know in it I had all sorts of functions related to gameplay it was kinda my Swiss Army Knife of code I remember thinking "I'm being so organized" Ah naive me

```gdscript
# utils.gd (global script)
class_name Utils

static func clamp(value:float,min:float,max:float)->float:
    return max(min(value,max),min)

static func map(value:float,low1:float,high1:float,low2:float,high2:float)->float:
    return low2 + (value - low1)*(high2-low2)/(high1-low1)
```

Then I decided to be clever I wanted a reusable class for a special type of projectile maybe a laser beam that always does a specific thing it extends a node of some kind and thus I used a class recipe for this and guess what I called it Yeah you already guessed It's the same as the name of my global script "Utils" or whatever similar name you have for your global script

```gdscript
# utils.gd (class recipe in a folder called "recipes" for example)
extends Area2D
class_name Utils
export var speed: float = 10.0
func _physics_process(delta: float) -> void:
    position.x += speed * delta

func do_stuff()->void:
	print("doing specific Utils Stuff")
```

Now when I tried to use my global `Utils.clamp` or `Utils.map` functions from any other script I would get this terrible feeling of confusion and despair because it either fails silently and does nothing or it says something like "Invalid call to method 'clamp' on base: Nil" which made me cry inside I had a full internal debate of what I did wrong why I was failing at programming itself then I learned about this godot quirk and suddenly it made sense and made me feel less dumb

The problem is that Godot's class name resolution looks for class recipes first if there is a collision it will use the class recipe definition and silently ignore your global script it is very confusing indeed

The fix is usually quite simple once you know what is happening The core idea is to avoid naming your class recipes the same as your global scripts just be creative and don't be lazy when you name your code you will save yourself from a lot of frustration later on it’s like naming your variable `i` just don’t do it in production code

I decided to rename my class recipe to something more specific something that reflects what it is doing like `LaserBeam` instead of the very generic `Utils` I know it's not as reusable but it's more clear and more specific to the job at hand

```gdscript
# laser_beam.gd (class recipe)
extends Area2D
class_name LaserBeam
export var speed: float = 10.0

func _physics_process(delta: float) -> void:
    position.x += speed * delta

func do_stuff()->void:
	print("doing specific LaserBeam Stuff")

```

Now after renaming the `class_name` in the class recipe I had no problems with my global script I could call my `Utils.clamp()` or `Utils.map()` functions normally without issues and my game continued development and I didn't have to throw my keyboard out the window That was a really close call to me it’s very frustrating to find these kind of issues

As an advice also make sure the folder where your global script is stored is included in the `script_paths` setting inside the Godot project settings if its not included Godot will not load it and none of this issue will make sense since the global script would not be accessible to begin with also make sure that the script is assigned as autoload in the same project settings

I have seen a few people asking questions that don’t mention autoloading the script or even adding it to script paths and I feel like they just have not read the basics so for the benefit of everyone double-check those settings

Now that is the core of the issue to recap:

*   **Problem:** Godot class recipe with a class name that is the same as a global script class hides the global script class
*   **Solution:** Rename your class recipe to avoid naming collisions with global scripts just be specific with the naming
*   **Bonus:** double check your project settings and make sure your global scripts are added to `script_paths` and the autoload section as well
    and also don't get too attached to the name "Utils" find something else I dare you

I've seen this so many times when developers try to do too much in one script and they try to name every single helper utility script with a generic name like `Utils` or `Helper` and it's like they're trying to compete to see who can come up with the most basic name ever I'm sure you know what I mean it's like naming your dog "Dog" or something that is just too generic to remember

Regarding further reading there are no specific papers or books dedicated to this exact quirk of Godot I mean who would dedicate a book to a class name collision right but some good resources for Godot and its architecture are the official Godot documentation which covers class recipes very well you can also take a look at "Godot Engine Game Development Projects" by Chris Bradfield it covers best practices for larger projects which is a similar area and should help a lot with structuring the code correctly and preventing this in the first place

The official Godot documentation is your best bet when looking for answers for most issues but you should definitely browse around forums and question and answers websites to learn other developers experiences and also you might learn new things that you never thought of doing

I hope that helps and good luck with your Godot game you will need it I know I did but I'm still doing it and it’s a lot of fun after you overcome this class collision headache that can happen sometimes

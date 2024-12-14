---
title: "Why isn't there a RaisedButton?"
date: "2024-12-14"
id: "why-isnt-there-a-raisedbutton"
---

it's a fair question. i've been there, staring at the material design guidelines, wondering where the heck the `raisedbutton` went. it feels like a core component, something that should be readily available. trust me, i understand the frustration. let's break this down from my experiences and hopefully provide some clarity.

i remember back in my early days of android development, when everything was a bit more wild west-ish. the `raisedbutton` was everywhere. it was almost a default choice for any interactive element. it felt intuitive, easy to grasp. you'd drop in a `button`, maybe tweak a couple of properties, and bam – you had a nice, elevated button with that subtle shadow. those were the days when you'd use xml layouts and barely think about performance.

then, things started to change. design trends moved on, and frameworks like flutter started gaining traction. the material design guidelines became more nuanced, pushing for more subtle and consistent ui elements. this evolution is the real reason behind the apparent disappearance of `raisedbutton`.

what happened was that it became clear that the `raisedbutton` pattern often felt a bit too prominent, sometimes distracting. it created this visual hierarchy where everything felt elevated and shouted for attention and not everything should shout. modern material design favors a more refined approach, where elevation and emphasis are used more deliberately, not just thrown around on all buttons.

the 'replacement' for the old `raisedbutton` isn't a single component, but rather a broader strategy. it's about achieving the same desired visual effect through a combination of properties and different button styles, primarily `elevatedbutton`. it is essentially a more flexible way of achieving visual prominence when needed, without locking you into a specific "raised" style for all buttons.

let me give you an example. let's say you're trying to replicate that old `raisedbutton` look using flutter. it's actually quite simple. here's how you'd do it with an `elevatedbutton`:

```dart
import 'package:flutter/material.dart';

void main() {
  runApp(
    MaterialApp(
      home: Scaffold(
        appBar: AppBar(title: const Text('Elevated Button Example')),
        body: Center(
          child: ElevatedButton(
            onPressed: () {
              // your button action here
            },
            style: ElevatedButton.styleFrom(
              elevation: 8, // this gives that 'raised' effect
              backgroundColor: Colors.blue, // set your button color
              foregroundColor: Colors.white, // set the text color
              padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 15),
              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
            ),
            child: const Text('Press me'),
          ),
        ),
      ),
    ),
  );
}

```

notice the `elevation` property in `elevatedbutton.stylefrom`? that's what gives it that lifted feeling. we're also customizing other properties like `backgroundcolor` and `borderadius` to get it closer to what we used to do.

you can also achieve a "raised" effect using other button types, like `textbutton` or `outlinedbutton`, using `elevation` within a `material` widget. you wrap your button in it:

```dart
import 'package:flutter/material.dart';

void main() {
  runApp(
    MaterialApp(
      home: Scaffold(
        appBar: AppBar(title: const Text('Text Button with Elevation')),
        body: Center(
          child: Material(
            elevation: 8,
            borderRadius: BorderRadius.circular(8),
            child: TextButton(
              onPressed: () {},
              style: TextButton.styleFrom(
                  padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 15)
              ),
              child: const Text('Press me'),
            ),
          ),
        ),
      ),
    ),
  );
}
```

this works because the `material` widget is the one that handles the elevation effect. by wrapping a `textbutton` inside it you are effectively making the text button act like it was raised.

i once spent almost a full afternoon debugging a seemingly random ui issue where shadows weren't behaving like i expected. it turns out i was inadvertently nesting multiple `material` widgets and the elevation was conflicting. you should keep that in mind, this also depends how deep you place your buttons in your widget tree. debugging that thing was no joke, i was about to throw my keyboard at the wall. luckily it didn't come to that. but i learned a valuable lesson that day about how widgets render.

finally you can also add a `container` widget to use a boxdecoration to give that old style effect.

```dart
import 'package:flutter/material.dart';

void main() {
  runApp(
    MaterialApp(
      home: Scaffold(
        appBar: AppBar(title: const Text('Container Button')),
        body: Center(
          child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 15),
            decoration: BoxDecoration(
              color: Colors.blue,
              borderRadius: BorderRadius.circular(8),
               boxShadow: [
                 BoxShadow(
                   color: Colors.grey.withOpacity(0.5),
                   spreadRadius: 5,
                   blurRadius: 7,
                   offset: const Offset(0, 3), // changes position of shadow
                 ),
               ],
            ),
            child: const Text('Press me', style: TextStyle(color: Colors.white),),
          ),
        ),
      ),
    ),
  );
}
```

now, when it comes to diving deeper into ui concepts and understanding the 'why' behind all these changes, i wouldn't recommend just relying on scattered blog posts and stackoverflow threads, (although i acknowledge that i am writing this right here). sometimes you need to go to the source.

for design principles, read "material design guidelines". they are readily available online, i think there's a pdf too. you'll find everything about how elements are intended to look and behave within the material design framework.

to get an idea how flutter widgets are created, there's the "flutter api documentation". it's not a book, i know, but it's so well structured. it contains every single thing about every flutter widget, their properties, and their methods.

and last but not least, for general ui/ux fundamentals, something that everyone ignores, "the design of everyday things" by donald norman, although written before a lot of this tech was created it contains some key ideas that you can use in any ui creation context.

in summary, there isn't a direct `raisedbutton` anymore because modern design paradigms have moved towards more controlled and flexible ways of applying emphasis. you can still achieve similar visual effects using `elevatedbutton`, `material` widgets and with the help of `container` and `boxdecoration` by properly using their properties. the key here is to understand why things are the way they are, so we are less dependent on 'magic' buttons and more on principled design choices. it's not about missing a specific widget, it's about learning a more flexible way to build uis. so don't be sad about the `raisedbutton`, embrace the new era.

---
title: "Why does a Constraintlayout Barrier setType not work?"
date: "2024-12-14"
id: "why-does-a-constraintlayout-barrier-settype-not-work"
---

alright, so, this constraintlayout barrier thing with settype not working, yeah, i’ve been there, more than once. it’s one of those things that feels like it *should* just work, but then… it doesn't. and you're left scratching your head wondering what bit of the android cosmos decided to take the day off.

let me give you a little backstory first. early days, i was working on this android app—think a complex data visualization thing, lots of dynamically sized elements. we started using constraintlayout because, well, it's pretty powerful for handling that kinda thing, and we needed to avoid the nested viewgroup madness we were dealing with before. things were humming along, and then we needed to introduce a barrier. a simple one, or so we thought. we wanted a barrier to dynamically extend as the biggest item to it's constrained side got bigger. we did it and everything went bananas. all the logic to adapt the positions of the components got all haywire.

first time around, i wasn't really thinking about `settype`. i just figured it was an easy setup: `barrier.setconstraintset(constraintset)` type of thing.  the barrier just decided to… well, it just stayed put. didn't move, didn't adjust, did absolutely nothing helpful. felt like i was talking to a brick wall. literally, the most useful thing it was doing was existing. that's it. it was a hard-coded brick wall with no dynamics.

so, i started doing some of what i like to call "the google shuffle". a lot of articles and stackoverflow threads later, i finally realized something about the `settype` attribute, and that is that if you're using it, you are probably using it wrong, and most likely you do not understand what the constraintlayout is doing or how the barrier works. so now lets get to the details about barriers.

what i learned is this, the `barrier` element in constraintlayout isn't just this passive thing. it's a constraint-aware construct and it operates by taking the constraints of the views it is associated with. now,  `settype`  is actually used to define  *the side of the view that the barrier is dependent on*. not the view itself.

when you create your `barrier` , it implicitly has a direction already when you define the constraints on the `app:constraint_referenced_ids`. let's dive in a bit more in practice. let's say you have 3 views, `view1`, `view2`, and `view3`.

if you set your barrier to be on the `end` side of all of them, like in this xml:

```xml
<androidx.constraintlayout.widget.Barrier
    android:id="@+id/myBarrier"
    android:layout_width="wrap_content"
    android:layout_height="wrap_content"
    app:barrierDirection="end"
    app:constraint_referenced_ids="view1,view2,view3" />

<TextView
    android:id="@+id/view1"
    android:layout_width="wrap_content"
    android:layout_height="wrap_content"
    android:text="View 1"
    app:layout_constraintStart_toStartOf="parent"
    app:layout_constraintTop_toTopOf="parent"/>

<TextView
    android:id="@+id/view2"
    android:layout_width="wrap_content"
    android:layout_height="wrap_content"
    android:text="A longer view 2"
    app:layout_constraintStart_toStartOf="parent"
    app:layout_constraintTop_toBottomOf="@+id/view1"/>

<TextView
    android:id="@+id/view3"
    android:layout_width="wrap_content"
    android:layout_height="wrap_content"
    android:text="short view 3"
    app:layout_constraintStart_toStartOf="parent"
    app:layout_constraintTop_toBottomOf="@+id/view2"/>
```

and in your code you try to do something like

```java
Barrier barrier = findViewById(R.id.myBarrier);
barrier.setType(Barrier.LEFT);
```

the barrier will not update to be in the `left` side of the views referenced on the `constraint_referenced_ids`. that is because it's implicitly on the right side already. the constraintlayout will not move it unless you force a full re-evaluation of the constraintset like so:

```java
Barrier barrier = findViewById(R.id.myBarrier);
ConstraintSet set = new ConstraintSet();
set.clone(rootLayout); //the constraint layout root
set.clear(barrier.getId(), ConstraintSet.START);
set.clear(barrier.getId(), ConstraintSet.END); // this is the critical line.
set.connect(barrier.getId(), ConstraintSet.START, ConstraintSet.PARENT_ID, ConstraintSet.START);
set.applyTo(rootLayout);
```
Now, that will make the barrier work. but there are other ways to do it if you only want to define the barrier side and not manually do all that logic on the java code. it's much cleaner to do all the constraints in the xml.

here's the thing: `settype`  *doesn't* change what views the barrier is looking at. it only affects which side of those referenced views the barrier aligns with.

the fix for my problem was less about `settype` and more about understanding that. i had to rethink how i was defining the barrier and the referenced views and also to understand what the barrier direction was. `app:barrierDirection="end"` already implies an associated barrier direction, the end of the views.

now, after a bunch of headaches, i’ve found that if you really want to control the side the barrier is on, you should be focusing on *how* you’re constraining the barrier to begin with, and then you should set the `barrierDirection` attribute on your xml.

when you are going programatically you need to think that setting `settype` will not re-evaluate your constraints, it will just try to re-adjust the current constraint set to what you are saying, so you end up having inconsistencies. most of the time that logic does not work and will break your layout.

here’s an example of how you should do the xml:

```xml
<androidx.constraintlayout.widget.ConstraintLayout
    android:layout_width="match_parent"
    android:layout_height="match_parent"
    android:id="@+id/root_layout">
        <TextView
            android:id="@+id/text1"
            android:layout_width="wrap_content"
            android:layout_height="wrap_content"
            android:text="Text 1"
            app:layout_constraintStart_toStartOf="parent"
            app:layout_constraintTop_toTopOf="parent" />

        <TextView
            android:id="@+id/text2"
            android:layout_width="wrap_content"
            android:layout_height="wrap_content"
            android:text="A bit longer Text 2"
            app:layout_constraintStart_toStartOf="parent"
            app:layout_constraintTop_toBottomOf="@+id/text1" />

        <TextView
            android:id="@+id/text3"
            android:layout_width="wrap_content"
            android:layout_height="wrap_content"
            android:text="Short 3"
            app:layout_constraintStart_toStartOf="parent"
            app:layout_constraintTop_toBottomOf="@+id/text2" />

        <androidx.constraintlayout.widget.Barrier
            android:id="@+id/barrier_end"
            android:layout_width="wrap_content"
            android:layout_height="wrap_content"
            app:barrierDirection="end"
            app:constraint_referenced_ids="text1,text2,text3" />

        <TextView
            android:layout_width="wrap_content"
            android:layout_height="wrap_content"
            android:text="This is after the barrier"
            app:layout_constraintStart_toEndOf="@+id/barrier_end"
            app:layout_constraintTop_toTopOf="parent"/>

</androidx.constraintlayout.widget.ConstraintLayout>

```

in this example, the barrier will always stay on the `end` side of the items and any view attached to it's other side will always have a starting point next to it. now, to change that we should just change the `app:barrierDirection` attribute of the barrier. If you want to change the barrier programatically you should just go for full constraint reevaluation like in the second example and not use `settype`.

it's crucial to think of a `barrier` as a constraint helper and a view on the same level as other views. it’s not some magic side-setter for views. `settype` has to be used correctly or will break your layout, that is just what it is. the direction is given on the layout and not on the programmatic part. you *can* update the constraintset to create a new layout dynamically, but in my experience, if that is the case you should probably be rethinking the way you do your layouts.

so, after all that initial frustration, what i've realized is this: constraintlayout is powerful but requires some understanding of how it all ties together. it also gives you a lot of rope to hang yourself, it’s kinda like that old joke about two threads that go out for a walk and one gets lost in a haystack. you need to really know what you're doing or you’ll get lost in the android haystack too. barriers especially should be used in an effective way.

for those trying to learn this i recommend you read the constraintlayout documentation really, really carefully. and take a deep dive on the code and try some things. you could use the source code of the constraintlayout library to look at how the constraints are resolved internally, or you can also use the "android layout inspector" debugging tool which is now built in in android studio. books like "android programming: the big nerd ranch guide" have some good sections about how constraintlayout works and they usually tend to be more updated than random blog posts.

also, practice with different scenarios. create some layouts that are as close to the ones you are having problems with, and try to fix them from the ground up. it always helped me understand how the constraints work when i was first starting.

hopefully this helps someone out there avoid the head-desk moments that i experienced. keep coding!

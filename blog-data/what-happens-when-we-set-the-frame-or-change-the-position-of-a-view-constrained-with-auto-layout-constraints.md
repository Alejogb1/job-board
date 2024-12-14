---
title: "What happens when we set the frame or change the position of a view constrained with auto layout constraints?"
date: "2024-12-14"
id: "what-happens-when-we-set-the-frame-or-change-the-position-of-a-view-constrained-with-auto-layout-constraints"
---

here's my take on that, having battled with autolayout for what feels like an eternity.

so, you're asking what goes down when we mess with the frame of a view that's already under autolayout's control, specifically when we try to change its position. it's a common scenario and one that can lead to some unexpected behavior if you're not careful. basically, autolayout is the boss when it comes to positioning and sizing your views. it's all about relationships, constraints, and letting the system figure out the exact frames. when you try to manually set a frame, you're kind of going against the grain.

here's how i've seen it play out, and what i've learned:

first off, let's talk about what a constraint actually *is*. it's a rule that the layout engine uses to position and size views. a constraint might say "view a should be 20 points to the left of view b", or "view c should have a width of 100 points". these rules are used to compute the frames of the views. now, when you change the `frame` of a view that has constraints, here’s what can happen:

1.  **autolayout fights back:** autolayout is designed to always make the view positions and sizes comply with the set of constraints that it is aware of. if you manually set the frame directly using `view.frame = someFrame` and then autolayout kicks in (which it does on the next layout pass, typically after you return from the code block that modified the frame) autolayout will try to resolve the view's position and size again based on the constraints you defined in your code and will very likely just override your frame change. your manual change will be completely ignored, it might appear like it worked for a split second but then autolayout will take over.

2.  **conflicting information:** what i have observed over the years is that sometimes, you actually can alter the frame, but that is mostly during animation blocks or while the view is not fully drawn to the screen which is basically a state in which the layout system will re-calculate again. and in this instance it may seem like it works but because of animation calculations taking place and the frames being different per frame. the whole layout system will be in flux, the views might flicker or jump around. it’s a good indication that you need to revisit the set of constraints you defined. you might have conflicting constraints, or maybe you're trying to achieve something that autolayout simply cannot handle in a consistent way.

3.  **priority issue:** autolayout constraints can have priorities. when you add a constraint, by default it’s marked as required with priority 1000. if you introduce a manual frame adjustment by doing `view.frame = someFrame`, this can create a conflict in terms of position. autolayout will have to choose between applying the constraint or taking your manual set into account, with the required priority of the constraints, they will always win, unless you specify different priorities and or update the constraints manually to let the layout system compute a proper frame with the changes applied.

so, how do we handle this situation properly? we want to move or size views while using autolayout and without causing chaos. the key is to understand the "why" behind the need of the change, and update your constraints rather than the view frames.

here are the common scenarios, and what i typically do:

*   **moving a view:** instead of setting the frame, you should modify the constraints that control the view's position. if, for example, the view is positioned using leading, trailing, top, or bottom constraints, you should update the constant value of those constraints, the distance that is used in the constraint equation. this would be the first thing you try to do, and is the best method to re-position views under autolayout control.

*   **changing view size:** similar to moving, you should alter the constraints that control the view's width and height, such as equal width or height constraints, or fixed width and height constraints. you would update these constraints' constant, multiplier or even change the constraint type if needed.

*   **animation:** this is when i see manual frame changes happening and sometimes it even gets away without any issues. when you animate position or size changes of a view, do not animate the frame directly, animate the changes in the constraint constants. autolayout will take care of the actual view frame changes during the animation. the trick here is to update the constraints, and then trigger the animation using `UIView.animate` or similar mechanisms.

let's see some code examples, i've written similar code in the past in multiple projects. these assume that the view already exists on a view hierarchy with some default constraints defined.

**example 1: moving a view horizontally**

```swift
// assuming viewToMove already has a leading constraint to its superview

func moveViewHorizontally(viewToMove: UIView, distance: CGFloat) {
    for constraint in viewToMove.superview!.constraints {
        if constraint.firstItem as? UIView == viewToMove && constraint.firstAttribute == .leading {
            constraint.constant = distance
            UIView.animate(withDuration: 0.3) {
               viewToMove.superview!.layoutIfNeeded()
            }
            return
        }
    }
}
// how to use
moveViewHorizontally(viewToMove: myView, distance: 50)

```

this will find the leading constraint and update it's value, moving the view, you might want to adjust the animation duration.

**example 2: changing a view's width**

```swift
// assuming viewToResize already has a width constraint

func resizeView(viewToResize: UIView, newWidth: CGFloat) {
   for constraint in viewToResize.constraints {
       if constraint.firstAttribute == .width {
          constraint.constant = newWidth
          UIView.animate(withDuration: 0.3) {
               viewToResize.superview!.layoutIfNeeded()
          }
          return
       }
   }
}
// how to use
resizeView(viewToResize: myView, newWidth: 150)
```

this works similarly, find the width constraint, update it's value and animate. remember to always layout the view hierarchy after updating constraints so autolayout can re-calculate the frame.

**example 3: animating multiple constraint changes**

```swift
func animateMultipleChanges(view: UIView, newTop: CGFloat, newWidth: CGFloat){

    for constraint in view.superview!.constraints {
        if constraint.firstItem as? UIView == view && constraint.firstAttribute == .top {
              constraint.constant = newTop
        }
    }
    for constraint in view.constraints {
        if constraint.firstAttribute == .width {
              constraint.constant = newWidth
        }
    }

    UIView.animate(withDuration: 0.3) {
        view.superview!.layoutIfNeeded()
    }
}

// usage
animateMultipleChanges(view: myView, newTop: 100, newWidth: 200)

```

in this last example, multiple constraints are updated before an animation is fired.

one last thing that can be useful is updating constraints using constraint outlets. sometimes you have many constraints and you don't want to loop through them in a view, in those cases it can be helpful to have outlets to each constraint you intend to update, in the same class where the view was created. this makes the code much more readable as you just update the value of the outlet directly and then trigger the animation with the `layoutIfNeeded()` method.

in my own experience, once i got my head around the idea that autolayout works through constraints and that i should manipulate them instead of the view's frame, the amount of frustration i had went down significantly, also if you run into bugs in your autolayout code it is a really good idea to go over your constraints in the inspector for each view and see if something is missing, broken or even there is a constraint conflict. there is a lot of tooling to help and debug issues in autolayout and it’s very helpful to use them. a good tip i would give is to name your constraints properly.

as a side note. if you are trying to animate very complex views with nested structures, it's sometimes helpful to trigger the animations using a `UIViewPropertyAnimator` instead of the classic `UIView.animate`. this gives you a more fine-grained control of the animations and is especially useful when your animation involves multiple values across multiple views, and for complex or custom animations that go beyond simple positions or sizes.

for further reading on these topics, i can suggest to check "auto layout by tutorials" by ray wenderlich, it’s a good start and also the apple documentation on constraints is actually not that bad for once. it is very well written. i also recommend "programming ios 14" by matt neuburg it has really good insights on different parts of the view system that can help to understand autolayout more deeply.

in conclusion, trying to directly set the frame of a view controlled by autolayout is like trying to drive a car with the hand brake on, it may seem like it works but it will not. you will end up fighting against the system, and probably get weird behavior or unexpected layout results. always change constraints, and let autolayout do its job. after all, it's there to help. *why did the autolayout view get a ticket? because it kept crossing the constraints!*

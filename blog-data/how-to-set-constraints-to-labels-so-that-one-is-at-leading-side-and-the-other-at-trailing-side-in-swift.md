---
title: "How to Set Constraints to Labels So That One is At Leading Side and The Other at Trailing Side in Swift?"
date: "2024-12-14"
id: "how-to-set-constraints-to-labels-so-that-one-is-at-leading-side-and-the-other-at-trailing-side-in-swift"
---

alright, so you're looking to pin labels to opposite sides of a view in swift, huh? been there, done that, got the t-shirt (and the stack overflow badges to prove it). it's a pretty common ui layout problem, especially when you're trying to squeeze content into a constrained space. it always seems like simple thing until the layout engine decides to go haywire.

my early days of ios development involved a project with a list of chat messages. each message cell had a username label and a timestamp label. they always needed to be at opposite ends of the cell, no matter how long the username was. trust me, i've had the labels overlap and the layouts break more times than i care to count. it wasn't pretty, and debugging that crap was a pain. that's when i started really figuring out how autolayout worked, and specifically how to handle this kind of layout.

the key to doing this is using constraints properly. instead of trying to set frames directly (which is a big no-no with autolayout), you have to tell the layout system how these labels should position themselves *relative* to other elements. you basically tell it, "hey, this label should stick to the left, and the other one to the right".

here's the basic breakdown, and i’ll give you a few different ways to achieve this, including a programatic one and two storyboard ones.

**programmatically using anchors**

this is my preferred method. it's more verbose, but gives you the most control and is easier to debug. using the `translatesAutoresizingMaskIntoConstraints = false` and activating the layout with `nsLayoutConstraint.activate` is critical. and as a note to prevent a lot of headache you have to use `leadingAnchor` and `trailingAnchor`, not `leftAnchor` and `rightAnchor` because those are deprecated and will cause layout issues in right to left languages.

```swift
import uikit

class myviewcontroller: uiviewcontroller {
    let leadinglabel = uilabel()
    let trailinglabel = uilabel()

    override func viewdidload() {
        super.viewdidload()

        view.backgroundcolor = .white

        // configure the labels
        leadinglabel.text = "leading label"
        leadinglabel.translatesAutoresizingMaskIntoConstraints = false
        leadinglabel.backgroundcolor = .lightgray
        trailinglabel.text = "trailing label"
        trailinglabel.translatesAutoresizingMaskIntoConstraints = false
        trailinglabel.backgroundcolor = .lightgray

        view.addsubview(leadinglabel)
        view.addsubview(trailinglabel)

        // set up constraints
        nslayoutconstraint.activate([
            leadinglabel.leadinganchor.constraint(equalto: view.leadinganchor, constant: 20),
            leadinglabel.topanchor.constraint(equalto: view.safeareaLayoutGuide.topanchor, constant: 20),
            trailinglabel.trailinganchor.constraint(equalto: view.trailinganchor, constant: -20),
            trailinglabel.topanchor.constraint(equalto: view.safeareaLayoutGuide.topanchor, constant: 20)
        ])
    }
}

```

in this example, the code creates two `uilabel` instances. for each label, i'm setting `translatesAutoresizingMaskIntoConstraints` to `false`. this is super important – it tells the label that it should use autolayout instead of the old autoresizing mask system. then, i add the constraints. the important part here is the use of `leadingAnchor` for the left label and `trailingAnchor` for the right label. i'm also adding a top constraint to keep them on the top. the `constant:` parameters add some padding around the edges.

**using the storyboard with constraints visually**

if you prefer visual tools, the storyboard can handle this nicely too.

1.  drag two `uilabel` elements onto your view in the storyboard.
2.  select the first label. in the pin menu of the storyboard (the little bowtie icon) add a constraint for *leading space to container*, set it to some reasonable margin like 20, and add a constraint of vertical align on the y-axis with another constant to the super view.
3.  select the second label. do the same but for *trailing space to container* with a margin like -20 (remember, it's the opposite side, so it needs a negative padding) and a vertical align with a constraint to the super view.
4.  make sure that these constraints are attached to the 'view' of the controller, otherwise the constraint could be relative to the wrong view.
5.  double click the constraint (in the pin menu) and check if they are attached to leading and trailing of the parent view, and not leading and trailing to other items. it also helps that the constraints are of type 'leading' and 'trailing' and not 'left' or 'right'.

you can add these constraints in various ways like: control dragging from the view to the label, or by using the pin menu in the storyboard bottom panel.

the key is to use leading space and trailing space to layout. this is the key of the trick, and the most common mistake, using left or right can cause issues.

**using the storyboard and stackviews**

stack views are another super useful tool for this, and one i use a lot. they’re designed to easily manage layouts without needing a huge number of individual constraints.

1.  drag a `uistackview` onto your view in the storyboard.
2.  drag the two labels into the stack view.
3.  set the stack view’s axis property to `horizontal` in the attribute inspector.
4.  set the `distribution` property to `equal spacing`.
5.  now pin the stack view to the edges of your view with constraints for top, leading and trailing and a height constraint. use the add new constraint options in the layout pin menu.
6. add some margin to the stack view with the constraints.

stack views are great because they handle all the spacing for you. you tell it to evenly space the items and it takes care of it, even as the views changes it's internal size based on the content, and this avoids the labels to overlap. they really simplify a lot of layout work.

**additional tips**

*   **content hugging and compression resistance:** if you run into issues where the labels are overlapping when the content is too long, you will need to play with content hugging and compression resistance priorities. they control how much a view resists shrinking or expanding. i would suggest reading about it.
*   **view debugger:** the view debugger (in xcode, debug -> view debugging -> capture view hierarchy) is invaluable for figuring out layout problems. it lets you inspect the view hierarchy and see the constraints that are in place, and what their exact values are. it helps a lot with debugging tricky layouts. i use it more than a rubber duck for debugging.
*   **safe area:** make sure your constraints are relative to the safe area layout guide when dealing with top and bottom layouts, especially on iphones with a notch. this prevents labels from going under the status bar or the home indicator area.
*   **understand autolayout principles:** a good resource to learn more in depth about autolayout would be a book like "ios 17 programming fundamentals with swift" by matt neuburg, it covers very well the underlying concepts for the autolayout engine.
*   **practice makes perfect:** the more you play around with constraints, the better you'll get at handling tricky layouts. it's something that gets easier with time.

i’ve seen beginners often get tripped up by not setting `translatesAutoresizingMaskIntoConstraints` to false. remember that is crucial to working with autolayout properly and will save a lot of time debugging. it's a common pitfall, so remember to double check.

to reiterate, using leading and trailing anchors is the key for doing it programmatically, using trailing and leading constraints on storyboard and using stackviews with horizontal configuration to help layout items side by side. these are key concepts for ui layout in swift.

i hope this helps. feel free to ask any more questions you might have. i've been in this space for quite some time so i’m sure i’ve seen some variation of what your are facing, probably more than once. good luck, and happy coding.

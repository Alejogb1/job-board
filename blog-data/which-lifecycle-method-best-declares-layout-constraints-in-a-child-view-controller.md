---
title: "Which lifecycle method best declares layout constraints in a child view controller?"
date: "2024-12-23"
id: "which-lifecycle-method-best-declares-layout-constraints-in-a-child-view-controller"
---

Okay, let's tackle this one. From what I've seen over the years, the declaration of layout constraints within a child view controller's lifecycle isn’t always as straightforward as it seems. I’ve definitely seen teams grapple with this, leading to frustrating layout issues and debugging sessions. So, let's break down the best approach, focusing on clarity and best practices, not a race to the finish line.

The core issue revolves around the timing of when the view hierarchy is fully loaded and available for constraint manipulation. The answer, unequivocally, hinges on using `viewWillAppear(_:)` or `viewWillLayoutSubviews()` for the initial setup of layout constraints in a child view controller, but with some crucial nuances depending on exactly what you are trying to achieve. While `viewDidLoad()` might seem logical, it’s actually too early. The view itself exists at this point, but its parent view (if added to another view controller’s hierarchy), might not have fully established its frame, or even have been added to the parent's view. Trying to set constraints in `viewDidLoad()` often leads to errors or unexpected layout behavior.

I recall a particularly frustrating project I worked on years back. We had a complex tabbed interface where each tab's content was handled by a distinct child view controller. Initially, we tried setting constraints within `viewDidLoad()` of these child controllers. The result was a mess: views appearing off-screen, overlapping, and generally not behaving as intended. After much head scratching and breakpoint-driven debugging, we realized that the parent container view's size was not yet available in `viewDidLoad()`, thus the layout constraints were being calculated with incorrect values. This taught me, the hard way, the importance of waiting for the view hierarchy to be fully set up.

So, why `viewWillAppear(_:)` or `viewWillLayoutSubviews()` then? `viewWillAppear(_:)` is called right before the view appears on the screen, giving you the assurance that the parent’s frame is set up and available. This means that any constraints referencing the parent’s boundaries or the dimensions of other views relative to the parent are now reliable. The downside with `viewWillAppear(_:)` is that it will be called multiple times if the view controller is part of a navigation or tabbar stack (i.e., being pushed and popped from a stack), which could lead to performance issues if the view has expensive constraint manipulation code inside this method, especially if the view has already been setup.

Alternatively, `viewWillLayoutSubviews()` is called every time the view needs to re-layout its subviews. This includes things like orientation changes, or updates to constraints of other views that affect this view. If your child view's layout depends on intrinsic content sizes and content size changes, or needs to be very dynamic, then `viewWillLayoutSubviews()` can also be a good option. The main disadvantage with `viewWillLayoutSubviews()` is that, it will be called multiple times as well. Therefore it is important to ensure your constraint setup code is efficient and avoids redundant work.

Let’s illustrate this with a few code snippets:

**Example 1: Simple layout using `viewWillAppear(_:)`:**

```swift
class ChildViewController: UIViewController {

    let label = UILabel()

    override func viewDidLoad() {
        super.viewDidLoad()
        view.backgroundColor = .white
        label.text = "Hello, child!"
        label.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(label)
    }

    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        NSLayoutConstraint.activate([
            label.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            label.centerYAnchor.constraint(equalTo: view.centerYAnchor)
        ])
    }
}

class ParentViewController: UIViewController {

    let childVC = ChildViewController()

    override func viewDidLoad() {
        super.viewDidLoad()
        addChild(childVC)
        view.addSubview(childVC.view)
        childVC.didMove(toParent: self)

        childVC.view.translatesAutoresizingMaskIntoConstraints = false
        NSLayoutConstraint.activate([
            childVC.view.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            childVC.view.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            childVC.view.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor),
            childVC.view.bottomAnchor.constraint(equalTo: view.safeAreaLayoutGuide.bottomAnchor),
        ])
    }
}
```

In this scenario, the constraints for the label are set inside the child's `viewWillAppear(_:)`, ensuring the view is properly placed after the parent's view hierarchy has been setup in the `ParentViewController`. If you attempt this in the child's `viewDidLoad()`, the label might end up misaligned.

**Example 2: Complex dynamic layout with intrinsic content sizes using `viewWillLayoutSubviews()`:**

```swift
class DynamicChildViewController: UIViewController {

    let contentLabel = UILabel()
    var content: String = "" {
        didSet {
            contentLabel.text = content
            contentLabel.sizeToFit() // Ensure the label updates intrinsic size
            view.setNeedsLayout() // Force layout update
        }
    }

     override func viewDidLoad() {
        super.viewDidLoad()
        contentLabel.numberOfLines = 0
        contentLabel.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(contentLabel)
        view.backgroundColor = .systemGray6
    }


    override func viewWillLayoutSubviews() {
        super.viewWillLayoutSubviews()
         NSLayoutConstraint.activate([
            contentLabel.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 20),
            contentLabel.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -20),
            contentLabel.topAnchor.constraint(equalTo: view.topAnchor, constant: 20),
        ])
    }

}
```
Here, `viewWillLayoutSubviews()` makes more sense. The label's layout depends on the `content`'s text and how it wraps, which can change during runtime. This allows us to update layout dynamically if the `content` is updated (via the `didSet` property observer), and it also allows the view to layout properly with changes to it's parent view (e.g. during rotation).

**Example 3: Lazy constraint setup (avoiding duplicate setups) with viewWillAppear**

```swift
class LazyChildViewController: UIViewController {

    let label = UILabel()
    private var constraintsSet = false

    override func viewDidLoad() {
        super.viewDidLoad()
        view.backgroundColor = .white
        label.text = "Hello, lazy child!"
        label.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(label)
    }

    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        if !constraintsSet {
            NSLayoutConstraint.activate([
                label.centerXAnchor.constraint(equalTo: view.centerXAnchor),
                label.centerYAnchor.constraint(equalTo: view.centerYAnchor)
            ])
            constraintsSet = true
        }
    }
}
```

This example uses a boolean `constraintsSet` to ensure the constraints are applied only once, mitigating the potential for duplicate work in `viewWillAppear(_:)` if this view controller is pushed and popped within a navigation stack.

For deeper dives into view controller lifecycle and layout, I highly recommend checking out Apple’s official documentation on `UIViewController` and `UIView`. Also, the book "Effective Objective-C 2.0" by Matt Galloway (the concepts mostly translate to Swift) provides an excellent understanding of object lifecycles and memory management. Additionally, "Advanced iOS App Architecture" by Ben Scheirman, Soroush Khanlou, and Brian Gesiak dives deeper into how to better structure your layout code in more complex scenarios using more advanced techniques. Finally, the WWDC sessions regarding layout, especially those surrounding Auto Layout and UIViewController best practices, are incredibly valuable resources. Search for relevant videos on Apple’s developer website for sessions like “Building Custom Layouts with Auto Layout” and “View Controller Fundamentals in iOS”.

In short, while `viewDidLoad()` might seem like the place for layout, the correct approach is `viewWillAppear(_:)` or `viewWillLayoutSubviews()` for setting initial layout constraints in a child view controller. Choose wisely based on your specific needs. Keep in mind the potential for multiple calls to those methods and use flags as necessary. This will save you headaches and ensure your views render correctly, consistently, and reliably. From experience, it’s worth taking the time to get this part down.

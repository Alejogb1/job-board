---
title: "How to resize a container on click in Flutter?"
date: "2025-01-30"
id: "how-to-resize-a-container-on-click-in"
---
The fundamental challenge in dynamically resizing a container on a click event in Flutter stems from the framework’s reactive nature.  The UI is rebuilt based on changes to the framework's internal state; therefore, altering a container's size requires modifying a state variable that the container's dimensions are dependent on. A direct manipulation of the container's render object, while technically feasible, is not the idiomatic and robust approach. This is something I learned acutely during a previous project where performance and maintainability suffered from attempts to circumvent state-driven updates.

To resize a container on click, we must incorporate state management, typically using `StatefulWidget`. The container’s height and/or width, instead of being hardcoded values, will be bound to state variables.  A tap gesture detector will trigger a state change, and Flutter’s reactivity will rebuild the container with the new dimensions. The core idea is to make the container’s sizing dependent on the value of state variables. This method ensures UI updates are consistent and predictable and aligns well with Flutter's component-based architecture.

Here is an example demonstrating resizing a container’s height on click:

```dart
import 'package:flutter/material.dart';

class ResizableContainerExample extends StatefulWidget {
  const ResizableContainerExample({super.key});

  @override
  State<ResizableContainerExample> createState() => _ResizableContainerExampleState();
}

class _ResizableContainerExampleState extends State<ResizableContainerExample> {
  double _containerHeight = 100.0;

  void _toggleContainerHeight() {
    setState(() {
      _containerHeight = _containerHeight == 100.0 ? 200.0 : 100.0;
    });
  }

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: _toggleContainerHeight,
      child: Container(
        width: 200.0,
        height: _containerHeight,
        color: Colors.blue,
        alignment: Alignment.center,
        child: const Text(
          'Tap to Resize',
          style: TextStyle(color: Colors.white),
        ),
      ),
    );
  }
}
```

This first example uses a `StatefulWidget`, `ResizableContainerExample`. The `_containerHeight` state variable initializes the container's height to 100.0. When the `GestureDetector` is tapped, the `_toggleContainerHeight` method executes. This method updates the `_containerHeight`, using `setState` to notify Flutter to rebuild the UI.  The container, with a width of 200.0, will have its height dynamically adjust between 100.0 and 200.0 on each tap. This illustrates the fundamental principle: state change drives UI change.

Let’s consider another example, this time adding a transition animation using `AnimatedContainer`:

```dart
import 'package:flutter/material.dart';

class AnimatedResizableContainerExample extends StatefulWidget {
  const AnimatedResizableContainerExample({super.key});

  @override
  State<AnimatedResizableContainerExample> createState() => _AnimatedResizableContainerExampleState();
}

class _AnimatedResizableContainerExampleState extends State<AnimatedResizableContainerExample> {
    double _containerHeight = 100.0;
    double _containerWidth = 100.0;
    bool _isExpanded = false;

    void _toggleContainerSize() {
        setState(() {
          _isExpanded = !_isExpanded;
          _containerHeight = _isExpanded ? 200.0 : 100.0;
          _containerWidth = _isExpanded ? 200.0 : 100.0;
        });
    }

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
        onTap: _toggleContainerSize,
        child: AnimatedContainer(
          duration: const Duration(milliseconds: 300),
          curve: Curves.easeInOut,
          width: _containerWidth,
          height: _containerHeight,
          color: Colors.green,
          alignment: Alignment.center,
            child: Text(
              _isExpanded ? 'Shrink' : 'Expand',
              style: const TextStyle(color: Colors.white),
          ),
        )
    );
  }
}
```

Here, the `AnimatedContainer` replaces the standard `Container`. The `AnimatedContainer` automatically adds a smooth transition between state changes, based on `duration` and `curve` properties. Two state variables, `_containerHeight` and `_containerWidth`, control the container's dimensions. The boolean `_isExpanded` helps determine which dimensions to use, and the text also updates dynamically. The `_toggleContainerSize` method toggles `_isExpanded` and updates the dimension variables using `setState`. This enhances the user experience compared to a sudden dimension change.

Lastly, consider an example where the size change is dependent on a dynamically changing set of values:

```dart
import 'package:flutter/material.dart';

class DynamicSizeContainerExample extends StatefulWidget {
  const DynamicSizeContainerExample({super.key});

  @override
  State<DynamicSizeContainerExample> createState() => _DynamicSizeContainerExampleState();
}

class _DynamicSizeContainerExampleState extends State<DynamicSizeContainerExample> {
  final List<double> _sizes = [100.0, 150.0, 200.0, 150.0, 100.0];
  int _currentIndex = 0;

  void _updateContainerSize(){
    setState(() {
      _currentIndex = (_currentIndex + 1) % _sizes.length;
    });
  }

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: _updateContainerSize,
      child: Container(
        width: 200.0,
        height: _sizes[_currentIndex],
        color: Colors.purple,
        alignment: Alignment.center,
        child: Text(
          'Size: ${_sizes[_currentIndex]}',
           style: const TextStyle(color: Colors.white)
          ),
        ),
    );
  }
}
```

In this third scenario, the container's height is not just toggled between two values, but rather cycles through a list of sizes, `_sizes`.  `_currentIndex` tracks which size from the list should be applied to the container's height. The modulo operator (`%`) ensures the index cycles back to 0 when the end of the list is reached. The `Text` widget dynamically displays the current container size. This demonstrates how the state can also be an index into a list of options, not merely a binary toggle.

In summary, the core mechanism for resizing a container on click involves managing state, specifically with state variables for the dimensions within a `StatefulWidget`. The `GestureDetector` detects taps and triggers state updates using `setState`, which in turn causes Flutter to rebuild the UI. The `AnimatedContainer` provides smooth transitions, and lists of dynamic values expand the design possibilities. Avoiding direct manipulation of rendering objects, and embracing Flutter’s reactive paradigm through state management will yield more maintainable and robust code.

Further exploration into the Flutter framework would be beneficial. The documentation for `StatefulWidget`, `GestureDetector`, `AnimatedContainer`, and the overall state management principles are crucial for understanding the broader implications of each component. Additionally, investigating state management libraries like Provider, Riverpod, or BLoC can be beneficial for larger applications requiring more complex state management strategies. These offer sophisticated patterns for managing more complex relationships between components and state. These topics, coupled with practical experience, are fundamental to building complex and dynamic Flutter applications.

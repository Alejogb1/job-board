---
title: "Why does repainting not reflect changes made to a paintComponent variable via a setter?"
date: "2025-01-30"
id: "why-does-repainting-not-reflect-changes-made-to"
---
The core issue stems from the Swing event dispatch thread (EDT) and the inherent asynchronous nature of Swing painting.  Changes to a `paintComponent`'s internal state, even through a setter method, are not guaranteed to trigger an immediate repaint.  This is a common source of confusion for developers new to Swing, and I've personally debugged countless instances of this over my years developing Java desktop applications. The `repaint()` method, while seemingly straightforward, operates within the context of the EDT and its scheduling mechanism.  Let's clarify this behavior and examine solutions.

**1. Explanation: The EDT and Repaint Mechanism**

Swing, for reasons of thread safety, mandates that all UI updates occur on the EDT.  Direct manipulation of UI components from other threads results in undefined behavior, often leading to exceptions or visual inconsistencies.  The `repaint()` method doesn't directly redraw the component; instead, it queues a repaint request on the EDT.  The EDT then processes these requests at its leisure, balancing them with other UI events.  If you modify a variable within a setter method, and that variable is used within `paintComponent`, the change won't be immediately reflected unless a `repaint()` call is explicitly made *after* updating the variable. This is crucial because the `paintComponent` method is called *only* when the system deems it necessary, typically after a `repaint()` request.  Simply changing the variable does not automatically trigger a repaint; it only changes the underlying data.  The painting process remains oblivious to these data changes until the `paintComponent` method is explicitly invoked.

**2. Code Examples with Commentary**

**Example 1: Incorrect Implementation**

```java
import javax.swing.*;
import java.awt.*;

public class IncorrectRepaint extends JPanel {

    private Color fillColor = Color.RED;

    public void setFillColor(Color color) {
        this.fillColor = color;
    }

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        g.setColor(fillColor);
        g.fillRect(0, 0, getWidth(), getHeight());
    }

    public static void main(String[] args) {
        JFrame frame = new JFrame("Incorrect Repaint");
        IncorrectRepaint panel = new IncorrectRepaint();
        frame.add(panel);
        frame.setSize(300, 200);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setVisible(true);

        panel.setFillColor(Color.BLUE); // No repaint!
    }
}
```

In this example, changing `fillColor` to blue has no visible effect.  `paintComponent` uses the old value because no `repaint()` call was made following the setter invocation.

**Example 2: Correct Implementation Using `repaint()`**

```java
import javax.swing.*;
import java.awt.*;

public class CorrectRepaint extends JPanel {

    private Color fillColor = Color.RED;

    public void setFillColor(Color color) {
        this.fillColor = color;
        repaint(); // Crucial addition!
    }

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        g.setColor(fillColor);
        g.fillRect(0, 0, getWidth(), getHeight());
    }

    public static void main(String[] args) {
        JFrame frame = new JFrame("Correct Repaint");
        CorrectRepaint panel = new CorrectRepaint();
        frame.add(panel);
        frame.setSize(300, 200);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setVisible(true);

        panel.setFillColor(Color.BLUE); // Repaint is now triggered.
    }
}
```

Here, the addition of `repaint()` within the `setFillColor` method ensures that the updated `fillColor` is used in the subsequent `paintComponent` call. This is the fundamental solution.

**Example 3:  Handling Updates from Another Thread**

```java
import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

public class ThreadedRepaint extends JPanel {

    private Color fillColor = Color.RED;

    public void setFillColor(final Color color) {
        SwingUtilities.invokeLater(new Runnable() {
            @Override
            public void run() {
                fillColor = color;
                repaint();
            }
        });
    }

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        g.setColor(fillColor);
        g.fillRect(0, 0, getWidth(), getHeight());
    }

    public static void main(String[] args) {
        JFrame frame = new JFrame("Threaded Repaint");
        final ThreadedRepaint panel = new ThreadedRepaint();
        frame.add(panel);
        frame.setSize(300, 200);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setVisible(true);

        JButton button = new JButton("Change Color");
        button.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                panel.setFillColor(Color.GREEN); // Update from another thread
            }
        });
        frame.add(button, BorderLayout.SOUTH);
    }
}
```

This example demonstrates a crucial aspect:  updating the UI from a thread other than the EDT. `SwingUtilities.invokeLater()` ensures that the `setFillColor` operation and subsequent `repaint()` are executed on the EDT, preventing thread safety issues.  I've encountered numerous scenarios where neglecting this led to unpredictable results and crashes.  Using `invokeLater` is paramount when dealing with asynchronous tasks or external events impacting the UI state.

**3. Resource Recommendations**

I strongly recommend consulting the official Java documentation on Swing, specifically the sections on painting, event dispatch thread, and `repaint()`. Thoroughly studying these will significantly improve your understanding of Swing's intricacies.  Furthermore, I've personally found comprehensive Swing tutorials and books invaluable in solidifying my understanding.  A good book on concurrent programming in Java is also beneficial, particularly when dealing with threading and UI updates. These resources will help you grasp the underlying principles and avoid common pitfalls.  Remember, consistent practice and diligent debugging are crucial for mastering Swing.

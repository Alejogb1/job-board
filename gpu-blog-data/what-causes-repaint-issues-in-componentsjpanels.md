---
title: "What causes repaint() issues in components/JPanels?"
date: "2025-01-30"
id: "what-causes-repaint-issues-in-componentsjpanels"
---
The most frequent cause of unexpected or erratic `repaint()` behavior in Swing components, particularly `JPanel` instances, stems from a misunderstanding of the Swing event dispatch thread (EDT) and the non-thread-safe nature of Swing's graphical operations. I've spent considerable time debugging complex UIs, often finding that the culprit lies not within the repaint logic itself but in its improper invocation outside the EDT.

Swing, being a single-threaded GUI framework, mandates that all updates to its components, including calls to `repaint()`, must originate from the EDT. This is a crucial concept because graphical components internally manage their state, and concurrent modifications from multiple threads can lead to inconsistent or corrupted rendering, manifesting as flickering, incorrect drawing, or even application hangs. When we directly invoke `repaint()` from a worker thread or any thread other than the EDT, we are effectively violating Swing’s threading model. This results in the framework trying to synchronize the changes which often doesn’t work as intended.

The `repaint()` method itself does not immediately force the component to redraw. Instead, it signals to Swing that a redraw is needed. This signal is added to the EDT’s event queue, and Swing, when it is its turn to process the event, will invoke the paint methods. Several factors influence when this actual drawing occurs. First, the number of pending events on the EDT queue affects the responsiveness to repaint requests. If the EDT is bogged down with other processing tasks, repaint requests will be delayed. Second, the Swing framework optimises repaint requests. If multiple `repaint()` calls occur before the EDT gets to the paint method, Swing consolidates them into a single repaint of the smallest bounding rectangle that encapsulates all updated areas. Third, nested calls to `repaint()` can lead to more delays if they occur within each other.

Another reason for suboptimal `repaint()` behavior stems from not implementing custom painting within the correct method. In `JPanel` or its subclasses, we should override the `paintComponent(Graphics g)` method and avoid overriding the `paint(Graphics g)` method unless you completely intend to manage the painting of the container's border, which is rare. Failure to adhere to this convention can lead to unexpected rendering artifacts, particularly if the component needs to maintain its background fill. When custom painting is incorrectly implemented on the `paint(Graphics g)` method, the component's background might not be correctly filled, resulting in painting issues.

Additionally, modifications to the underlying data that a custom paint method utilizes, if not synchronized with the painting process, can cause incorrect visuals. Consider for example a component that draws a graph where the underlying dataset is modified in a different thread, while the painting is done in EDT. Without proper data synchronization mechanism, such as lock or volatile fields, a race condition will cause the paint to use an outdated or partially modified dataset which is causing inconsistent rendering issues.

Finally, not utilizing the `setOpaque(true)` method appropriately impacts repaint. When a component is opaque, the graphics context will clear the background before rendering the component to ensure no artefacts remains from the previous drawing cycle. If you wish to create custom backgrounds with a transparent color, consider that component may need to repaint more than usual if the parent component is repainted.

To illustrate these concepts, let’s consider the following three examples. The first one will demonstrate the issue of calling `repaint()` from non-EDT, the second will show how to make a thread-safe repaint by using the `SwingUtilities.invokeLater()`, and the third will show how to do custom painting.

**Example 1: Incorrect `repaint()` Invocation from a Worker Thread**

```java
import javax.swing.*;

public class RepaintIssue extends JFrame {
    private JPanel panel;

    public RepaintIssue() {
        panel = new JPanel() {
            @Override
            protected void paintComponent(java.awt.Graphics g) {
                super.paintComponent(g);
                g.fillRect(50, 50, 100, 100);
            }
        };
        add(panel);
        setSize(300, 300);
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setVisible(true);

        new Thread(() -> {
            try {
                Thread.sleep(2000);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
            panel.repaint(); // INCORRECT: Invoking repaint from non-EDT
        }).start();
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(RepaintIssue::new);
    }
}
```

In this example, a simple `JPanel` draws a rectangle. After a two-second delay, a separate thread calls `panel.repaint()`. Because this call occurs outside the EDT, you might observe flickering, screen tearing, or an incorrect rendering behaviour, particularly on operating systems that are more strict with threading. While the code might *appear* to work on some platforms, it is fundamentally flawed and can be unpredictable in the long run. This demonstrates how we violate Swing's threading policy by calling `repaint()` directly from a different thread.

**Example 2: Correct `repaint()` Invocation using `SwingUtilities.invokeLater()`**

```java
import javax.swing.*;

public class CorrectRepaint extends JFrame {
    private JPanel panel;

    public CorrectRepaint() {
         panel = new JPanel() {
            @Override
            protected void paintComponent(java.awt.Graphics g) {
                super.paintComponent(g);
                g.fillRect(50, 50, 100, 100);
            }
        };
         add(panel);
        setSize(300, 300);
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setVisible(true);

        new Thread(() -> {
            try {
                Thread.sleep(2000);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
           SwingUtilities.invokeLater(() -> panel.repaint()); // CORRECT: Invoking repaint on EDT
        }).start();
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(CorrectRepaint::new);
    }
}
```

This modified version addresses the previous issue by wrapping the `panel.repaint()` call within `SwingUtilities.invokeLater()`. This ensures that the `repaint()` call executes on the EDT, respecting Swing's threading rules. The lambda expression `() -> panel.repaint()` represents a task that the EDT will execute at the next opportunity after it finishes with all the other events it's currently processing. As a result, the repaint will be executed in a thread-safe manner, eliminating any thread related visual glitches.

**Example 3: Custom Painting and `paintComponent`**

```java
import javax.swing.*;
import java.awt.*;
import java.util.Random;

public class CustomPainting extends JFrame {
    private JPanel panel;
    private int[] dataPoints;
    private final Random random = new Random();

    public CustomPainting() {
        dataPoints = generateData(100);

        panel = new JPanel() {
            @Override
            protected void paintComponent(Graphics g) {
                super.paintComponent(g);
                g.setColor(Color.BLUE);
                int panelHeight = getHeight();
                int panelWidth = getWidth();
                 int xStep = panelWidth/dataPoints.length;
                 int x = 0;

                for(int y : dataPoints) {
                    g.fillRect(x, panelHeight - y, 5, y);
                    x += xStep;
                }
             }
        };

        add(panel);
        setSize(500, 300);
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setVisible(true);

        Timer timer = new Timer(1000, e -> {
            updateData();
            panel.repaint();
        });
         timer.start();
    }

    private int[] generateData(int count) {
        int[] result = new int[count];
        for(int i = 0; i< count; i++){
            result[i] = random.nextInt(getHeight()- 20);
        }
        return result;
    }

    private void updateData() {
       for(int i = 0; i < dataPoints.length; i++){
            dataPoints[i] = random.nextInt(getHeight()-20);
        }
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(CustomPainting::new);
    }
}
```

This example goes further and demonstrates custom painting by generating a histogram. It uses `paintComponent` to draw the bars based on the `dataPoints` array. Additionally, a `javax.swing.Timer` is utilized to modify the data and invoke `repaint()` periodically. This example shows a basic setup of how data updates are reflected on the UI through `repaint()` after being properly managed to occur on the EDT. Note that the updateData method is invoked on the EDT by the timer, which is why the data can be modified directly. If the data was coming from a non-EDT thread, a proper locking mechanism would be necessary.

For further learning, I would highly recommend reviewing documentation on the Swing EDT, including any official resources available for Swing's threading model. Understanding the nuances of how and when the paint methods are invoked is critical for writing applications that perform well and do not exhibit rendering errors. I also suggest reviewing resources detailing `SwingUtilities.invokeLater()` and `SwingUtilities.invokeAndWait()` for managing EDT related tasks, and how to perform custom drawing. Finally, be sure to look into best practices in managing data that is displayed in swing components, paying particular attention to threading and concurrent access issues that can arise when dealing with non-EDT data manipulation.

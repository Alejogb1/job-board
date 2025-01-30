---
title: "Why doesn't the Paint event fire with FlatStyle.System?"
date: "2025-01-30"
id: "why-doesnt-the-paint-event-fire-with-flatstylesystem"
---
The core issue is that when a `Control`’s `FlatStyle` property is set to `FlatStyle.System`, the control relinquishes the responsibility for its visual rendering to the underlying operating system's native theming engine. This significantly alters the traditional Windows Forms drawing pipeline, thus preventing the standard `Paint` event from being reliably triggered. I've encountered this hurdle numerous times, particularly when attempting custom drawing enhancements on controls utilizing system themes within enterprise applications. The `Paint` event, in a typical scenario, provides a canvas for drawing operations performed by the application code. However, system-themed controls delegate this task entirely to the operating system, and the window message that would normally trigger the `Paint` event gets intercepted and handled differently.

The mechanism at play involves Windows’ native drawing API, often referred to as the “visual styles” API. When `FlatStyle.System` is enabled, Windows draws the control’s appearance using pre-defined style information specific to the user's chosen theme (e.g., Classic, Aero, or newer Modern styles). This means that any custom rendering efforts made via the control’s `Paint` event are effectively bypassed because the control itself doesn't perform its own drawing. The Windows theming engine paints directly on the window handle, using its internal mechanisms for generating the correct visual style.

The standard drawing lifecycle for a Windows Forms control, when *not* using `FlatStyle.System`, begins with the `WM_PAINT` message. This message notifies the control that it needs to redraw its surface. The framework’s handling of this message triggers the `OnPaint` method, which subsequently raises the `Paint` event. Event handlers attached to the `Paint` event can then execute custom drawing logic. However, when the `FlatStyle` is set to `FlatStyle.System`, the control indicates to Windows that it will not handle its own visual appearance. Instead, Windows directly manages the drawing, and the `WM_PAINT` message is processed in a theme-aware manner, bypassing the typical sequence of events related to custom drawing.

I've found it's particularly important to understand this distinction when migrating legacy applications that relied on heavy custom painting. The migration path to new operating systems with updated visual styles often involves an accidental adoption of `FlatStyle.System` due to default configurations or developer preference for UI that matches the underlying operating system. The result can be unexpected failures of custom drawing routines and confusion regarding why the `Paint` event does not fire, as would normally be expected. This usually leads to a need for a thorough examination of the control's `FlatStyle` property and an appropriate rewrite of the affected code to accommodate the new drawing mechanism.

Here are a few scenarios illustrating this behavior and suggesting potential workarounds:

**Example 1: Illustrating the Missing `Paint` Event**

```Csharp
using System;
using System.Drawing;
using System.Windows.Forms;

public class FlatStyleSystemTest : Form
{
    private Button myButton;

    public FlatStyleSystemTest()
    {
        myButton = new Button();
        myButton.Text = "Test Button";
        myButton.FlatStyle = FlatStyle.System; // Key configuration
        myButton.Size = new Size(100, 30);
        myButton.Location = new Point(50, 50);
        myButton.Paint += new PaintEventHandler(MyButton_Paint); // Trying to add custom drawing
        Controls.Add(myButton);

        this.ClientSize = new Size(200, 150);
        this.Text = "Flat Style System Paint Test";
    }


    private void MyButton_Paint(object sender, PaintEventArgs e)
    {
        e.Graphics.FillRectangle(Brushes.Red, 0, 0, myButton.Width, myButton.Height);
        e.Graphics.DrawString("Custom Paint", this.Font, Brushes.White, 5, 5);
        //This code will NOT execute due to FlatStyle.System
    }


    public static void Main(string[] args)
    {
        Application.Run(new FlatStyleSystemTest());
    }
}
```

In this example, the `MyButton_Paint` event handler, which attempts to draw a red rectangle and text, will *not* be executed. The button will render using the native visual style of the operating system. The `FlatStyle.System` configuration prevents the custom drawing code from being triggered. This illustrates the primary problem. The `Paint` event is not directly coupled with how the control is rendered by Windows; it is a separate mechanism that is bypassed in the case of `FlatStyle.System`.

**Example 2: Workaround Using Custom Controls (Minimal)**

```Csharp
using System;
using System.Drawing;
using System.Windows.Forms;

public class CustomButton : Control
{
  
    public CustomButton()
    {
       this.SetStyle(ControlStyles.AllPaintingInWmPaint | ControlStyles.UserPaint | ControlStyles.OptimizedDoubleBuffer, true);
    }
    protected override void OnPaint(PaintEventArgs e)
    {
      base.OnPaint(e);
      e.Graphics.FillRectangle(Brushes.Blue, 0, 0, this.Width, this.Height);
       e.Graphics.DrawString("Custom Painted", this.Font, Brushes.White, 5, 5);
    }
}


public class CustomControlTest : Form
{
    private CustomButton customButton;

    public CustomControlTest()
    {
      customButton = new CustomButton();
      customButton.Size = new Size(150,40);
      customButton.Location = new Point(50,50);
      Controls.Add(customButton);

        this.ClientSize = new Size(300, 200);
        this.Text = "Custom Control Paint Test";
    }
     public static void Main(string[] args)
    {
        Application.Run(new CustomControlTest());
    }
}
```

Here, a custom `Control` derived class, `CustomButton`, forces the execution of its `OnPaint` event handler and is able to draw its own content. This is the approach to take when needing custom drawing, essentially by bypassing Windows Forms prebuilt control rendering. The control style settings, which include `UserPaint`, direct Windows that your custom code manages drawing operations. The control's appearance is now directly governed by the custom `OnPaint` implementation.

**Example 3: Alternative Strategy: Using the SystemDraw API**

```Csharp
using System;
using System.Drawing;
using System.Windows.Forms;
using System.Runtime.InteropServices;

public class SystemDrawButton : Button
{
 [DllImport("uxtheme.dll", CharSet = CharSet.Unicode, ExactSpelling = true)]
    private static extern int DrawThemeBackground(IntPtr hTheme, IntPtr hdc, int iPartId, int iStateId, ref RECT pRect, IntPtr pClipRect);

[StructLayout(LayoutKind.Sequential)]
        public struct RECT
        {
            public int Left;
            public int Top;
            public int Right;
            public int Bottom;
        }

        [DllImport("uxtheme.dll", ExactSpelling = true, CharSet = CharSet.Unicode)]
        private static extern IntPtr OpenThemeData(IntPtr hWnd, string pszClassList);

        [DllImport("uxtheme.dll", ExactSpelling = true)]
        private static extern int CloseThemeData(IntPtr hTheme);

    private bool _customPaint = false;
    public bool CustomPaint{ get{return _customPaint;} set {_customPaint = value; this.Invalidate();}}
    public SystemDrawButton()
    {
        this.FlatStyle = FlatStyle.System;
    }
      protected override void OnPaint(PaintEventArgs e)
    {
        if(!CustomPaint) {base.OnPaint(e); return;}


        IntPtr themeHandle = OpenThemeData(this.Handle,"Button");
       if(themeHandle != IntPtr.Zero)
        {
            RECT rect = new RECT() { Left = 0, Top = 0, Right = this.Width, Bottom = this.Height };

           DrawThemeBackground(themeHandle, e.Graphics.GetHdc(), 1, 1, ref rect, IntPtr.Zero);

           e.Graphics.ReleaseHdc();
            CloseThemeData(themeHandle);
            e.Graphics.DrawString("Custom Paint over System", this.Font, Brushes.White, 5, 5);

        }
    }
}

public class SystemDrawTest : Form {

    private SystemDrawButton systemButton;
     public SystemDrawTest()
    {
      systemButton = new SystemDrawButton();
      systemButton.Size = new Size(150,40);
      systemButton.Location = new Point(50,50);
      systemButton.Text = "System Button";
      systemButton.CustomPaint = true;
      Controls.Add(systemButton);


        this.ClientSize = new Size(300, 200);
        this.Text = "System Draw Test";
    }
     public static void Main(string[] args)
    {
        Application.Run(new SystemDrawTest());
    }
}
```

This example shows a more complex approach where we invoke the theming API directly to draw the button's background. This involves platform invoke (`DllImport`) to utilize the `uxtheme.dll` for theme rendering. This demonstrates how custom drawing can be combined with system themes. The `SystemDrawButton` draws the native theme, but still manages to add the “Custom Paint over System” string. This method can be used when needing to augment, rather than replace, the native theme. The `CustomPaint` property allows you to choose whether to employ the native system theming or to inject the custom draw.

When working with Windows Forms and encountering drawing challenges related to `FlatStyle.System`, it's crucial to understand the underlying mechanism. The direct involvement of the operating system for theme rendering means that the standard `Paint` event won’t be a feasible hook for custom drawing. The key lies in deciding whether your approach is to replace native rendering with a custom solution or augment it. For replacing, use a custom control. For supplementing native rendering, the system theming API is the route to take.

For further reading, research “Windows Visual Styles API” and “Windows Forms Control Styles” to develop a greater understanding of how to approach custom drawing scenarios. Specific books on Windows Forms custom control development can also offer in-depth information on handling scenarios like the ones discussed. Numerous articles describing custom painting patterns for Windows applications are available through online searches as well; they focus on overcoming the hurdle presented by controls with `FlatStyle.System` enabled. Consulting the Microsoft documentation directly is a good starting point to better understand these concepts.

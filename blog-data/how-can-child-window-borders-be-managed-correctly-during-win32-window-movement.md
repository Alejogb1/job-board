---
title: "How can child window borders be managed correctly during Win32 window movement?"
date: "2024-12-23"
id: "how-can-child-window-borders-be-managed-correctly-during-win32-window-movement"
---

Let’s tackle this one. I remember a project back in '08; a rather complex multi-document interface application that pushed the boundaries of what we thought was acceptable with child windows within a single parent. Border management? Oh, that was a frequent headache, and it’s surprisingly easy to get it wrong. Here's what I've learned over time, broken down into manageable parts:

The crux of the issue with managing child window borders during Win32 parent window movements comes down to the fact that child windows don't inherently "know" when their parent has moved. The operating system doesn't automatically adjust the child window's client area or borders when the parent is repositioned. It's crucial to recognize that child windows are positioned relative to their parent's client area, *not* the screen directly. Consequently, when the parent moves, the child appears to move relative to the parent, but its actual pixel-based coordinates in screen space have changed. The visual consequence? Misaligned borders, partially obscured controls, and just general visual chaos.

To properly manage this, we need to understand how Windows messaging works with window positioning and utilize that to our advantage. Specifically, the `WM_MOVING` and `WM_MOVE` messages sent to a parent window are our primary tools here, coupled with careful attention to coordinate systems and how we interact with `SetWindowPos` or similar functions. These messages are not sent to the children directly, so we have to explicitly handle them in the parent and propagate positional changes.

First, let’s talk about the `WM_MOVING` message. This message gives us an opportunity *before* the window is moved to either modify the movement parameters or cancel the movement altogether. We often don’t need to actually alter the movement here, just prepare. Crucially, after `WM_MOVING`, comes `WM_MOVE`, and it is within the handler for `WM_MOVE` that we should focus on repositioning child windows relative to the new parent window location. Doing it in `WM_MOVING` tends to introduce issues with incomplete data and unwanted side effects.

Now, here’s a critical detail: You must account for different border styles of the parent window when calculating where to move the children. A window with a sizable border will have a different client area origin and offset relative to the main window frame than one with no border. This isn’t just an aesthetic concern; it impacts the entire calculated position of children.

Here's a working example in a pseudo-C++ style, showcasing the handler for a hypothetical `MainWindow` class which owns child windows of type `ChildWindow`.

```cpp
LRESULT MainWindow::WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
  switch (uMsg) {
    case WM_MOVE:
    {
      RECT parentRect;
      GetClientRect(hwnd, &parentRect); // Get client area of the parent window
      POINT parentClientOrigin;
      parentClientOrigin.x = parentRect.left;
      parentClientOrigin.y = parentRect.top;
      ClientToScreen(hwnd, &parentClientOrigin); // Convert to screen coordinates

      for (auto& child : children) {
          RECT childRect;
          GetWindowRect(child->GetHwnd(), &childRect); // Get window coordinates of the child
           // Child position relative to the parent's client area
          int relativeX = childRect.left - parentClientOrigin.x;
          int relativeY = childRect.top - parentClientOrigin.y;

          // Calculate new screen position based on the parent's new position
          int newScreenX = parentClientOrigin.x + relativeX;
          int newScreenY = parentClientOrigin.y + relativeY;

          SetWindowPos(child->GetHwnd(), nullptr, newScreenX, newScreenY, 0, 0,
                       SWP_NOSIZE | SWP_NOZORDER); // Move child window to new position.
      }
      return 0;
     }

    default:
        return DefWindowProc(hwnd, uMsg, wParam, lParam);
  }
}
```

This example retrieves the parent window’s client area, converts it to screen coordinates, and then iterates through a collection of child windows. For each child, it retrieves the child's window rectangle, calculates its position relative to the parent's client area, adjusts the coordinates to reflect the moved parent window, and uses `SetWindowPos` to move it. It's imperative to use the `SWP_NOSIZE` and `SWP_NOZORDER` flags to prevent unintentional resizing or z-order changes, which are also common pitfalls.

Let's look at a slightly more sophisticated scenario. Suppose you're managing a series of child windows that are meant to have a fixed margin relative to the parent client area on all sides.

```cpp
LRESULT MainWindow::WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
  switch (uMsg) {
    case WM_MOVE:
     {
       RECT parentRect;
       GetClientRect(hwnd, &parentRect);
       POINT parentClientOrigin;
       parentClientOrigin.x = parentRect.left;
       parentClientOrigin.y = parentRect.top;
       ClientToScreen(hwnd, &parentClientOrigin);
       int margin = 10; // Our fixed margin, e.g. 10 pixels

       for (auto& child : children) {
           int childWidth = 100; // Assume static width and height for this example
           int childHeight = 50;

           // New position with our fixed margins
           int newScreenX = parentClientOrigin.x + margin;
           int newScreenY = parentClientOrigin.y + margin;

           // Set position, and in this case, also resize
           SetWindowPos(child->GetHwnd(), nullptr, newScreenX, newScreenY,
                         childWidth, childHeight, SWP_NOZORDER);
        }
       return 0;
     }
      default:
         return DefWindowProc(hwnd, uMsg, wParam, lParam);
  }
}

```

In this example, we’ve introduced a margin. The child windows will now always maintain a constant distance from the edges of the parent's client area. Note that `SetWindowPos` now includes a specific size, which demonstrates its full capacity.

Finally, there may be instances where some child windows might also have internal repositioning logic or have to react to the parent's movement based on their unique logic. If, for example, we want to maintain a dynamic child window that is aligned relative to a corner, but also maintains a minimal size while the parent is being dragged and resized (that is covered in WM_SIZING and WM_SIZE, so keep in mind you also have to handle those).

```cpp
LRESULT MainWindow::WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
  switch (uMsg) {
        case WM_MOVE:
          {
             RECT parentRect;
             GetClientRect(hwnd, &parentRect);
             POINT parentClientOrigin;
             parentClientOrigin.x = parentRect.left;
             parentClientOrigin.y = parentRect.top;
             ClientToScreen(hwnd, &parentClientOrigin);


             for (auto& child : children) {
                //Get child dimensions
                 RECT childRect;
                 GetWindowRect(child->GetHwnd(),&childRect);
                 int childWidth = childRect.right - childRect.left;
                 int childHeight = childRect.bottom - childRect.top;
                // Position the child in the bottom-right corner
                 int newScreenX = parentClientOrigin.x + (parentRect.right - parentRect.left) - childWidth - 10;
                 int newScreenY = parentClientOrigin.y + (parentRect.bottom - parentRect.top) - childHeight - 10;

                 SetWindowPos(child->GetHwnd(), nullptr, newScreenX, newScreenY,
                             0, 0, SWP_NOSIZE | SWP_NOZORDER); // Move child window to the bottom-right.
              }
              return 0;
           }
           default:
            return DefWindowProc(hwnd, uMsg, wParam, lParam);
    }
}

```
In this scenario, the child is dynamically repositioned to maintain a position in the bottom-right corner of the parent window, even though the parent window moves. Note that these examples are using `GetClientRect` or `GetWindowRect` because they are more precise in terms of client space calculations. This detail is important to consider given all window styles impact on the size of the client area.

When considering additional reading on this, I’d recommend starting with Charles Petzold’s “Programming Windows,” a foundational text on the Windows API. Also, the official Microsoft documentation on window management and Win32 messaging is absolutely indispensable. Look specifically into the documentation for `WM_MOVE`, `WM_MOVING`, `SetWindowPos`, and `GetClientRect`/`GetWindowRect`. Careful study of these resources and understanding coordinate system conversion using `ClientToScreen` will resolve the majority of issues related to child window management.

Finally, remember to test thoroughly across different DPI settings and themes. Differences in display settings can sometimes reveal subtle errors, but with a solid understanding of these principles, you should be well-equipped to handle the intricacies of window border management in Win32.

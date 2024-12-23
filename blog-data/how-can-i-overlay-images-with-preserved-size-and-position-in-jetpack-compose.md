---
title: "How can I overlay images with preserved size and position in Jetpack Compose?"
date: "2024-12-23"
id: "how-can-i-overlay-images-with-preserved-size-and-position-in-jetpack-compose"
---

Okay, let's tackle this one. Image overlays, particularly maintaining precise size and position in Compose, are something I've dealt with extensively, especially in mobile game UI development a few years back. We needed to render character sprites over backgrounds without constant recalculations and weird scaling. It wasn't as straightforward as it initially seemed. Let's get into the details.

The core challenge isn't just about layering; it's about ensuring the overlay maintains its intended dimensions and relative positioning despite varying screen sizes or parent layout constraints. Compose offers several tools to accomplish this, but understanding how they interact is key to avoiding common pitfalls. Fundamentally, we're going to be using `Box` to stack elements, and leveraging modifiers like `offset`, `size`, and potentially `layout` modifiers for more control. However, `Modifier.size` alone might not always give us the pixel-perfect control we need, especially when dealing with different device densities. So, let's break it down.

First, the basic approach using `Box` is almost always the starting point. It provides the foundation for stacking elements on top of each other. Here's the most straightforward version:

```kotlin
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.size
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.res.imageResource
import androidx.compose.ui.unit.dp

@Composable
fun SimpleOverlay(backgroundRes: Int, overlayRes: Int) {
    Box {
        Image(
            bitmap = ImageBitmap.imageResource(backgroundRes),
            contentDescription = "Background"
        )
        Image(
            bitmap = ImageBitmap.imageResource(overlayRes),
            contentDescription = "Overlay",
            modifier = Modifier
                .size(100.dp, 100.dp)
                .offset(50.dp, 50.dp)
        )
    }
}
```

In this snippet, the `Box` allows us to render a background image first, and then an overlay image on top using the `Image` composable. The `size` modifier ensures the overlay has explicit dimensions and `offset` sets its position relative to the top-left corner of the `Box`. This works well for fixed sizes and positions, but it falters when dynamic sizing or position adjustments relative to the background image’s boundaries are needed.

The issue with static offset values is that they don’t scale well with different backgrounds. Let's suppose you wanted an overlay consistently positioned 20% from the top and 30% from the left, regardless of the background size. For that, we'll delve a little deeper. We'll need to use `Modifier.layout` to access the parent composable’s layout information. Let’s look at an example which calculates the overlay position as a percentage of the background image size. This example also adds scaling to the overlay:

```kotlin
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Box
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.layout.layout
import androidx.compose.ui.res.imageResource
import androidx.compose.ui.unit.IntOffset
import androidx.compose.ui.unit.dp

@Composable
fun PercentageOverlay(backgroundRes: Int, overlayRes: Int, overlayWidth: Float = 0.2f, overlayHeight: Float = 0.2f, offsetX: Float = 0.2f, offsetY: Float = 0.2f) {
    Box {
        Image(
            bitmap = ImageBitmap.imageResource(backgroundRes),
            contentDescription = "Background",
            modifier = Modifier
                .layout { measurable, constraints ->
                    val placeable = measurable.measure(constraints)
                    layout(placeable.width, placeable.height) {
                        placeable.placeRelative(0, 0)
                    }
                }
        )
       
        Image(
            bitmap = ImageBitmap.imageResource(overlayRes),
            contentDescription = "Overlay",
            modifier = Modifier
                .layout { measurable, constraints ->

                    val parentWidth = this.parentData?.layoutInfo?.width ?: constraints.maxWidth
                    val parentHeight = this.parentData?.layoutInfo?.height ?: constraints.maxHeight

                    val overlayPlaceable = measurable.measure(constraints.copy(maxWidth = (parentWidth * overlayWidth).toInt(), maxHeight = (parentHeight * overlayHeight).toInt()))

                     layout(overlayPlaceable.width, overlayPlaceable.height) {
                       val x = (parentWidth * offsetX).toInt()
                       val y = (parentHeight * offsetY).toInt()

                       overlayPlaceable.placeRelative(IntOffset(x, y))
                     }
                  }
            )
    }
}

```

In this example, I’m using `Modifier.layout` on both the background and the overlay. The layout modifier for the background calculates its actual size which is required in order to calculate the relative positions for the overlay using the provided percentage values `offsetX` and `offsetY`. The overlay layout modifier receives layout information about the parent (the Box) through the `parentData` attribute and uses this information to calculate the overlay dimensions, applying the `overlayWidth` and `overlayHeight` percentage scale values to size the overlay image. Finally, the positioning is handled within the same modifier based on the `offsetX` and `offsetY` parameters. This setup creates an overlay that's proportionally positioned and scaled based on the background image size.

It is worth noting here that if we did not use the parentData object to get the size of the `Box` in this second example, Compose would be unable to accurately render the overlay. `constraints.maxHeight` and `constraints.maxWidth` would be the size of the view that contains the entire composable, not just the `Box` that acts as the parent of our overlay. This is an important distinction to make when dealing with layered composables.

There’s one more scenario we should consider: what if you have multiple images layered on top of each other where their size and position must be relative to the immediate parent. For this, we can use different boxes as parents, and the same layout logic. Here is the code snippet:

```kotlin
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Box
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.layout.layout
import androidx.compose.ui.res.imageResource
import androidx.compose.ui.unit.IntOffset

@Composable
fun MultiLayeredOverlay(backgroundRes: Int, overlay1Res: Int, overlay2Res: Int) {
    Box {
        Image(
            bitmap = ImageBitmap.imageResource(backgroundRes),
            contentDescription = "Background",
             modifier = Modifier
                .layout { measurable, constraints ->
                    val placeable = measurable.measure(constraints)
                    layout(placeable.width, placeable.height) {
                        placeable.placeRelative(0, 0)
                    }
                }
        )
        Box(modifier = Modifier
        .layout { measurable, constraints ->
           
             val parentWidth = this.parentData?.layoutInfo?.width ?: constraints.maxWidth
             val parentHeight = this.parentData?.layoutInfo?.height ?: constraints.maxHeight
              val placeable = measurable.measure(constraints.copy(maxWidth = (parentWidth * 0.5f).toInt(), maxHeight = (parentHeight * 0.5f).toInt()))
            
            layout(placeable.width, placeable.height) {
                 val x = (parentWidth * 0.25f).toInt()
                 val y = (parentHeight * 0.25f).toInt()
                placeable.placeRelative(IntOffset(x, y))
            }
        }){
            Image(
                bitmap = ImageBitmap.imageResource(overlay1Res),
                contentDescription = "Overlay 1"
             )
         }

         Box(modifier = Modifier
        .layout { measurable, constraints ->
           
             val parentWidth = this.parentData?.layoutInfo?.width ?: constraints.maxWidth
             val parentHeight = this.parentData?.layoutInfo?.height ?: constraints.maxHeight
              val placeable = measurable.measure(constraints.copy(maxWidth = (parentWidth * 0.3f).toInt(), maxHeight = (parentHeight * 0.3f).toInt()))
            
            layout(placeable.width, placeable.height) {
                 val x = (parentWidth * 0.7f).toInt()
                 val y = (parentHeight * 0.7f).toInt()
                placeable.placeRelative(IntOffset(x, y))
            }
        }){
            Image(
                bitmap = ImageBitmap.imageResource(overlay2Res),
                contentDescription = "Overlay 2"
            )
        }

    }
}
```
Here we have nested boxes, each containing an `Image` composable.  Each `Box` calculates its position and size relative to its immediate parent using the same logic as the previous example. This approach can be scaled up to as many layered images as you need by nesting `Box` composables and adjusting sizing and positioning calculations.

For further reading, I strongly recommend delving into the “Jetpack Compose internals” documentation on the Android developer site. Specifically, the layout section is a must-read. Additionally, the "Compose Internals" series of blog posts can be valuable. Another resource I've found beneficial is the "Android UI Development with Jetpack Compose" book by Jorge Castillo. It dives deep into compose concepts and provides practical advice that you'd find useful when working with more complex layouts. These resources will help you solidify your understanding and give you the tools to approach future challenges more efficiently.

In summary, while `Box` and basic modifiers like `size` and `offset` are foundational, using `Modifier.layout` to tap into layout details of parent views provides the power to maintain relative size and position of layered images. Understanding this distinction unlocks true flexibility in composable layouts.

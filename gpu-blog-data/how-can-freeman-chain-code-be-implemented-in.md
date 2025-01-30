---
title: "How can Freeman chain code be implemented in Java?"
date: "2025-01-30"
id: "how-can-freeman-chain-code-be-implemented-in"
---
The efficient representation of digital curves using a compact encoding is crucial for various image processing applications, such as contour analysis and shape recognition. One method employed for this purpose is the Freeman chain code. My experience developing a custom optical character recognition (OCR) engine led me to implement this encoding technique for analyzing character outlines, where it proved instrumental in reducing storage requirements and simplifying feature extraction.

The core concept of Freeman chain code, sometimes also referred to as a directional code, involves representing a curve as a sequence of directional movements from one pixel to the next along the boundary. It operates by defining a set of direction codes, often associated with cardinal and intercardinal directions. Specifically, an 8-connected chain code uses eight directions, typically coded from 0 to 7, representing moves to the east (0), northeast (1), north (2), northwest (3), west (4), southwest (5), south (6), and southeast (7). The starting point of the chain is generally stored separately, allowing reconstruction of the original curve later. Crucially, the representation is inherently translation-invariant.

The implementation process can be broken down into several key stages: first, identifying the boundary pixels of the object within the digital image; second, selecting a starting pixel on this boundary; third, traversing the boundary, recording the directional moves using the pre-defined code, and finally, storing the start pixel location and the resulting chain code sequence.

Let me illustrate this with three code examples, explaining each step and providing relevant context. I'll use a simple 2D boolean array as a representation of the image, where `true` indicates a pixel belonging to the object of interest and `false` represents the background. These examples assume you have a populated boolean array already defined.

**Example 1: Boundary Detection & Start Pixel Identification**

This first example demonstrates how to identify a start pixel and initialize the boundary traversal process. The `findBoundaryStart` method iterates through the pixel array, searching for an object pixel that has at least one adjacent background pixel. This satisfies the requirement of a boundary pixel.

```java
import java.util.ArrayList;
import java.util.List;

public class ChainCode {

    public static class Point {
        int x;
        int y;

        public Point(int x, int y) {
            this.x = x;
            this.y = y;
        }
    }

    public static Point findBoundaryStart(boolean[][] image) {
        int rows = image.length;
        int cols = image[0].length;

        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                if (image[row][col]) { // Object pixel found
                    if (isBoundaryPixel(image, row, col)) {
                        return new Point(col, row); // X, Y format
                    }
                }
            }
        }
        return null; // No boundary found
    }


    private static boolean isBoundaryPixel(boolean[][] image, int row, int col){
        int rows = image.length;
        int cols = image[0].length;
        for (int i = row - 1; i <= row + 1; i++) {
            for (int j = col - 1; j <= col + 1; j++) {
               if(i >= 0 && i < rows && j >= 0 && j < cols) {
                   if((i!=row || j!= col) && !image[i][j])
                       return true;
               }
            }
        }
        return false;
    }

    public static void main(String[] args) {
         boolean[][] image = {
            {false, false, false, false, false},
            {false, true, true, true, false},
            {false, true, false, true, false},
            {false, true, true, true, false},
            {false, false, false, false, false}
        };

        Point start = findBoundaryStart(image);
        if (start != null) {
            System.out.println("Start pixel: (" + start.x + ", " + start.y + ")");
        } else {
            System.out.println("No start pixel found.");
        }
    }
}

```

This example employs a nested for-loop to examine every pixel and calls a helper `isBoundaryPixel` function. This function verifies if a particular pixel is a boundary pixel by checking if any of its eight neighbors are background pixels. Note the coordinates returned by the start point are in (x,y) format. The `main` method demonstrates usage of this method on a simple test `image` and prints the identified starting pixel coordinates to console.

**Example 2: Generating the Chain Code**

Having found the starting point, we need to generate the chain code. The `generateChainCode` method traverses the boundary, using the current pixel location to identify the next boundary pixel, thereby defining the direction of traversal.

```java
public static List<Integer> generateChainCode(boolean[][] image, Point start) {
        int rows = image.length;
        int cols = image[0].length;
        List<Integer> chainCode = new ArrayList<>();
        int[][] directions = {{1, 0}, {1, -1}, {0, -1}, {-1, -1}, {-1, 0}, {-1, 1}, {0, 1}, {1, 1}};
        int currentX = start.x;
        int currentY = start.y;
        int previousDirection = 0; // Initial direction doesn't matter much

        // Mark the start pixel to prevent re-visiting during boundary traversal.
        // Important for preventing an infinite loop if the shape has a hole.
        boolean[][] visited = new boolean[rows][cols];
        visited[currentY][currentX] = true;

        do {
            boolean foundNext = false;
            for(int i = 0; i< 8; i++){
                int directionIndex = (previousDirection + i + 1) % 8;
                int nextX = currentX + directions[directionIndex][0];
                int nextY = currentY + directions[directionIndex][1];

                if (nextX >= 0 && nextX < cols && nextY >= 0 && nextY < rows && image[nextY][nextX] && !visited[nextY][nextX]) {
                    chainCode.add(directionIndex);
                    visited[nextY][nextX] = true;
                    previousDirection = directionIndex;
                    currentX = nextX;
                    currentY = nextY;
                    foundNext = true;
                    break;

                }
            }
            if(!foundNext)
               break;

        } while (currentX != start.x || currentY != start.y);


        return chainCode;
    }

    public static void main(String[] args) {
        boolean[][] image = {
            {false, false, false, false, false},
            {false, true, true, true, false},
            {false, true, false, true, false},
            {false, true, true, true, false},
            {false, false, false, false, false}
        };

        Point start = findBoundaryStart(image);
         if (start != null) {
            List<Integer> chainCode = generateChainCode(image, start);
             System.out.print("Chain code: ");
              for (int code : chainCode) {
              System.out.print(code + " ");
            }
          System.out.println();

        } else {
          System.out.println("No start pixel found.");
        }
    }
```

Here we store the 8 directions using `directions` array. The algorithm maintains the previous direction, which is initially irrelevant, and uses this to order the next direction search to prevent a back and forth traversal that could occur if a simple clockwise search was used. The next pixel is found by iterating through eight potential neighbor pixels, using the `directions` array to calculate relative offsets. Each time a pixel is found, the corresponding direction index is added to the `chainCode`, the next location becomes the current location, and a visited `boolean` array is used to prevent infinite loops. The process continues until the traversal returns to the initial starting point. Finally, the main method shows the result.

**Example 3: Combining Steps for a Full Implementation**

This final example integrates boundary detection and chain code generation. The `generateFullChainCode` method encapsulates both operations.

```java

public class ChainCode {
    public static class Point {
        int x;
        int y;

        public Point(int x, int y) {
            this.x = x;
            this.y = y;
        }
    }

    public static Point findBoundaryStart(boolean[][] image) {
        int rows = image.length;
        int cols = image[0].length;

        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                if (image[row][col]) { // Object pixel found
                    if (isBoundaryPixel(image, row, col)) {
                        return new Point(col, row); // X, Y format
                    }
                }
            }
        }
        return null; // No boundary found
    }


    private static boolean isBoundaryPixel(boolean[][] image, int row, int col){
        int rows = image.length;
        int cols = image[0].length;
        for (int i = row - 1; i <= row + 1; i++) {
            for (int j = col - 1; j <= col + 1; j++) {
               if(i >= 0 && i < rows && j >= 0 && j < cols) {
                   if((i!=row || j!= col) && !image[i][j])
                       return true;
               }
            }
        }
        return false;
    }


    public static List<Integer> generateChainCode(boolean[][] image, Point start) {
        int rows = image.length;
        int cols = image[0].length;
        List<Integer> chainCode = new ArrayList<>();
        int[][] directions = {{1, 0}, {1, -1}, {0, -1}, {-1, -1}, {-1, 0}, {-1, 1}, {0, 1}, {1, 1}};
        int currentX = start.x;
        int currentY = start.y;
        int previousDirection = 0;

        boolean[][] visited = new boolean[rows][cols];
        visited[currentY][currentX] = true;

        do {
            boolean foundNext = false;
            for(int i = 0; i< 8; i++){
                int directionIndex = (previousDirection + i + 1) % 8;
                int nextX = currentX + directions[directionIndex][0];
                int nextY = currentY + directions[directionIndex][1];

                if (nextX >= 0 && nextX < cols && nextY >= 0 && nextY < rows && image[nextY][nextX] && !visited[nextY][nextX]) {
                    chainCode.add(directionIndex);
                    visited[nextY][nextX] = true;
                    previousDirection = directionIndex;
                    currentX = nextX;
                    currentY = nextY;
                    foundNext = true;
                    break;

                }
            }
            if(!foundNext)
               break;

        } while (currentX != start.x || currentY != start.y);
        return chainCode;
    }


     public static List<Integer> generateFullChainCode(boolean[][] image) {
        Point start = findBoundaryStart(image);
        if(start== null) return null;
        return generateChainCode(image,start);
    }



    public static void main(String[] args) {
         boolean[][] image = {
            {false, false, false, false, false},
            {false, true, true, true, false},
            {false, true, false, true, false},
            {false, true, true, true, false},
            {false, false, false, false, false}
        };

        List<Integer> fullChainCode = generateFullChainCode(image);
          if(fullChainCode != null) {
           System.out.print("Full chain code: ");
              for (int code : fullChainCode) {
                System.out.print(code + " ");
              }
            System.out.println();
        } else {
          System.out.println("No chain code generated.");
        }
    }
}
```

This revised example consolidates the logic into a single, more convenient method. The main method demonstrates the output on the test image.

For further exploration, I recommend studying introductory textbooks on digital image processing, which typically dedicate sections to boundary representation and chain codes. Specific areas to focus on include algorithms for boundary tracking and curve simplification, as well as the potential for using chain codes as features in pattern recognition applications. Publications from computer vision conferences often contain detailed analysis of advanced techniques based on the core principles. Furthermore, experimenting with different types of images, including images with multiple closed contours, is a practical way to test and refine the implementation.

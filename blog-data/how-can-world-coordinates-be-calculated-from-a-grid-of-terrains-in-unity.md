---
title: "How can world coordinates be calculated from a grid of terrains in Unity?"
date: "2024-12-23"
id: "how-can-world-coordinates-be-calculated-from-a-grid-of-terrains-in-unity"
---

Let's tackle this, shall we? I've certainly spent my fair share of time elbow-deep in Unity's coordinate systems. Calculating world coordinates from a grid of terrains, while seemingly straightforward, can quickly spiral into a debugging exercise if not approached systematically. I recall a project years back, involving procedural terrain generation for a strategy game; we initially stumbled with some quirky placement issues, all stemming from misunderstandings of how the grid, terrain, and world coordinate spaces interplay. Here's how I’d approach this, drawing on that experience, and offering some solid code examples.

The primary challenge lies in the fact that a grid of terrains doesn't inherently possess a one-to-one correspondence with the world coordinate system. Your grid might represent individual terrain tiles, each of which lives within its *local* coordinate space relative to its terrain object. The world coordinates, conversely, are the absolute positions within the game environment. To transition effectively between these spaces, you need to apply a series of transformations.

Essentially, we're reverse-engineering the process that Unity implicitly handles when you create terrain objects and place them in the world. You need to account for: the grid position, the terrain's size, the terrain's origin point within the world, and any rotation that might be applied to the terrain object itself.

Let's break it down into steps. First, you'll need to know your grid's dimensions and the size of each terrain tile. Imagine a 2D array representing your terrain grid: `terrainGrid[x, z]`. Let's say each cell in this grid corresponds to a single terrain object. Each terrain object, itself, has local dimensions (width and length) typically matching the terrain data's resolution. Let's label these dimensions as `terrainWidth` and `terrainLength`.

Next, the position of the terrain within the world needs to be taken into account. This involves obtaining the world space position of, for instance, the lower-left corner of the first terrain in your grid. We'll refer to this starting point as `gridOriginWorldPosition`. This `Vector3` will serve as our base point when mapping grid indices back to world coordinates.

Finally, the rotation of the terrain also impacts mapping. A terrain not rotated on the y-axis, simplifies this process. However, if any rotation is applied, you’d have to perform coordinate transformation using the terrain's transformation matrix. If there is no rotation, it simplifies to adding offsets calculated from grid indices and terrain dimensions.

Let's solidify this with code. Here’s how you would calculate the world position of a specific grid point without rotation, assuming all terrains are of uniform size and with no rotation applied:

```csharp
using UnityEngine;

public class TerrainWorldCalculator : MonoBehaviour
{
    public Vector3 gridOriginWorldPosition; // Origin point of our grid in world space
    public int gridWidth = 5; // Number of terrains in X direction
    public int gridLength = 5; // Number of terrains in Z direction
    public float terrainWidth = 100f; // Width of each individual terrain
    public float terrainLength = 100f; // Length of each individual terrain

    public Vector3 GetWorldPositionFromGrid(int gridX, int gridZ)
    {
        if (gridX < 0 || gridX >= gridWidth || gridZ < 0 || gridZ >= gridLength)
        {
            Debug.LogError("Grid coordinates out of bounds.");
            return Vector3.zero; // Return zero vector or some other fallback strategy
        }

        float worldX = gridOriginWorldPosition.x + (gridX * terrainWidth);
        float worldZ = gridOriginWorldPosition.z + (gridZ * terrainLength);
        // Note: Y coordinate in this case is based on terrain height.
        // You'd usually use `terrain.SampleHeight(localX, localZ)` to get terrain height.
        // But this example only addresses flat terrain.

        return new Vector3(worldX, gridOriginWorldPosition.y, worldZ);
    }

    // Example usage (this function needs to be called from another script or a button press)
    public void TestConversion()
    {
        //Get world position for grid position (2, 3)
        Vector3 worldPos = GetWorldPositionFromGrid(2, 3);
        Debug.Log("World position for grid (2, 3): " + worldPos);
    }
}
```
This is a simple case. The code takes the `gridOriginWorldPosition` and offsets it according to the provided grid indices and terrain dimensions. It assumes your terrain is perfectly flat. For more realistic scenarios, we need to handle elevation within the terrain itself.

Here's an adaptation to handle terrain height. This assumes you have a reference to your terrain object:

```csharp
using UnityEngine;

public class TerrainWorldCalculatorWithHeight : MonoBehaviour
{
  public Vector3 gridOriginWorldPosition;
  public int gridWidth = 5;
  public int gridLength = 5;
  public float terrainWidth = 100f;
  public float terrainLength = 100f;
  public Terrain[,] terrainGrid; //Assuming you store your Terrain objects in a grid

  public Vector3 GetWorldPositionFromGrid(int gridX, int gridZ)
  {
    if (gridX < 0 || gridX >= gridWidth || gridZ < 0 || gridZ >= gridLength)
    {
       Debug.LogError("Grid coordinates out of bounds.");
       return Vector3.zero;
    }

    Terrain currentTerrain = terrainGrid[gridX, gridZ];

    if (currentTerrain == null)
    {
        Debug.LogError("Terrain object at grid position (" + gridX + ", " + gridZ + ") is null.");
        return Vector3.zero; // or handle this differently based on requirement
    }

    float worldX = gridOriginWorldPosition.x + (gridX * terrainWidth);
    float worldZ = gridOriginWorldPosition.z + (gridZ * terrainLength);

    // Local position inside the current terrain
    float localX = worldX - currentTerrain.transform.position.x;
    float localZ = worldZ - currentTerrain.transform.position.z;

    // Sample the height of current terrain
    float worldY = currentTerrain.SampleHeight(new Vector3(localX,0,localZ));

    //This version adds the y position of the terrain as well.
    worldY+= currentTerrain.transform.position.y;

    return new Vector3(worldX, worldY, worldZ);
  }

  //Example usage (this function needs to be called from another script or a button press)
  public void TestConversion()
  {
       //Get world position for grid position (2, 3)
      Vector3 worldPos = GetWorldPositionFromGrid(2, 3);
      Debug.Log("World position for grid (2, 3): " + worldPos);
  }
}
```

In this enhanced example, we also fetch the `y` component from the underlying terrain, thereby accounting for the elevation differences. Note, that the `localX` and `localZ` variables are calculated in respect to the current terrain object's position.

Let's tackle a final, more comprehensive, example that incorporates rotation of individual terrains. Here, we'll need to use the transformation matrix of each terrain. This example assumes all terrains use the same height map scale and same size:

```csharp
using UnityEngine;

public class TerrainWorldCalculatorRotated : MonoBehaviour
{
    public Vector3 gridOriginWorldPosition;
    public int gridWidth = 5;
    public int gridLength = 5;
    public float terrainWidth = 100f;
    public float terrainLength = 100f;
    public Terrain[,] terrainGrid;

    public Vector3 GetWorldPositionFromGrid(int gridX, int gridZ)
    {
      if (gridX < 0 || gridX >= gridWidth || gridZ < 0 || gridZ >= gridLength)
      {
        Debug.LogError("Grid coordinates out of bounds.");
        return Vector3.zero;
      }

        Terrain currentTerrain = terrainGrid[gridX, gridZ];

      if (currentTerrain == null)
      {
        Debug.LogError("Terrain object at grid position (" + gridX + ", " + gridZ + ") is null.");
        return Vector3.zero; // or handle differently
      }

        // Calculate grid position
        Vector3 gridPosition = new Vector3(gridX * terrainWidth, 0, gridZ * terrainLength);

        // Convert grid position to world space using terrain transform
        Vector3 worldPosition = currentTerrain.transform.TransformPoint(gridPosition);

        //calculate local position in terrain space for height lookup
        float localX = worldPosition.x - currentTerrain.transform.position.x;
        float localZ = worldPosition.z - currentTerrain.transform.position.z;
        
        // Add terrain height at calculated local position
        worldPosition.y = currentTerrain.SampleHeight(new Vector3(localX, 0, localZ)) + currentTerrain.transform.position.y;


        return worldPosition;
    }

      //Example usage (this function needs to be called from another script or a button press)
  public void TestConversion()
  {
        //Get world position for grid position (2, 3)
      Vector3 worldPos = GetWorldPositionFromGrid(2, 3);
      Debug.Log("World position for grid (2, 3): " + worldPos);
    }
}
```

In this final iteration, the `TransformPoint` method of the individual terrain object is used, allowing us to transform our grid position to world space, properly accounting for rotation. The height sampling is still done to get the correct `y` value for the final result.

For further study, I'd recommend these resources: For understanding transformations in 3D, "3D Math Primer for Graphics and Game Development" by Fletcher Dunn and Ian Parberry is exceptionally helpful. Unity’s official documentation on `UnityEngine.Transform` and `UnityEngine.Terrain` also offers detailed information on available methods and the underlying mechanics. Furthermore, a deep dive into linear algebra as it pertains to game development (any intro to linear algebra book will do the job), will solidify your understanding of coordinate transformations. Finally, don't neglect the Unity Scripting API documentation for specifics on how each of these classes functions. With a solid grasp of these concepts, you will be able to handle more complex terrain arrangements, and avoid those debugging sessions where the world itself seems to have a different idea of where things are placed.

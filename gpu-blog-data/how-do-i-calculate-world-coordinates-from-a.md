---
title: "How do I calculate world coordinates from a terrain grid in Unity?"
date: "2025-01-30"
id: "how-do-i-calculate-world-coordinates-from-a"
---
The fundamental challenge in translating from a 2D terrain grid to 3D world coordinates within Unity lies in understanding the relationship between discrete grid indices and continuous, positional values in the game world. Specifically, we're concerned with mapping a point described by its row and column (i, j) on a grid to its corresponding x, y, and z coordinates in Unity's 3D space. This process requires knowledge of the terrain's origin, scale, and potentially its heightmap data if elevation is a concern. Having spent significant time optimizing terrain generation and interaction systems, I've found that clarity regarding these factors is paramount for accurate transformations.

The first step is to define the reference point for the terrain. Most commonly, the grid's origin (index 0,0) is aligned with a specific world position, often at the terrain's lower-left corner. We'll refer to this as the `terrainOrigin`. The scale of the grid directly translates to the distance between two adjacent grid points in world space along the x and z axes. I denote these scale factors as `gridScaleX` and `gridScaleZ`. Note that typically these values are equal and represent the size of each grid cell in world units. Furthermore, if the terrain has height variations (and it usually does), we will require a way to determine the Y coordinate which, in our context, represents the terrain’s vertical elevation. This frequently involves querying the terrain's heightmap or equivalent.

To elaborate, if we neglect terrain height for a moment, the x and z world coordinates, denoted `worldX` and `worldZ`, can be derived by:

```
worldX = terrainOrigin.x + (j * gridScaleX);
worldZ = terrainOrigin.z + (i * gridScaleZ);
```

Here, `j` is the column index, and `i` is the row index. Note that we are adding the scaled offset from the origin, not just the raw indexes. This calculation assumes a grid that starts at (0,0), but this can be easily adjusted if needed by adding or subtracting a constant offset to each index before the calculations. Now, obtaining the `worldY` component, the vertical height, requires sampling the heightmap, which is specific to Unity’s terrain system.

In practice, I've found that encapsulation is essential, typically implemented as a method within a TerrainManager class or a similar structure:

```csharp
using UnityEngine;

public class TerrainManager : MonoBehaviour
{
    public Terrain terrain; // Assigned in the Inspector
    public float gridScale = 1f;
    private Vector3 terrainOrigin;

    void Start()
    {
        if(terrain != null)
        {
            terrainOrigin = terrain.transform.position;
        } else
        {
            Debug.LogError("Terrain is not assigned in the inspector");
        }

    }

    public Vector3 GridToWorld(int i, int j)
    {
      if(terrain == null)
          return Vector3.zero; //Return zero if the terrain was not assigned in inspector.

        float worldX = terrainOrigin.x + (j * gridScale);
        float worldZ = terrainOrigin.z + (i * gridScale);
        float worldY = terrain.SampleHeight(new Vector3(worldX, 0, worldZ)) + terrainOrigin.y;

        return new Vector3(worldX, worldY, worldZ);
    }
}
```

In this C# example, the `GridToWorld` method takes the row (`i`) and column (`j`) as inputs. The `gridScale` variable, assignable via the Inspector, controls the cell spacing of the grid. The `terrain` reference must be assigned within the Unity Editor to the relevant terrain object. Inside the function, we calculate the X and Z world coordinates as described before. To determine the Y-coordinate we use Unity's `Terrain.SampleHeight` method, passing in the calculated (X,0,Z) position, then adding the Terrain's position offset to it. Note the added null check on the assigned terrain variable to gracefully return a Vector3.zero if no terrain is assigned, preventing potential errors.

When the grid size and world space cell size do not match, I’ve encountered scenarios requiring a more complex calculation. Consider the case where each grid cell represents a larger area than one world unit:

```csharp
using UnityEngine;

public class TerrainManager : MonoBehaviour
{
    public Terrain terrain;
    public int gridWidth = 100;  // Number of cells along X
    public int gridHeight = 100; // Number of cells along Z
    public float worldWidth = 500f; // Width of the terrain in world units
    public float worldHeight = 500f; // Height of the terrain in world units
    private Vector3 terrainOrigin;
    private float gridScaleX;
    private float gridScaleZ;

    void Start()
    {
        if(terrain != null)
        {
            terrainOrigin = terrain.transform.position;
            gridScaleX = worldWidth / gridWidth;
            gridScaleZ = worldHeight / gridHeight;
        } else
        {
            Debug.LogError("Terrain is not assigned in the inspector");
        }

    }

    public Vector3 GridToWorld(int i, int j)
    {
      if(terrain == null)
          return Vector3.zero; //Return zero if the terrain was not assigned in inspector.

        float worldX = terrainOrigin.x + (j * gridScaleX);
        float worldZ = terrainOrigin.z + (i * gridScaleZ);
        float worldY = terrain.SampleHeight(new Vector3(worldX, 0, worldZ)) + terrainOrigin.y;
        return new Vector3(worldX, worldY, worldZ);
    }
}

```

Here, `gridWidth` and `gridHeight` specify the number of grid cells across the terrain. `worldWidth` and `worldHeight` denote the total size of the terrain in world space. We calculate `gridScaleX` and `gridScaleZ` by dividing the total world size by the number of cells in each dimension. This method permits the usage of an arbitrary grid resolution that does not match the resolution used to render the terrain. Note the division in the start function and the utilization of those computed values in the transform function. I frequently utilize this approach when generating procedural textures mapped onto a terrain.

Furthermore, when dealing with grids where data is stored in one dimensional arrays instead of a typical 2D array, one must first calculate the corresponding 2D index values before performing the 3D world coordinate calculation. For example:

```csharp
using UnityEngine;

public class TerrainManager : MonoBehaviour
{
    public Terrain terrain;
    public int gridWidth = 100;  // Number of cells along X
    public float worldWidth = 500f; // Width of the terrain in world units
    public float worldHeight = 500f; // Height of the terrain in world units

    private Vector3 terrainOrigin;
    private float gridScale;

    void Start()
    {
      if(terrain != null)
      {
            terrainOrigin = terrain.transform.position;
            gridScale = worldWidth / gridWidth;
        } else
        {
           Debug.LogError("Terrain is not assigned in the inspector");
        }
    }

    public Vector3 IndexToWorld(int index)
    {
        if(terrain == null)
            return Vector3.zero; //Return zero if the terrain was not assigned in inspector.

        int i = index / gridWidth;
        int j = index % gridWidth;

        float worldX = terrainOrigin.x + (j * gridScale);
        float worldZ = terrainOrigin.z + (i * gridScale);
        float worldY = terrain.SampleHeight(new Vector3(worldX, 0, worldZ)) + terrainOrigin.y;

        return new Vector3(worldX, worldY, worldZ);
    }
}
```

In this version, a single `index` refers to the location in a flattened grid. To convert this 1D index to 2D grid indices `i` and `j`, we use integer division and the modulo operator respectively. This approach is commonly employed with data structures where memory access patterns favor contiguous memory allocation. The resulting `i` and `j` values are then used in the same manner described previously to calculate the X and Z coordinates and then the associated height from the Unity terrain component.

For continued study, exploring resources on 3D coordinate systems, linear transformations, and Unity's terrain API is recommended. Textbooks covering computer graphics and game development algorithms can provide a deeper theoretical understanding of these operations. Additionally, the official Unity scripting documentation offers comprehensive explanations of methods like `Terrain.SampleHeight` and related functions. Examining open-source terrain generation projects on sites like GitHub can provide additional context and practical examples of these types of calculations. Studying these resources will help enhance understanding when calculating 3D world coordinates from terrain grids in Unity and similar environments.

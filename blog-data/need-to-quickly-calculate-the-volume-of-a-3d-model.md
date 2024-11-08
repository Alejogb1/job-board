---
title: "Need to Quickly Calculate the Volume of a 3D Model?"
date: '2024-11-08'
id: 'need-to-quickly-calculate-the-volume-of-a-3d-model'
---

```csharp
public static double CalculateVolume(GeometryModel3D model)
{
    var mesh = model.Geometry as MeshGeometry3D;
    if (mesh == null)
    {
        throw new ArgumentException("GeometryModel3D must have a MeshGeometry3D.");
    }

    double volume = 0;
    for (int i = 0; i < mesh.TriangleIndices.Count; i += 3)
    {
        var v1 = mesh.Positions[mesh.TriangleIndices[i]];
        var v2 = mesh.Positions[mesh.TriangleIndices[i + 1]];
        var v3 = mesh.Positions[mesh.TriangleIndices[i + 2]];
        volume += (((v2.Y - v1.Y) * (v3.Z - v1.Z) - (v2.Z - v1.Z) * (v3.Y - v1.Y)) * (v1.X + v2.X + v3.X)) / 6;
    }

    return volume;
}
```

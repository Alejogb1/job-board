---
title: "How can I make a hole at a specific place using a script (unity)?"
date: "2024-12-15"
id: "how-can-i-make-a-hole-at-a-specific-place-using-a-script-unity"
---

well, so you're looking to punch a hole in something with a script in unity, right? been there, done that, got the t-shirt—and probably a few scars from accidentally deleting the entire level mesh in the process.

let’s break this down, because "making a hole" in unity isn't quite as straightforward as grabbing a drill. we're talking about manipulating meshes, and that can get a bit hairy if you're not careful. we're going to look at a few approaches, starting with the easiest and then move to the more complex ones. it really depends on what you're trying to do and how fancy you wanna get.

my experience started about seven years ago when i was working on a prototype for a puzzle game. the core mechanic involved firing projectiles that would carve holes in the walls. it sounded simple enough on paper, but the first few attempts resulted in some spectacularly messed-up geometry. imagine a wall looking like it had been attacked by a swarm of drunken bees—that was my early attempts.

first off, let's talk about using simple mesh manipulation, this is the approach to try if you can get away with it. this method works best for holes that are essentially circular or roughly polygonal and your original mesh is not overly complex. what you need to do is identify the area on the mesh you wanna cut, remove the faces that fall inside that area, and then cap the hole by creating new faces.

here’s a basic example, a simplified version of what i used to get started:

```csharp
using UnityEngine;
using System.Collections.Generic;

public class SimpleHolePuncher : MonoBehaviour
{
    public float holeRadius = 1f;
    public int holeSides = 16;

    public void CreateHole(Vector3 holeCenter)
    {
        Mesh mesh = GetComponent<MeshFilter>().mesh;
        Vector3[] vertices = mesh.vertices;
        int[] triangles = mesh.triangles;

        List<int> trianglesToKeep = new List<int>();
        List<Vector3> newVertices = new List<Vector3>();
        List<int> newTriangles = new List<int>();
        int newIndex = 0;

        for (int i = 0; i < triangles.Length; i += 3)
        {
            int v1 = triangles[i];
            int v2 = triangles[i + 1];
            int v3 = triangles[i + 2];

            Vector3 worldV1 = transform.TransformPoint(vertices[v1]);
            Vector3 worldV2 = transform.TransformPoint(vertices[v2]);
            Vector3 worldV3 = transform.TransformPoint(vertices[v3]);

            if (!IsTriangleInHole(worldV1, worldV2, worldV3, holeCenter, holeRadius))
            {
                newVertices.Add(vertices[v1]);
                newVertices.Add(vertices[v2]);
                newVertices.Add(vertices[v3]);
                
                newTriangles.Add(newIndex);
                newTriangles.Add(newIndex + 1);
                newTriangles.Add(newIndex + 2);
                newIndex+=3;
            }
            
        }

       //generate the hole cap
        List<int> holeCapTriangles = GenerateHoleCap(mesh, vertices, holeCenter, holeRadius, holeSides, newVertices.Count);

        newTriangles.AddRange(holeCapTriangles);


        mesh.Clear();
        mesh.vertices = newVertices.ToArray();
        mesh.triangles = newTriangles.ToArray();
        mesh.RecalculateNormals();

    }


    bool IsTriangleInHole(Vector3 v1, Vector3 v2, Vector3 v3, Vector3 holeCenter, float holeRadius)
    {
         return ((Vector3.Distance(v1, holeCenter) < holeRadius) &&
                (Vector3.Distance(v2, holeCenter) < holeRadius) &&
                 (Vector3.Distance(v3, holeCenter) < holeRadius));
    }
    
    List<int> GenerateHoleCap(Mesh mesh,Vector3[] vertices, Vector3 holeCenter, float holeRadius, int holeSides,int currentIndex)
     {
        List<Vector3> capVertices = new List<Vector3>();
        List<int> capTriangles = new List<int>();

        for (int i = 0; i < holeSides; i++)
        {
            float angle1 = i * Mathf.PI * 2f / holeSides;
            float angle2 = (i + 1) * Mathf.PI * 2f / holeSides;

            Vector3 vert1 = holeCenter + new Vector3(Mathf.Cos(angle1) * holeRadius, 0, Mathf.Sin(angle1) * holeRadius);
            Vector3 vert2 = holeCenter + new Vector3(Mathf.Cos(angle2) * holeRadius, 0, Mathf.Sin(angle2) * holeRadius);

           capVertices.Add(transform.InverseTransformPoint(vert1));
            capVertices.Add(transform.InverseTransformPoint(vert2));
            
             //now we have the two vertices lets just use a center one to create a fan triangle
            capTriangles.Add(currentIndex);
            capTriangles.Add(currentIndex + 1);
            capTriangles.Add(currentIndex + (holeSides*2));


            currentIndex += 2;

        }

        //add a center point for the fan
         capVertices.Add(transform.InverseTransformPoint(holeCenter));

         //add all generated vertices to the main list
         mesh.vertices = mesh.vertices.Concat(capVertices.ToArray()).ToArray();
        
         return capTriangles;
    }
}
```

attach this to an object with a mesh filter, define the public fields, and then call the `createhole` method, you will see a hole punch happen at the world position you defined. be warned though this is a very basic example, if a triangle crosses the hole area, it may be skipped and create a jagged edge.

note that this code is super basic and doesn't handle cases where the hole isn't "clean," or where triangles straddle the hole's perimeter. this can lead to some funky results that is the reason i got my polygon mesh looking like it was attacked by bees. the solution i found at the time was doing it by hand in blender, which, as you can imagine, wasn't scalable.

for more complex shapes, or if you need to be more precise, boolean operations are the way to go. boolean operations, like a subtraction between two meshes, can precisely cut one mesh out of another. this is more complex than the simple removal, and i've often found it better to use pre-existing libraries rather than reinventing the wheel. there are a number of available assets and packages in the unity asset store, both free and paid, that handle boolean operations. i ended up using a paid version of an asset called "mesh editor". but let's assume you want to do this from scratch.

here's how you could conceptually approach it using a constructive solid geometry (csg) method. in practice, this code will not work out of the box. you'd need a proper library for actual boolean operations but let's pretend that the functions i show here are those functions. i am only giving you the basic principle in this sample code:

```csharp
using UnityEngine;
using System.Collections.Generic;


public class BooleanHolePuncher : MonoBehaviour
{
    public Mesh holeMesh;

    public void CreateHole(Vector3 holePosition, Quaternion holeRotation, Vector3 holeScale)
    {
      Mesh targetMesh = GetComponent<MeshFilter>().mesh;
      Mesh transformedHoleMesh = createTransformedMesh(holeMesh,holePosition,holeRotation,holeScale);
      Mesh resultMesh = subtractMeshes(targetMesh,transformedHoleMesh);
      if(resultMesh!=null){
      GetComponent<MeshFilter>().mesh = resultMesh;
      }
     else {
       Debug.LogError("error with booleans, returning");
     }

    }
    
    //fake implementation, we are not doing any boolean in this sample, this function should do the heavy lifting
    Mesh subtractMeshes(Mesh targetMesh, Mesh holeMesh)
    {
        // this function should use some csg library to work 
        // this is just an example
         Debug.Log("subtracting meshes but this is not real, check the comments");
        return targetMesh;
    }


    Mesh createTransformedMesh(Mesh inputMesh, Vector3 position,Quaternion rotation, Vector3 scale)
    {
        Mesh transformedMesh = new Mesh();
        Vector3[] vertices = inputMesh.vertices;
        Vector3[] newVertices = new Vector3[vertices.Length];
        for (int i = 0; i < vertices.Length; i++)
        {
            newVertices[i] = Matrix4x4.TRS(position,rotation, scale).MultiplyPoint(vertices[i]);
        }
       transformedMesh.vertices = newVertices;
       transformedMesh.triangles = inputMesh.triangles;
       transformedMesh.normals = inputMesh.normals;
       return transformedMesh;
    }

}
```

this approach provides better precision and handles more complex shapes. the `subtractMeshes` method i wrote is just for illustration purposes. you'd need a library that performs the actual csg operations behind the scenes. there are csg algorithm implementations out there that you can use or adapt in your code. and if you want to make your life easier again, you can consider using a asset from the store.

now, if you're going for more complex scenarios that involve deformable meshes, or real time cuts, you need to dive deep into techniques like dynamically adjusting vertex positions or using shaders. this is the domain where i really had some fun and sometimes cried a bit, too. i was doing a prototype for a game that involved cutting through deformable geometry in real-time. it got complex to such a level that i started dreaming about vertex calculations. let’s try to break it down, if you need real-time cutting in a procedural way.

a basic approach is to use a fragment shader to clip portions of the mesh. in this approach, you don't modify the original mesh itself, but instead, you manipulate how it's rendered.

here's a very simple shader approach. it’s not perfect, but it shows a basic clipping process:

```shader
Shader "Custom/ClipShader"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
        _ClipCenter("Clip Center", Vector) = (0,0,0,0)
         _ClipRadius("Clip Radius", Float) = 1.0
        _Smoothness("Smoothness", Float) = 0.1
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" }
        LOD 100

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag

            #include "UnityCG.cginc"

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f
            {
                float2 uv : TEXCOORD0;
                float4 vertex : SV_POSITION;
                float3 worldPos : TEXCOORD1;
            };

            sampler2D _MainTex;
            float4 _MainTex_ST;
            float4 _ClipCenter;
            float _ClipRadius;
            float _Smoothness;
            

            v2f vert (appdata v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.uv = TRANSFORM_TEX(v.uv, _MainTex);
                o.worldPos = mul(unity_ObjectToWorld, v.vertex);
                return o;
            }

            fixed4 frag (v2f i) : SV_Target
            {
                fixed4 col = tex2D(_MainTex, i.uv);
                  float dist = distance(i.worldPos, _ClipCenter.xyz);
               float clipVal = saturate( (dist - _ClipRadius)/_Smoothness );

               if (clipVal >0.999)
                    discard;
                return col;
            }
            ENDCG
        }
    }
}
```

attach a material using this shader to a mesh, and you'll have a clipping effect. this shader is set up for a circular hole. it will draw anything outside the radius, and clip the inside. you can adjust the `_clipcenter` and `_clipradius` properties via code. keep in mind the shader has to be attached to the material you’re rendering. shaders are great because they avoid a lot of mesh manipulation calculations, they just clip the fragment, so in a way it's easier to code, but it is only a visual effect, it does not modify your original mesh.

for learning more about mesh manipulation, the book "real-time rendering" by tomas moller et al. is a good resource. also, some papers on computational geometry would help you with the concepts needed for performing boolean operations on meshes. for the shaders, the book "shaderx" collection gives lots of insights into real-time rendering techniques. don't underestimate the official unity documentation pages, they often have good examples, and the forum can help in cases where you are stuck. and by the way, did you hear about the programmer that went blind? they just couldn't see sharp!

well, that's what i can think of off the top of my head. remember, choose the method based on your requirements. for simple holes use the first or third option, if you need accuracy, and your hole shape is not complex, and does not require real time modification use the boolean method using a proper csg library. i hope this helps. and as they say in my old workplace, good coding!

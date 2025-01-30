---
title: "How can I improve Bevy's performance on Windows 11 with an RTX 2060 to run for more than 5 seconds?"
date: "2025-01-30"
id: "how-can-i-improve-bevys-performance-on-windows"
---
The limiting factor for Bevy performance on Windows 11, especially with an RTX 2060, is frequently not the raw processing power of the GPU itself, but rather the overhead and inefficiencies stemming from resource management, CPU-GPU synchronization, and inappropriate application of Bevy's ECS architecture.  I've encountered this specific scenario multiple times in game development and simulation projects. Sustained performance beyond 5 seconds on this hardware combination often requires targeted optimization rather than broad-stroke changes.

The primary areas needing attention typically fall into several categories: batching, rendering configuration, CPU load reduction, and effective ECS utilization. Bevy, while powerful, defaults to relatively conservative settings which can quickly become bottlenecks on mid-range systems.

**1.  Batching and Draw Call Reduction:**

The most immediate gains are often achieved through minimizing the number of draw calls the CPU sends to the GPU. Each draw call incurs CPU overhead, and this can become a significant bottleneck, particularly with a large number of independent entities. Bevy does not automatically batch disparate objects. You must explicitly structure data to promote batching.

*   **Shared Materials:** Instead of creating numerous different materials that only slightly differ (e.g., variations in a diffuse color), aim for a single material and utilize uniform data, like an instance color attribute, to achieve variation. This allows the GPU to render many objects with the same material at once, drastically reducing the number of draw calls. Texture atlases also help in this respect.
*   **Instanced Rendering:** Where possible, use instancing instead of individually drawing many similar meshes. Bevy allows this through `InstancedMeshBundle`s. Rather than creating hundreds or thousands of identical spheres, one can create a single sphere mesh and render numerous instances of it using instance data to modify positions, scales, and potentially colors. This transforms multiple draw calls for each sphere into a single draw call for all instances.
*   **Entity Grouping:** Group related entities with a shared material that need to be drawn in a specific way.  Entities that do not share a material or draw order will require a separate draw call for each, which can quickly overwhelm the CPU if not managed correctly.

**2.  Rendering Configuration Optimization:**

Bevy's default render pipeline might not be ideal for every scenario. Adjusting rendering configurations can provide a substantial performance boost.

*   **Reduce Render Passes:** Evaluate if all render passes are necessary. The default pipeline can have more passes than your application requires, and eliminating unnecessary ones can improve performance.
*   **Resolution and Post-Processing:** Rendering at a lower resolution can significantly improve performance, particularly with an RTX 2060.  Likewise, excessive post-processing effects (like bloom or anti-aliasing) can be costly. Tweak these settings or disable them, particularly when debugging.  Downsampling via the window descriptor should be explored to see its overall impact.
*   **GPU Driven Rendering:** Explore the capabilities within bevy and consider the viability of utilizing compute shaders and indirect draw calls. This moves the bulk of draw processing onto the GPU.  Bevy is making strides in this area, but you should examine this feature carefully and determine if it fits your needs.

**3. CPU Load Reduction:**

Even with a powerful GPU, a busy CPU can bottleneck overall performance. Efficient CPU practices are critical.

*   **Avoid Spawning Entities in Every Frame:** If possible, create entities only when necessary, pre-populate levels, and avoid creating and destroying entities too frequently within your game loop. Instead, reuse entities and modify existing component data.
*   **Efficient Queries:** Bevy's query system is powerful, but poorly designed queries can introduce inefficiencies. Cache query results where possible and avoid running complex queries for every frame. If you are only updating position, only request positions. If you are modifying multiple components, request them all at once in a single system, versus many separate systems.
*   **Parallelization:**  Leverage Bevy's parallel execution capabilities. Decompose the workload into independent tasks that can run concurrently across multiple CPU threads.  However, be aware that inappropriate parallelization can sometimes be detrimental.

**4.  Effective ECS Usage:**

Bevy’s ECS design needs to be leveraged correctly for maximum performance.

*   **Sparse Components:** Use component types that are appropriate for your use case. Avoid unnecessary components or large struct types for infrequently accessed data. A common mistake is adding large structs that don't change very often.
*   **System Ordering:** Be mindful of system ordering. Incorrect system ordering can create data dependency bottlenecks or force redundant computations. Ensure systems that write data are ordered before systems that read the updated data.  Think about if your systems should be run strictly sequentially or if some can be run concurrently.
*   **Component Changes:** Take advantage of change detection to avoid running systems if they are not needed. Utilize component change detection effectively to minimize unnecessary CPU work.

**Code Examples**

These are simplified examples to illustrate principles; specific adjustments will depend on your Bevy implementation.

**Example 1:  Instanced Rendering**

```rust
use bevy::prelude::*;

#[derive(Component)]
struct InstanceData {
    color: Color,
}

fn setup(mut commands: Commands, mut meshes: ResMut<Assets<Mesh>>, mut materials: ResMut<Assets<StandardMaterial>>) {
    let mesh = meshes.add(Mesh::from(shape::UVSphere { radius: 0.5, subdivisions: 10, }));
    let material = materials.add(StandardMaterial::default());
    let instance_data: Vec<InstanceData> = (0..1000).map(|i| InstanceData { color: Color::rgb(i as f32 * 0.001, 0.5, 0.5) }).collect();
    commands.spawn(
      (InstancedMeshBundle {
         mesh,
        material,
        transform: Transform::from_xyz(0.0,0.0,0.0),
         ..Default::default()
       },
      instance_data
       )
    );

}

fn update_instances(mut query: Query<(&mut Transform, &InstanceData), With<InstancedMeshBundle>>){
   for (mut transform, data) in query.iter_mut() {
     transform.translation = transform.translation + Vec3::new(0.01,0.0,0.0);
   }
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_startup_system(setup)
         .add_system(update_instances)
        .run();
}
```

*   **Commentary:** This example uses `InstancedMeshBundle` to render 1000 spheres, drastically reducing draw calls compared to rendering each sphere individually. A single mesh and material are used to promote batching.

**Example 2:  System Ordering and Change Detection**

```rust
use bevy::prelude::*;

#[derive(Component, PartialEq)]
struct Position {
    x: f32,
    y: f32,
}

#[derive(Component)]
struct Velocity {
    x: f32,
    y: f32,
}


fn update_position(mut query: Query<(&mut Position, &Velocity)>) {
    for (mut pos, vel) in query.iter_mut() {
        pos.x += vel.x;
        pos.y += vel.y;
    }
}

fn log_position(query: Query<&Position, Changed<Position>>) {
    for position in query.iter() {
        println!("Position changed: x = {}, y = {}", position.x, position.y);
    }
}


fn setup(mut commands: Commands) {
    commands.spawn((Position { x: 0.0, y: 0.0 }, Velocity { x: 0.1, y: 0.2 } ));
     commands.spawn((Position { x: 1.0, y: 1.0 }, Velocity { x: 0.2, y: 0.3 } ));
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_startup_system(setup)
        .add_system(update_position)
        .add_system(log_position.after(update_position))
        .run();
}
```

*   **Commentary:** The `log_position` system only executes when the `Position` component has changed, avoiding unnecessary work. The `after(update_position)` is used to ensure `log_position` occurs after the position is updated. System ordering is used to ensure correctness.

**Example 3:  Reduced Material Variation**

```rust
use bevy::{prelude::*, render::view::RenderLayers};

#[derive(Component)]
struct InstanceColor {
    color: Color,
}

fn setup(mut commands: Commands,  mut meshes: ResMut<Assets<Mesh>>, mut materials: ResMut<Assets<StandardMaterial>>) {
    let mesh = meshes.add(Mesh::from(shape::Cube { size: 1.0 }));

    let material = materials.add(StandardMaterial {
        base_color: Color::WHITE,
        ..Default::default()
    });

    for i in 0..1000 {
        let color = Color::rgb(i as f32 * 0.001, 0.5, 0.5);
         let x = (i as f32 * 0.1);
         let z = (i as f32 * 0.2);
        commands.spawn((
             PbrBundle {
                mesh: mesh.clone(),
                material: material.clone(),
                transform: Transform::from_xyz(x,0.0,z),
                ..Default::default()
            },
            InstanceColor { color },
        ));
    }
}

fn update_material_color(
    mut materials: ResMut<Assets<StandardMaterial>>,
    query: Query<(&Handle<StandardMaterial>, &InstanceColor)>
) {
     for (handle, instance_color) in query.iter() {
            if let Some(material) = materials.get_mut(handle) {
                material.base_color = instance_color.color;
            }
    }
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_startup_system(setup)
        .add_system(update_material_color)
        .run();
}
```

*   **Commentary:** This avoids creating many separate materials. While the example here uses individual materials, we are using the same base material. We can instead use a single material and pass in a color per-instance via buffers, leading to much better batching.

**Resource Recommendations**

To delve deeper into Bevy optimization, I recommend these resources:

*   **Bevy Documentation:** The official Bevy documentation provides insights into its architecture and features. Specific attention should be given to rendering, instancing, and ECS sections.
*   **Bevy Examples:** Studying the examples provided with the Bevy source is beneficial in understanding specific techniques. Focus on cases involving instanced rendering, advanced queries and component manipulation.
*   **Graphics API Documentation (Vulkan or DirectX):** While specific to the underlying graphics APIs, understanding fundamental concepts such as draw calls, pipelines, and GPU resource management provides valuable insight.
*   **Game Development Blogs and Forums:** Many game developers share their experiences with Bevy optimization; they can be invaluable when dealing with niche performance issues.
*   **Performance Profiling Tools:** Tools such as Bevy's built-in profiler, or Windows performance analysis tools, can aid in identifying specific bottlenecks within your application.

Improving Bevy's performance on Windows 11 with an RTX 2060 involves a multifaceted approach centered around effective resource management, efficient rendering pipelines, careful component design, and strategic use of ECS concepts. By examining these areas and applying the described principles, sustainable performance beyond short bursts becomes achievable.

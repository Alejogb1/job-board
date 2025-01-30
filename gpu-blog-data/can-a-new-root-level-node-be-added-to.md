---
title: "Can a new root-level node be added to a device tree overlay?"
date: "2025-01-30"
id: "can-a-new-root-level-node-be-added-to"
---
The Device Tree (DT) structure, while typically immutable at runtime, allows for dynamic modifications via overlays, but the scope of these changes is primarily limited to augmenting or modifying existing nodes defined in the base DT. Direct addition of a new root-level node during overlay application is *not* supported by the standard Device Tree Overlay (DTO) mechanisms. My experience across multiple embedded Linux projects, spanning custom SoCs to commercially available single-board computers, has consistently reinforced this limitation. Instead, modifications are targeted using *path targeting*, relying on node labels already present within the base device tree.

Here’s a breakdown of why this restriction exists and how to work within these constraints:

The core principle of DTOs is to apply a set of delta changes – additions, removals, and modifications – *relative* to an existing, compiled base DT. The base DT provides the fundamental structure and naming context, against which the overlay is applied. The DTO specification, both in its source format (.dts) and compiled format (.dtbo), uses the node paths to identify the locations for these modifications. Since a root-level node implicitly lacks a parent and the base DT defines the root node (denoted as `/`), there isn't a clearly defined path for an overlay to reference and introduce a new root-level node. It's analogous to trying to append a new top-level directory in a filesystem without an existing mount point for this operation. The path targeting mechanism simply does not facilitate this.

Instead of creating new root-level nodes, overlay files operate in the following ways:

1.  **Node Modification:** Existing nodes can have their properties altered. This includes changing the values of properties like `reg`, `compatible`, or `status`.

2.  **Node Addition:** New child nodes can be added to an existing parent node within the base tree, effectively extending the device tree hierarchy.

3.  **Node Removal:** Entire existing nodes and their sub-tree can be removed.

4.  **Fragment Application:** Overlays can be split into fragments to allow for more complex changes that may need to be applied at different locations. These fragments are composed of node modifications, additions, or removals, as outlined above.

The inability to add a root-level node isn’t an arbitrary restriction; it stems from the fundamental design of how the DT is interpreted and how overlays are applied. Adding new root nodes would complicate the DT structure, introducing potential conflicts and making the system more difficult to manage. The base device tree establishes the primary structure and the DTO system is designed to augment and refine that structure.

Now, let's illustrate with some code examples.

**Example 1: Adding a Child Node**

Let's say our base DT has a node named `i2c0` representing an I2C controller. We want to add an I2C device, a temperature sensor, as a child node within `i2c0` via a DTO:

```dts
/dts-v1/;
/plugin/;

/ {
    fragment@0 {
        target = <&i2c0>;
        __overlay__ {
            temp_sensor@5a {
                compatible = "my-temp-sensor";
                reg = <0x5a>;
                status = "okay";
            };
        };
    };
};
```

*   **Commentary:** The `target = <&i2c0>;` line identifies the target node for this fragment’s modifications using the *label* `i2c0` previously assigned in the base DT. The `__overlay__` section houses the changes, in this case, the new node `temp_sensor@5a` added as a child of `i2c0`. This approach successfully integrates the new device into the existing DT hierarchy. Trying to place the `temp_sensor@5a` node outside the `fragment@0` would result in an error, as it would not adhere to path-based modification principles.

**Example 2: Modifying a Property**

Continuing from our previous example, suppose we later need to disable the temperature sensor, we achieve this through modifying the `status` property:

```dts
/dts-v1/;
/plugin/;

/ {
    fragment@0 {
        target = <&i2c0>;
        __overlay__ {
            temp_sensor@5a {
                status = "disabled";
            };
        };
    };
};
```

*   **Commentary:** This demonstrates modifying an *existing* node property. The node `temp_sensor@5a` exists from our first example, and here, its `status` is altered.  The overlay engine will find the `temp_sensor@5a` node under `i2c0` and apply this property change. This modification is scoped by the `target` node and illustrates a core functionality of DTOs.

**Example 3: Removing an entire node**

We can also remove a node as part of an overlay. In this example, let's assume we're removing a CAN bus node because we don't need it for this particular application. This can be done by leveraging the `delete-node` operator. Note that the node still has to exist in the base tree for removal.

```dts
/dts-v1/;
/plugin/;

/ {
    fragment@0 {
      target = <&can0>;
      __overlay__ {
        delete-node;
      };
    };
};
```

*   **Commentary:** This example removes the node labelled `can0` from the device tree. The `target` property indicates which node is going to be removed. The `delete-node` operator is the key element that tells the overlay subsystem to remove the node identified by the `target` property. This is a useful method for removing unnecessary functionality at runtime.

As you can see from the examples, all overlay operations are targeted toward a pre-existing node in the base tree. Adding a new root-level node is fundamentally outside the scope of these mechanisms.

If a specific driver or kernel component requires a device tree node to function, and this node is not already present in the base tree, a suitable solution is to adjust the base DT, rather than trying to achieve this at overlay level. Ideally, the base DT should include all foundational devices, and DTOs are used to customize for specifics or add dynamically discovered devices.

For more detailed information, I recommend consulting resources covering the Device Tree specification and associated tools. The official Linux kernel documentation provides in-depth details, along with guides for using the `dtc` compiler. Additionally, the resources provided by SoC vendors will typically include documentation specific to their platform's device tree configuration. Referencing examples within the Linux kernel source tree itself is an excellent way to gain practical understanding of the patterns used for configuring various hardware peripherals. Finally, embedded linux system books are a very good source of knowledge. They usually dedicate a whole chapter to understanding the device tree.

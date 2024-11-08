---
title: "Oculus Quest: Stuck on Single/Multi-Pass? Get It Working Now!"
date: '2024-11-08'
id: 'oculus-quest-stuck-on-single-multi-pass-get-it-working-now'
---

```csharp
// Assuming you have the Oculus XR Plugin installed and set up

using UnityEngine;
using UnityEngine.XR;

public class OculusSinglePassAndMultiPass : MonoBehaviour
{
    public bool UseSinglePass = false;

    void Start()
    {
        if (XRSettings.supportedDevices.Contains("Oculus"))
        {
            // Set the rendering path based on the chosen option
            XRSettings.stereoRenderingMode = UseSinglePass ? XRSettings.StereoRenderingMode.SinglePass : XRSettings.StereoRenderingMode.MultiPass;
        }
    }
}
```

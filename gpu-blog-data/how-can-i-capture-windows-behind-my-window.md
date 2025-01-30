---
title: "How can I capture windows behind my window using the GPU without altering window affinity?"
date: "2025-01-30"
id: "how-can-i-capture-windows-behind-my-window"
---
Capturing windows occluded by the currently active window, without altering their existing z-order or otherwise causing window affinity changes, presents a specific challenge in operating system graphics programming. The core issue revolves around the fact that standard window drawing operations, typically relying on the operating system's compositor, prioritize the active, top-most window. Direct access to the framebuffers of hidden windows is generally restricted for security and stability reasons. I've encountered this problem firsthand while developing a custom remote collaboration tool that required a "desktop preview" showing all windows, irrespective of their visibility. The techniques I've found to be effective involve leveraging specific operating system APIs that allow a degree of direct access to window content, rendered outside the scope of typical window drawing operations.

The crucial approach lies in bypassing the typical drawing pipeline and directly accessing the pixel data rendered by other processes into their respective window surfaces. This is accomplished by capturing a snapshot of the underlying framebuffers or, more accurately, their backing surfaces, using operating-system-specific functionalities that expose this capability for advanced use cases. This differs significantly from the typical ‘print screen’ method, which only captures the final rendered image from the compositor. Doing so ensures we don't interfere with the target window's normal drawing cycle or their z-order. We are interested in obtaining the window’s rendered content independently of how and whether the window is visible.

Windows operating systems, for example, expose the Desktop Window Manager (DWM) API, which provides functions capable of querying and accessing these hidden surfaces. Using DWM’s `DwmGetWindowAttribute` with specific attributes such as `DWMWA_EXTENDED_FRAME_BOUNDS` to obtain the absolute screen coordinates, and `DWMWA_VISIBLE_FRAME_BORDER_THICKNESS` to get the window border size, enables me to determine the precise bounding box of the window, even if it is partially or fully covered by other windows. More critically, the `DwmGetDxSharedSurface` or, in the latest Windows versions, `DwmGetCompositionSurfaceInfo`, can be used to retrieve a handle to the shared Direct3D surface associated with the window. This is the heart of the solution, as it provides a direct path to the rendered pixel data.

Once I have a handle to the shared Direct3D surface, I can use Direct3D APIs to create a texture representing the content of the window’s rendered surface. This texture can then be read back to the CPU, or, as is generally more performant and preferred, used directly on the GPU to render into the preview application's viewport. This entire process avoids any interference with window affinity or z-order and only works on windowed applications.

Here are three code examples illustrating the process on the Windows platform, focusing on Direct3D and the DWM APIs:

**Example 1: Acquiring a Shared Surface Handle**

This example demonstrates the initial step of retrieving the shared surface handle. It uses the `DwmGetWindowAttribute` function to retrieve the handle of the shared surface associated with a window.

```cpp
#include <windows.h>
#include <dwmapi.h>
#include <d3d11.h>
#include <iostream>

HRESULT GetWindowSharedSurfaceHandle(HWND hwnd, HANDLE* outHandle) {
    if (!hwnd || !outHandle) return E_INVALIDARG;

    HRESULT hr;
    DWM_SHARED_SURFACEINFO surfaceInfo = {};
    surfaceInfo.cbSize = sizeof(surfaceInfo);

    hr = DwmGetCompositionSurfaceInfo(hwnd, &surfaceInfo);
    if (FAILED(hr)) {
        std::cerr << "DwmGetCompositionSurfaceInfo failed: " << std::hex << hr << std::endl;
        return hr;
    }

   *outHandle = surfaceInfo.hSurface;
   return S_OK;
}

int main() {
    HWND targetWindow = FindWindow(NULL, L"Target Window Title"); // Replace with your actual window title.
    if (!targetWindow) {
        std::cerr << "Failed to find target window." << std::endl;
        return 1;
    }
    HANDLE sharedSurfaceHandle = nullptr;
    HRESULT hr = GetWindowSharedSurfaceHandle(targetWindow, &sharedSurfaceHandle);
    if (SUCCEEDED(hr)){
      std::cout << "Successfully retrieved handle: " << sharedSurfaceHandle << std::endl;
      // Use sharedSurfaceHandle
      CloseHandle(sharedSurfaceHandle); //Remember to close the handle when done.
    }

    return 0;
}
```

**Commentary:** This example uses the `FindWindow` function to locate a target window by its title. This is a basic example and you'd typically implement a more sophisticated targetting function. Crucially, I obtain the shared surface handle via `DwmGetCompositionSurfaceInfo`. It also includes a standard handle check and closure. This handle will be used in the next example to create a Direct3D surface from the handle. Error handling is important; all operations return an `HRESULT` which should be checked for success.

**Example 2: Creating a Direct3D Texture from the Shared Surface**

This example shows how to use the shared surface handle to create a Direct3D texture that can be used in the program's renderer.

```cpp
#include <windows.h>
#include <dwmapi.h>
#include <d3d11.h>
#include <iostream>
#include <wrl/client.h> //For ComPtr

using namespace Microsoft::WRL;

HRESULT CreateTextureFromSharedSurface(ID3D11Device* device, HANDLE sharedHandle, ID3D11Texture2D** outTexture) {
  if (!device || !sharedHandle || !outTexture) return E_INVALIDARG;

  ComPtr<ID3D11Device1> device1;
  HRESULT hr = device->QueryInterface(IID_PPV_ARGS(&device1));
  if(FAILED(hr)){
    std::cerr << "Failed to query ID3D11Device1: " << std::hex << hr << std::endl;
      return hr;
  }

  ComPtr<ID3D11Resource> sharedResource;

  hr = device1->OpenSharedResource(sharedHandle, IID_PPV_ARGS(&sharedResource));

  if(FAILED(hr)){
    std::cerr << "Failed to open shared resource: " << std::hex << hr << std::endl;
    return hr;
  }


  hr = sharedResource.As(&*outTexture);
  if (FAILED(hr)){
      std::cerr << "Failed to cast to texture2D : " << std::hex << hr << std::endl;
      return hr;
  }

  return S_OK;
}

int main(){
    // Assume you have a valid D3D11 device.
    // This part is omitted for brevity.

    ID3D11Device* d3dDevice; // Replace this with your device

    //  ... (Initialization of d3dDevice) ...

    HWND targetWindow = FindWindow(NULL, L"Target Window Title");
    HANDLE sharedSurfaceHandle;
    HRESULT hr = GetWindowSharedSurfaceHandle(targetWindow, &sharedSurfaceHandle);

    if(FAILED(hr))
    {
        return 1;
    }
    ComPtr<ID3D11Texture2D> texture;

    hr = CreateTextureFromSharedSurface(d3dDevice, sharedSurfaceHandle, texture.GetAddressOf());
    if (SUCCEEDED(hr)){
      // Use the texture. The texture now contains the rendered content
      std::cout << "Successfully Created Texture." << std::endl;
    }
    if (sharedSurfaceHandle){
        CloseHandle(sharedSurfaceHandle);
    }
    return 0;
}
```
**Commentary:** This example uses the `ID3D11Device1::OpenSharedResource` to open the shared surface obtained in Example 1, returning a generic `ID3D11Resource`, which is then cast to a `ID3D11Texture2D`. The example assumes you already have a valid Direct3D 11 device initialized. This would include creating the device context and swap chains. The `ComPtr` is used to manage the lifetime of the resource, a standard practice in COM programming. Again, thorough error checking is included. The `sharedSurfaceHandle` needs to be closed when the surface isn't needed anymore.

**Example 3: Rendering the Captured Texture**

This example gives an outline for using the created texture. It shows the basic rendering pipeline.

```cpp
#include <d3d11.h>
#include <wrl/client.h>

using namespace Microsoft::WRL;

void RenderTexture(ID3D11DeviceContext* context, ID3D11ShaderResourceView* textureView,
                 ID3D11RenderTargetView* renderTargetView,
                 ID3D11SamplerState* samplerState)
{

        context->OMSetRenderTargets(1, &renderTargetView, nullptr);
        context->ClearRenderTargetView(renderTargetView, (float*) & {0.0f, 0.0f, 0.0f, 1.0f}); //Clear the background
    
        D3D11_VIEWPORT viewport = {};
    
        // Assuming viewport size matches render target size.
        viewport.Width = 800;
        viewport.Height = 600;
        viewport.MinDepth = 0.0f;
        viewport.MaxDepth = 1.0f;
    
        context->RSSetViewports(1, &viewport);
    
        // Setup vertex buffer & shader pipeline is ommited for brevity
        // You'll need to setup the vertex buffer and shader pipeline to render a quad.

       //Set resources
       context->PSSetShaderResources(0, 1, &textureView);
       context->PSSetSamplers(0, 1, &samplerState);

       //Draw quad
       context->Draw(4,0);


}

int main() {
    // Assume valid D3D11 Device & Context and texture created with previous code.
    ID3D11DeviceContext* d3dContext; //Replace with your context
    ID3D11RenderTargetView* renderTargetView; // Replace with your render target.
    ComPtr<ID3D11Texture2D> sharedTexture; //Assume texture created before

    // ... Initializations
    
    D3D11_SHADER_RESOURCE_VIEW_DESC shaderViewDesc = {};
    shaderViewDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    shaderViewDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
    shaderViewDesc.Texture2D.MostDetailedMip = 0;
    shaderViewDesc.Texture2D.MipLevels = 1;

    ComPtr<ID3D11ShaderResourceView> textureView;
    HRESULT hr = d3dDevice->CreateShaderResourceView(sharedTexture.Get(), &shaderViewDesc, textureView.GetAddressOf());

    // Initialize the sampler state
    D3D11_SAMPLER_DESC samplerDesc;
    ZeroMemory(&samplerDesc, sizeof(samplerDesc));
    samplerDesc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
    samplerDesc.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
    samplerDesc.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
    samplerDesc.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
    ComPtr<ID3D11SamplerState> samplerState;

    hr = d3dDevice->CreateSamplerState(&samplerDesc, samplerState.GetAddressOf());
    if (FAILED(hr)) return 1;
    
    RenderTexture(d3dContext,textureView.Get(), renderTargetView, samplerState.Get());

    return 0;

}

```

**Commentary:** This example demonstrates how to render the captured window texture to a viewport. It first sets the render target view and clears the view. After, it sets the viewport. Finally, it creates a texture view from the captured texture and the sampler state, which are then used in a basic quad rendering pipeline. The exact vertex shader and pixel shader implementations are omitted for brevity and are required to have a functioning rendering pipeline. This example is meant to be integrated within the renderer of the application capturing the desktop.

For deeper understanding, I recommend reviewing the documentation for the Windows DWM API, specifically focusing on `DwmGetWindowAttribute` and `DwmGetCompositionSurfaceInfo` or, for older versions, `DwmGetDxSharedSurface`, and the Direct3D 11 or 12 documentation pertaining to texture creation and resource management.  Consulting resources related to shared surfaces in Direct3D will also be beneficial. These resources will provide a more in-depth view of the individual API functionalities and their nuances.

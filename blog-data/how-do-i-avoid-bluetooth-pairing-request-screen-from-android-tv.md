---
title: "How do I avoid Bluetooth pairing request screen from android TV?"
date: "2024-12-23"
id: "how-do-i-avoid-bluetooth-pairing-request-screen-from-android-tv"
---

Let's dive straight in, shall we? Avoiding that pesky Bluetooth pairing request screen on Android TV—it's a problem I've certainly encountered more than once. It often stems from the system interpreting peripheral advertising as an intention to pair, which, as we know, isn't always the case. I've spent my fair share of evenings debugging similar issues, and I've learned there isn’t one universal solution, but a few robust strategies. The core challenge lies in Android's Bluetooth stack automatically initiating pairing dialogues when it encounters connectable devices broadcasting services that it considers 'interesting' – typically HID (Human Interface Device) profiles. We need to be more specific in our requirements and, in some cases, filter out advertising data.

First, let’s understand the context. Android's Bluetooth implementation follows a service discovery protocol. When a device advertises a service, Android’s Bluetooth service can interpret this as a signal to prompt for pairing. This is useful for keyboards and mice, less so for other peripherals or custom devices. The key here is to understand the different types of Bluetooth profiles and how android handles them. Typically, if you're developing custom hardware that's advertising over Bluetooth, the first step is to ensure you're *not* using a standard HID profile, unless you specifically *want* the pairing to trigger.

Let's look at practical solutions. There are essentially three routes we can consider, and the best approach often depends on the context of your application or custom device:

**1. Service Filtering via Custom UUIDs:**

The most precise method involves ensuring your custom Bluetooth peripheral advertises a unique, custom UUID (Universally Unique Identifier). This requires you to define a custom GATT service that is not tied to any standard Bluetooth profile. By implementing this, you effectively communicate that this device is not an HID-style device, and will not trigger automatic pairing requests.

Here's an illustrative code snippet using the Android Bluetooth API:

```java
    // Assume `mBluetoothAdapter` is a valid instance of BluetoothAdapter

    private void startAdvertising(BluetoothAdapter mBluetoothAdapter) {
    BluetoothLeAdvertiser advertiser = mBluetoothAdapter.getBluetoothLeAdvertiser();
    AdvertiseSettings settings = new AdvertiseSettings.Builder()
            .setAdvertiseMode(AdvertiseSettings.ADVERTISE_MODE_LOW_LATENCY)
            .setConnectable(true)
            .setTimeout(0)
            .setTxPowerLevel(AdvertiseSettings.ADVERTISE_TX_POWER_HIGH)
            .build();

    // Replace with your custom UUID
    ParcelUuid customServiceUuid = new ParcelUuid(UUID.fromString("a1b2c3d4-e5f6-7890-1234-567890abcdef"));

    AdvertiseData data = new AdvertiseData.Builder()
            .addServiceUuid(customServiceUuid)
            .setIncludeDeviceName(false) //Optional, if you don't need the device name
             .build();

    advertiser.startAdvertising(settings, data, advertiseCallback);
    }


    private final AdvertiseCallback advertiseCallback = new AdvertiseCallback() {
        @Override
        public void onStartSuccess(AdvertiseSettings settingsInEffect) {
            Log.i("Bluetooth", "Advertising started successfully");
        }

        @Override
        public void onStartFailure(int errorCode) {
            Log.e("Bluetooth", "Advertising failed to start: " + errorCode);
        }
    };
```
In this code, the `customServiceUuid` uniquely identifies our custom service, preventing the Android system from interpreting it as a standard HID profile and triggering the pairing request dialog. Crucially, we *do not* include any generic or HID-related service UUIDs. On the central side (the Android TV), the application should also filter for this specific UUID.

**2. Filtering by Device Name (Less Reliable):**

A less reliable but sometimes useful method is to filter by device name on the Android side before attempting a connection. This approach relies on inspecting the advertising data and rejecting devices based on specific name patterns. It is less robust because the device name can be changed by the user or by the device's software.

Here's how you can approach this:

```java
    private void scanForDevices(BluetoothAdapter mBluetoothAdapter) {
       BluetoothLeScanner scanner = mBluetoothAdapter.getBluetoothLeScanner();

    ScanFilter filter = new ScanFilter.Builder()
           //Example: Ignore any device that begins with "HID-" (not recommended for production)
            .setDeviceName("HID-")
             .build();
    List<ScanFilter> filters = new ArrayList<>();
    filters.add(filter);

    ScanSettings settings = new ScanSettings.Builder()
            .setScanMode(ScanSettings.SCAN_MODE_LOW_LATENCY)
            .build();

    scanner.startScan(filters, settings, scanCallback);

}

private final ScanCallback scanCallback = new ScanCallback() {
    @Override
    public void onScanResult(int callbackType, ScanResult result) {
        BluetoothDevice device = result.getDevice();
        if (device.getName() != null && !device.getName().startsWith("HID-")) {
            // process the device
            Log.i("Bluetooth", "Discovered device: "+ device.getName());
            //Now you can decide to connect here
            // connectToDevice(device)
        } else {
            Log.i("Bluetooth", "Filtered device : "+ device.getName() );
        }
    }


    @Override
    public void onBatchScanResults(List<ScanResult> results) {
        for (ScanResult result : results) {
            BluetoothDevice device = result.getDevice();
           if (device.getName() != null && !device.getName().startsWith("HID-")){
              // process the device
              Log.i("Bluetooth", "Discovered device: "+ device.getName());
              //Now you can decide to connect here
              //connectToDevice(device)
           }else{
             Log.i("Bluetooth", "Filtered device : "+ device.getName() );
           }
        }
    }


    @Override
    public void onScanFailed(int errorCode) {
       Log.e("Bluetooth", "Scan Failed, error code :"+ errorCode);
    }

};
```

This code demonstrates filtering devices based on their name. While it’s straightforward, relying on device names can be unreliable. The filtering should always be an *additional* step not the main solution. It's often more of a stop-gap measure than a solid fix.

**3. Modifying the Advertising Interval (Last Resort):**

In a few situations where you lack control over the advertising peripheral (which should be avoided), reducing the advertising interval (i.e., making the device advertise less frequently) can sometimes make it less likely to trigger the immediate pairing request screen. However, this method is highly dependent on the Android TV's implementation and should be considered a last resort. Additionally, it can negatively affect the speed with which the device can be found and connected, and should be avoided if at all possible.

In conclusion, the most robust solution is to use custom UUIDs to advertise your device and handle the connection in a controlled way on your Android TV application. Relying solely on device name filters is brittle, and modifying the advertising interval should only be considered as a last resort. This avoids the pairing request because the system no longer recognizes the device as something that inherently requires pairing, providing a much cleaner and reliable experience.

For deeper insights into Bluetooth Low Energy (BLE) and GATT services, I’d recommend exploring "Bluetooth Low Energy: The Developer's Handbook" by Robin Heydon, along with the official Bluetooth specification documents, as these resources provide the needed foundational understanding. For an in depth dive into the Android Bluetooth Stack itself, the source code for the Android open source project is a good place to start looking.

Remember, the more you control the advertisement and handling process, the less likely you are to encounter these automatic pairing prompts, leading to a more refined user experience.

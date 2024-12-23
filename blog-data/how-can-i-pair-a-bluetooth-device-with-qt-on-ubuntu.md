---
title: "How can I pair a Bluetooth device with Qt on Ubuntu?"
date: "2024-12-23"
id: "how-can-i-pair-a-bluetooth-device-with-qt-on-ubuntu"
---

Alright, let's tackle this Bluetooth pairing issue with Qt on Ubuntu. I've spent considerable time battling similar setups in my past projects, specifically when crafting embedded systems with custom user interfaces. It's a process that involves navigating a few layers of abstraction, and getting the details correct makes all the difference. The key is understanding how Qt integrates with the underlying system's Bluetooth stack, usually BlueZ on Linux systems like Ubuntu.

First, the overall picture is this: Qt doesn't directly handle Bluetooth communication at the low level; rather, it provides an API, primarily through the `QBluetooth` module, which serves as an interface to BlueZ. So, your code is essentially sending commands to the system's Bluetooth daemon. Before diving into code, we need to ensure that BlueZ is properly installed and functioning on your Ubuntu machine. Usually, this is installed out of the box, but it's worth verifying:

```bash
sudo apt update
sudo apt install bluez bluez-tools
```

Once installed, you can use `bluetoothctl` from the command line to verify that your Bluetooth adapter is recognized and operating. The commands `devices` and `scan on` are handy for identifying discoverable devices. If you're experiencing problems at this stage, troubleshoot system-level issues first because Qt will rely on this foundational layer.

Now let's move to Qt. The crucial class here is `QBluetoothDeviceDiscoveryAgent`. This class manages the discovery process, and when devices are located, it emits the `deviceDiscovered` signal. Once we have a list of devices, we'll need to handle pairing.

Here's a snippet demonstrating a basic device discovery:

```cpp
#include <QBluetoothDeviceDiscoveryAgent>
#include <QBluetoothDeviceInfo>
#include <QDebug>

class BluetoothManager : public QObject
{
    Q_OBJECT
public:
    BluetoothManager(QObject *parent = nullptr) : QObject(parent) {
        discoveryAgent = new QBluetoothDeviceDiscoveryAgent(this);
        connect(discoveryAgent, &QBluetoothDeviceDiscoveryAgent::deviceDiscovered,
                this, &BluetoothManager::deviceDiscovered);
        connect(discoveryAgent, &QBluetoothDeviceDiscoveryAgent::finished,
                this, &BluetoothManager::scanFinished);
    }

    void startDiscovery() {
        qDebug() << "Starting device discovery...";
        discoveryAgent->start();
    }

public slots:
    void deviceDiscovered(const QBluetoothDeviceInfo &deviceInfo) {
        qDebug() << "Device found:" << deviceInfo.name() << "Address:" << deviceInfo.address().toString();
        discoveredDevices.append(deviceInfo);
    }

    void scanFinished() {
       qDebug() << "Device discovery finished.";
       // Process discovered devices, e.g., display a list to the user.
    }

private:
    QBluetoothDeviceDiscoveryAgent *discoveryAgent;
    QList<QBluetoothDeviceInfo> discoveredDevices;
};


int main(int argc, char *argv[])
{
    QCoreApplication app(argc, argv);

    BluetoothManager manager;
    manager.startDiscovery();

    return app.exec();
}
#include "main.moc"
```

This is a straightforward example. We instantiate `QBluetoothDeviceDiscoveryAgent`, connect the necessary signals to our slots, start the discovery process, and then log the found devices to the console. Remember to add `QT += bluetooth` in your `.pro` file for this to compile successfully. The `main.moc` file at the end is crucial for the signal/slot mechanism.

Now, let's tackle pairing. Once a device has been found, we can attempt pairing. The pairing logic is often initiated by user interaction, perhaps clicking a button on the GUI. We need to use the `QBluetoothLocalDevice` class to handle the pairing.

Here's a code snippet showing how to initiate the pairing process. I'm going to assume you have a `QBluetoothDeviceInfo` object representing the device you want to pair, perhaps selected from a list of discovered devices:

```cpp
#include <QBluetoothLocalDevice>
#include <QBluetoothDeviceInfo>
#include <QDebug>


class PairingManager : public QObject
{
  Q_OBJECT
public:
  PairingManager(QObject *parent = nullptr) : QObject(parent) {
       localDevice = new QBluetoothLocalDevice(this);
  }

    void pairDevice(const QBluetoothDeviceInfo &deviceInfo) {
        qDebug() << "Initiating pairing with device:" << deviceInfo.address().toString();
      if (localDevice->pairingStatus(deviceInfo.address()) == QBluetoothLocalDevice::Paired) {
          qDebug() << "Device is already paired.";
          return;
      }
      connect(localDevice, &QBluetoothLocalDevice::pairingFinished,
              this, &PairingManager::pairingFinished);
      localDevice->requestPairing(deviceInfo.address(), QBluetoothLocalDevice::Paired);
    }

public slots:
    void pairingFinished(const QBluetoothAddress &address, QBluetoothLocalDevice::Pairing pairingResult) {
        qDebug() << "Pairing with device:" << address.toString() << " Result:" << pairingResult;
    }

private:
  QBluetoothLocalDevice *localDevice;

};

int main(int argc, char *argv[])
{
  QCoreApplication app(argc, argv);

  // Assuming 'selectedDeviceInfo' is a previously discovered device.
  QBluetoothDeviceInfo selectedDeviceInfo;
    selectedDeviceInfo.setAddress(QBluetoothAddress("your_device_address"));//<-- Replace with actual address

   PairingManager pairingManager;
   pairingManager.pairDevice(selectedDeviceInfo);

    return app.exec();
}
#include "main.moc"
```

Remember to replace `"your_device_address"` with the actual address of the device you intend to pair with. This code first checks if the device is already paired and initiates the pairing process through `requestPairing`. The `pairingFinished` signal indicates the result of the pairing attempt. In some cases, especially for devices that require a pairing pin, the `pairingDisplayPinCode` or `pairingConfirmation` signals might be needed. These require more complex logic, potentially involving a user input interface, as a device may require pin entry on your application.

Finally, let's consider situations where a device might require pin confirmation or pairing via a confirmation key.

```cpp
#include <QBluetoothLocalDevice>
#include <QBluetoothDeviceInfo>
#include <QDebug>
#include <iostream> // For using std::cin to read user confirmation

class PinPairingManager : public QObject
{
  Q_OBJECT
public:
    PinPairingManager(QObject *parent = nullptr) : QObject(parent) {
        localDevice = new QBluetoothLocalDevice(this);
        connect(localDevice, &QBluetoothLocalDevice::pairingDisplayPinCode,
                this, &PinPairingManager::displayPinCode);
        connect(localDevice, &QBluetoothLocalDevice::pairingConfirmation,
                 this, &PinPairingManager::confirmationNeeded);
        connect(localDevice, &QBluetoothLocalDevice::pairingFinished,
             this, &PinPairingManager::pairingFinished);
    }

  void pairDevice(const QBluetoothDeviceInfo &deviceInfo) {
        qDebug() << "Initiating pairing with device requiring pin/confirmation: " << deviceInfo.address().toString();
        if(localDevice->pairingStatus(deviceInfo.address()) == QBluetoothLocalDevice::Paired) {
            qDebug() << "Device already paired.";
            return;
        }
        localDevice->requestPairing(deviceInfo.address(), QBluetoothLocalDevice::Paired);

  }
public slots:
    void displayPinCode(const QBluetoothAddress &address, const QString &pinCode) {
        qDebug() << "Pin code required for device: " << address.toString() << " Pin: " << pinCode;
        // You would normally display this pin code on a UI. For demonstration, we'll just output it here.
        // Then, you would inform the user to enter this pin code on the device itself.
    }

   void confirmationNeeded(const QBluetoothAddress &address){
       qDebug() << "Confirmation needed for device: " << address.toString() << ". Input 'y' to confirm:";
       char response;
       std::cin >> response;
       if(response == 'y' || response == 'Y'){
           localDevice->confirmPairing(address, true);
       } else {
           localDevice->confirmPairing(address, false);
       }

    }


   void pairingFinished(const QBluetoothAddress &address, QBluetoothLocalDevice::Pairing pairingResult) {
        qDebug() << "Pairing with device:" << address.toString() << " Result:" << pairingResult;
    }

private:
    QBluetoothLocalDevice *localDevice;

};

int main(int argc, char *argv[])
{
  QCoreApplication app(argc, argv);

  // Assuming 'selectedDeviceInfo' is a previously discovered device.
  QBluetoothDeviceInfo selectedDeviceInfo;
    selectedDeviceInfo.setAddress(QBluetoothAddress("your_device_address"));//<-- Replace with actual address

    PinPairingManager pinPairingManager;
    pinPairingManager.pairDevice(selectedDeviceInfo);

    return app.exec();
}
#include "main.moc"
```

Here, we handle both pin code display and confirmation requests from the device. The `displayPinCode` slot is invoked when a pin is required, which you might present on the user interface. The `confirmationNeeded` slot allows confirmation of a pairing. For simplicity, I've used `std::cin` to get confirmation from the user, however in a real-world scenario, you'd use a UI element like a confirmation dialogue.

For in-depth knowledge, I recommend consulting the official Qt documentation for the `QBluetooth` module, and also the *BlueZ API Documentation* found on the freedesktop.org website to fully grasp the underlying mechanism. *Linux Device Drivers* by Jonathan Corbet is a great resource for understanding how Linux manages hardware, including Bluetooth, if you want a more profound understanding of the lower layers. Additionally, searching the *Bluetooth Core Specification* documentation on bluetooth.com is a valuable resource for understanding the intricacies of the Bluetooth protocols.

Pairing Bluetooth devices in Qt is a process of coordinating Qt's APIs with BlueZ system services. Always begin by verifying the proper setup of BlueZ, then progressively move to your application logic. By tackling it step-by-step, you can confidently manage Bluetooth device pairings within your Qt application.

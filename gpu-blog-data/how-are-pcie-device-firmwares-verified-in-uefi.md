---
title: "How are PCIe device firmwares verified in UEFI secure boot?"
date: "2025-01-30"
id: "how-are-pcie-device-firmwares-verified-in-uefi"
---
The integrity of PCIe device firmware during UEFI Secure Boot relies heavily on cryptographic verification, specifically leveraging the Authenticated Variable service and the Secure Boot policy databases. From my experience debugging embedded systems, I've consistently seen that this process mirrors the validation flow for system firmware itself but extends to include these peripherals which directly impact the overall boot security posture. A failure to properly secure PCIe devices presents a significant vulnerability, since they can be vectors for malware to establish early in the boot process, circumventing later software layers.

The mechanism begins by requiring the PCIe option ROM (or firmware update capsules) to be signed with a cryptographic key that is trusted by the system’s UEFI firmware. This trust is established through the platform’s Secure Boot keys, often PK, KEK, and db. The Secure Boot policy itself dictates whether or not a specific signature, or absence thereof, will allow the firmware to execute. When the system power-on is initiated, the UEFI firmware enumerates the PCIe bus and locates devices with option ROMs or firmware descriptors. The enumeration process establishes memory mappings for the Option ROM or firmware region. The critical step now involves the signature verification. UEFI does not directly execute the option ROM or firmware. Instead, the UEFI environment first checks a signature using cryptographic primitives and the platform’s secure boot keys stored in Non-Volatile memory.

The Authenticated Variable service is crucial here. UEFI Secure Boot relies upon UEFI variables stored in non-volatile storage to manage its policies, including the secure boot keys mentioned earlier. The `db` variable, which holds Allowed Signatures, is a key component. The firmware image is hashed, and the resultant hash is compared to a value extracted from the digital signature. If both match, the device's firmware is considered legitimate, at least from the perspective of the defined policy. If not, the boot process halts or, at minimum, prevents the loading of the device's option ROM or execution of update firmware. The specifics depend on how secure boot policy is configured.

The process can also involve checking against `dbx`, the Forbidden Signatures database. This enables revoking previously trusted firmware images if a security vulnerability is discovered later. The process isn't a single static check, but a fluid verification happening before any firmware execution. The use of these databases allows administrators to maintain some control over what firmware gets accepted. The option ROM itself is not loaded directly; instead, it is verified first. Only once the signature is confirmed can the UEFI environment load the ROM and initialize the device.

There is also the possibility of using Signed Firmware Update Capsules. These are commonly used when updating existing firmware after the initial manufacturing process. They contain updated firmware for the PCIe device, along with a digital signature. During the update, the secure boot policies and the authenticated variables come into play again to verify the integrity and authenticity of the update capsule before flashing any new data to the device.

Now let's delve into some practical illustrations. These examples assume familiarity with basic programming concepts in a C-like environment, and represent simplified views of what a complex process looks like.

**Example 1: Basic Signature Verification (Conceptual)**

```c
typedef struct {
    unsigned char* firmwareImage;
    size_t firmwareSize;
    unsigned char* signature;
    size_t signatureSize;
} FirmwareData;

typedef struct {
    unsigned char* publicKeys;
    size_t keyCount;
} TrustedKeys;

//Simplified function for hash calculation, not a cryptographic hash
unsigned char* generateHash(unsigned char* data, size_t size){
  unsigned char *hash = malloc(size);
  for(int i=0;i<size;i++){
    hash[i] = data[i] + 1;
  }
  return hash;
}

// Simplified signature verification function
bool verifySignature(FirmwareData* firmware, TrustedKeys* trustedKeys){
    unsigned char* computedHash = generateHash(firmware->firmwareImage,firmware->firmwareSize);
    // Pseudo code: Compare signature with hash using a trusted public key (e.g., from db)
    // In reality this would involve actual crypto functions. This is for illustration.
    if(memcmp(computedHash, firmware->signature, firmware->signatureSize) == 0){
        free(computedHash);
        return true; //Signature is valid
    }
    free(computedHash);
    return false; //Signature is invalid
}

int main() {
    FirmwareData firmware;
    firmware.firmwareImage = (unsigned char*)"Example Firmware data here.";
    firmware.firmwareSize = strlen(firmware.firmwareImage);
    firmware.signature = (unsigned char*)"hash(Example Firmware data here.)+1"; //simulated signature value, in reality this would be cryptographically generated
    firmware.signatureSize = strlen(firmware.signature);

    TrustedKeys keys;
    keys.publicKeys = (unsigned char*)"Trusted Public Key"; //Simplified; in practice this is a list of public keys from PK, KEK, and db
    keys.keyCount = 1;

    if (verifySignature(&firmware, &keys)) {
        printf("Firmware signature is valid.\n");
    } else {
        printf("Firmware signature is invalid!\n");
    }

    return 0;
}
```

This code provides a very simplified model. It demonstrates the general idea of hashing firmware and comparing it to a signature, which in real UEFI implementation would use complex cryptography.  The `verifySignature` function highlights the core concept: verifying that the provided signature is associated with the actual firmware. The `generateHash` function is not secure, but illustrative to understand the hash generation concept. The real UEFI implementation relies upon cryptographic hash functions like SHA-256 and RSA based on parameters.

**Example 2: Secure Boot DB Lookup**

```c
typedef struct {
    unsigned char* signature;
    size_t signatureSize;
} SignatureEntry;

typedef struct {
  SignatureEntry* entries;
  size_t entryCount;
} SignatureDatabase;


// Function to check if the signature is in the Allowed Signature Database(db)
bool signatureInDB(unsigned char* signature, size_t signatureSize, SignatureDatabase* db){
  for(size_t i=0;i<db->entryCount;i++){
    if(memcmp(signature, db->entries[i].signature, signatureSize) == 0){
        return true; // Signature found in DB
    }
  }
  return false; // Signature not found in DB
}

int main(){
    unsigned char* firmwareSignature = (unsigned char*)"Another Simulated Sig"; // Simulated firmware signature. Real signature would be generated cryptographically.
    size_t sigSize = strlen(firmwareSignature);

    SignatureDatabase allowedDB;
    SignatureEntry entries[2];
    entries[0].signature = (unsigned char*)"Simulated signature 1";
    entries[0].signatureSize = strlen(entries[0].signature);
    entries[1].signature = (unsigned char*)"Another Simulated Sig";
    entries[1].signatureSize = strlen(entries[1].signature);

    allowedDB.entries = entries;
    allowedDB.entryCount = 2;

    if(signatureInDB(firmwareSignature, sigSize, &allowedDB)){
        printf("Firmware signature is present in the Allowed DB.\n");
    } else {
        printf("Firmware signature not found in the Allowed DB.\n");
    }
    return 0;
}
```
This example simplifies how a firmware signature is checked against a trusted signature database. The `signatureInDB` function simulates looking up a firmware’s signature in the allowed signature list within Secure Boot. If the signature is found, the firmware can be considered authorized according to the current policy. In reality, such comparison would involve more complex lookups of variable data.

**Example 3: Update Capsule Verification**

```c
typedef struct {
    unsigned char* capsuleData;
    size_t capsuleSize;
    unsigned char* signature;
    size_t signatureSize;
} UpdateCapsule;

// Simplified function to verify the capsule signature.
bool verifyUpdateCapsule(UpdateCapsule* capsule, TrustedKeys* trustedKeys){
    unsigned char* computedHash = generateHash(capsule->capsuleData, capsule->capsuleSize);
    if (memcmp(computedHash, capsule->signature, capsule->signatureSize) == 0){
        free(computedHash);
        return true;
    }
    free(computedHash);
    return false;
}

int main(){
    UpdateCapsule update;
    update.capsuleData = (unsigned char*)"New Firmware Data For Update";
    update.capsuleSize = strlen(update.capsuleData);
    update.signature = (unsigned char*)"hash(New Firmware Data For Update)+1";
    update.signatureSize = strlen(update.signature);


    TrustedKeys keys;
    keys.publicKeys = (unsigned char*)"Trusted Public Key";
    keys.keyCount = 1;

    if(verifyUpdateCapsule(&update, &keys)){
        printf("Update capsule verification successful.\n");
    } else {
        printf("Update capsule verification failed.\n");
    }
    return 0;
}

```
This final example portrays the verification of an update capsule. It highlights the use of the verification process for firmware updates. The `verifyUpdateCapsule` function indicates the logic that needs to be applied before updating a PCIe devices firmware. Only if the capsule’s signature is valid, then would the firmware update be permitted to proceed by the UEFI firmware.

In summary, the UEFI Secure Boot mechanism verifies PCIe device firmwares through rigorous cryptographic checks, including the use of signature databases and public key infrastructure. The process ensures that only trusted firmware executes on the PCIe devices, preventing potential attacks. Understanding this process is essential for anyone involved in embedded systems or security, and the examples provided give a glimpse of the complexities involved.

For further understanding of this process, I recommend consulting the UEFI Specification, particularly sections detailing Secure Boot and the Authenticated Variable service. Furthermore, the Platform Initialization (PI) specification provides more insights into device discovery and firmware loading. These documents, along with research papers on embedded security, provide detailed information on the UEFI secure boot process.

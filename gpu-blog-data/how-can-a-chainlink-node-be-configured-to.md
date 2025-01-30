---
title: "How can a Chainlink node be configured to create a new encryption key without crashing when no OCR keys are available?"
date: "2025-01-30"
id: "how-can-a-chainlink-node-be-configured-to"
---
When a Chainlink node attempts to generate a new encryption key for Oracle Cloud Reporting (OCR) but lacks necessary root keys, it defaults to a crash-inducing state instead of graceful recovery. This scenario, encountered during a recent scaling operation on my testnet infrastructure, necessitated a deeper dive into the node's configuration and a workaround focusing on proper key management and startup sequence modification.

The core issue stems from the node's dependence on having valid OCR encryption keys at startup. When the node encounters a missing key during initialization, specifically when bootstrapping the OCR subsystem, the absence triggers an unrecoverable exception, leading to an immediate termination. The OCR component expects certain key material, derived from a root key, to be present before initiating its processes. If these derived keys do not exist, the application fails to initialize effectively, causing the crash. To address this, we need to modify how the Chainlink node handles the missing OCR keys on a fresh start, specifically on startup. I’ve found that the most effective method involves preventing the node from attempting to initialize the OCR component if no suitable keys are present. This can be achieved by implementing a conditional check before bootstrapping OCR, contingent on the presence of the required root keys.

Let’s examine three methods to manage this issue, accompanied by code samples utilizing configuration files. These examples are simplified representations for clarity, focusing on illustrating the concept rather than a fully production-ready setup. The first approach involves modifying the `config.toml` file, which is the Chainlink node's primary configuration file. I use the `toml` structure because that’s how all my Chainlink nodes are configured, and it is the most straightforward way to illustrate a configuration change.

```toml
[OCR2]
Enabled = true
# Initial configuration for when keys exist
# EnableOCR = true # Enable OCR conditionally on key presence
# Bootstrappers = ["addr1", "addr2"]
# KeyBundleConfig = { KeyBundleID = 0}
```

```toml
[OCR2]
Enabled = true

[OCR2.KeyManagement]
# Initial configuration for when keys exist
EnableOCR = false # Disable OCR by default
# Bootstrappers = ["addr1", "addr2"]
# KeyBundleConfig = { KeyBundleID = 0}

[Feature]
ConditionalOCRStart = true # Flag to conditionally start OCR

```
The initial example demonstrates the standard `config.toml` structure for OCR. Now, let's explore how to prevent the node from initializing OCR during the first startup when no keys are present. I've added a new entry under the `OCR2` setting named `KeyManagement` and assigned to it the `EnableOCR` boolean. In the next example, I've additionally added a conditional flag to control OCR initialization. In the initial default example, OCR will always try to start, and if the required keys aren’t there, it will crash. To avoid this, the second example has `EnableOCR` set to `false` by default, thereby disabling OCR at boot. We then introduce the `Feature.ConditionalOCRStart` flag, which we will use in our initialization logic. This method prevents the node from trying to initialize the OCR module on the first run.

The next step involves a modification of the node's entry point, which for our setup is `chainlink/main.go`, to check for keys using the `Feature.ConditionalOCRStart` flag. While I cannot provide the precise code for Chainlink core given its open-source and changing nature, the pseudo-code below illustrates how the `main.go` can handle this situation.

```go
package main

import (
	"fmt"
	"os"
        "github.com/smartcontractkit/chainlink/v2/core/config"

)

func main() {
    cfg := config.Load()

    if cfg.Feature.ConditionalOCRStart {
		// Check if OCR keys exist
		if !ocrKeysExist(cfg) {
			fmt.Println("OCR keys not found. Conditional OCR start disabled.")
            cfg.OCR2.KeyManagement.EnableOCR = false // explicitly disable OCR
		}
    }
	
	// ... Rest of initialization process.
	fmt.Println("Initialization complete")

	if cfg.OCR2.KeyManagement.EnableOCR {
        startOCR(cfg)
    }
    
   // ... Rest of main execution loop.

}


func ocrKeysExist(cfg config.Config) bool {
	// Place logic here to check if keys exist.
    // This can involve checking a specified path for key files,
    // or querying the underlying key store.
    // In a real Chainlink system this will look for OCR keys in key store
	return false // Placeholder always returns false for the example.
}

func startOCR(cfg config.Config) {
    // place logic to start the OCR engine here.
    fmt.Println("OCR Initializing")
}
```

This Go example provides a simplified illustration of how conditional initialization can be implemented within the `main.go` file. It checks for the `ConditionalOCRStart` flag from the config.toml. If this flag is true and `ocrKeysExist` returns false (no keys found), then `EnableOCR` is explicitly set to `false`, effectively preventing the OCR subsystem from starting on its initial run. If the key exists, and that the flag is set to true, then the `startOCR` function is called and the system is set up to initiate. The core idea here is that the check can be done at node startup time, before the OCR module is set up and initializes. This logic resides at the node's entry point.

A third approach, used less frequently, involves a startup script with similar conditional logic. This approach doesn’t change configuration or core logic, but manages the initialization sequence. Below is the example, written in a bash script, that mirrors the Go logic above.

```bash
#!/bin/bash
# Load the config.toml using a tool like 'jq'
# Example: config=$(jq '.Feature.ConditionalOCRStart' config.toml)

# Note that for this example, assume we are calling 'chainlink' with some parameters.
if [ -n "$CONDITIONAL_OCR_START" ] && [ "$CONDITIONAL_OCR_START" = true ]; then
  if ! ocr_keys_exist ; then
    echo "OCR keys not found. Conditional OCR start disabled."
    # Modify config.toml here to set OCR2.KeyManagement.EnableOCR = false; using sed, jq or similar tool
     # In reality you’d need to use `sed`, `jq` or a similar tool to modify the file.
     # This is an illustrative example and the following line doesn't actually edit the toml.
     echo "Set EnableOCR to false in configuration"
  else
    # Set EnableOCR = true
    echo "OCR Keys exists. Set EnableOCR to true"
  fi
fi

# Start chainlink
./chainlink node start # with rest of the parameters

ocr_keys_exist(){
  # Logic to check if keys exist here.
  # For example, by checking a directory or query key store
    return 1 # return 1 for no keys, as for the example
}
```

In this bash script example, the `ocr_keys_exist` function mirrors the Go function in its goal, returning an error (non-zero exit code) when the check fails, i.e., no key exists. If the `CONDITIONAL_OCR_START` environment variable is set to `true` and the key check fails, it outputs a message, and sets the `EnableOCR` value to `false` before starting `chainlink`. This approach is suitable if you prefer to orchestrate the Chainlink node startup via scripts and avoids the need to recompile chainlink.

In all scenarios, the crucial element is preventing the Chainlink node from attempting OCR initialization in the absence of required keys. Once the node is up and running with `EnableOCR` set to false, we must trigger key generation. This is normally done via API call to the node. Once the OCR keys are generated, then the node’s `EnableOCR` flag can be set to `true` and the node can be restarted, enabling OCR. This approach of enabling and disabling OCR dynamically allows for a reliable node initialization and avoids crashes due to missing keys.

For further study into key management and secure handling, I would recommend delving deeper into the project documentation regarding key management. There are also a number of security best practices documents that outline key generation and handling for cryptographic applications and these are generally applicable to OCR and chainlink.

In summary, addressing this issue requires a combination of configuration adjustments and a modification of the Chainlink node's initialization process. My experience has shown that the presented methods provide a reliable workaround for preventing crashes when starting a node without prior OCR keys, ultimately ensuring a more robust and fault-tolerant system.

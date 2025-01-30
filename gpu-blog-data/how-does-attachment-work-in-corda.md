---
title: "How does attachment work in Corda?"
date: "2025-01-30"
id: "how-does-attachment-work-in-corda"
---
Corda's attachment mechanism, at its core, enables the inclusion of arbitrary binary data, such as JAR files or other resource bundles, alongside transaction data. This functionality is pivotal for deploying complex smart contracts (CorDapps) that require additional dependencies or data beyond the standard contract code. I've observed this firsthand while architecting distributed applications on Corda 4.x and 5.x, noting how the careful management of attachments directly impacts network performance and security. The primary goal of attachments is to maintain immutability of data associated with a transaction. The use of content-addressable storage (based on secure hashes) is central to the design, ensuring that any modification to the attached content is immediately detectable and renders the transaction invalid. This is crucial for preserving trust and preventing tampering within the network.

Attachments in Corda are not stored within the ledger itself, which is a crucial distinction from data held within states. Instead, they reside in the node's vault and are referenced by a hash which is recorded as part of a transaction's metadata. When a node receives a transaction containing attachments, it verifies the hashes against the stored attachments, fetching them only if they are not already present. This mechanism avoids unnecessary data replication and reduces bandwidth usage. Crucially, this also means that if an attachment is altered, the transaction becomes invalid. The system also prevents malicious entities from injecting corrupted attachments, as each attachment is validated before usage. This design is central to the integrity of Corda's distributed ledger.

The most common use case for attachments is the deployment of CorDapps. The CorDapp JAR file, containing contract code, flow logic, and other associated files, is included as an attachment to the initial transaction which registers the CorDapp with the node. Subsequent transactions using the CorDapp implicitly reference the attached code without re-transferring the JAR data. The hash-based reference ensures that all nodes are using the same, verified version of the code. Beyond CorDapp deployment, attachments can be used for providing transaction data, for example, templates or configuration files which the contract requires during execution. This offers a versatile mechanism to couple data with the specific business logic contained within a transaction.

Now, let's examine some code examples that demonstrate how attachments are used in practice.

**Example 1: Attaching a simple text file**

This example shows how to attach a plain text file to a transaction using a Corda flow. In this case, we are reading the file from the resources directory, but any `InputStream` could be used as the source of the attachment data.

```kotlin
import net.corda.core.flows.FlowLogic
import net.corda.core.flows.InitiatingFlow
import net.corda.core.utilities.getOrThrow
import java.io.InputStream
import java.nio.file.Files
import java.nio.file.Paths
import net.corda.core.transactions.TransactionBuilder
import net.corda.core.contracts.TransactionState
import net.corda.core.node.services.VaultService
import net.corda.core.contracts.ContractState
import net.corda.core.identity.Party
import net.corda.core.node.StatesToRecord
import net.corda.core.crypto.SecureHash

data class SimpleState(val value: String, override val participants: List<Party> = emptyList()) : ContractState

class SimpleContract : net.corda.core.contracts.Contract {
    companion object {
        const val ID = "com.example.SimpleContract"
    }
    override fun verify(tx: net.corda.core.transactions.LedgerTransaction) { }
}


@InitiatingFlow
class CreateAttachmentFlow(val filename: String) : FlowLogic<SecureHash>() {

    override fun call(): SecureHash {
        val inputStream: InputStream = javaClass.classLoader.getResourceAsStream(filename)
            ?: throw IllegalArgumentException("File not found: $filename")

        val attachmentId: SecureHash = serviceHub.attachments.importAttachment(inputStream)

        val notary = serviceHub.networkMapCache.notaryIdentities.first()

        val state = SimpleState("Test State", listOf(ourIdentity, notary))

        val txBuilder = TransactionBuilder(notary)
        txBuilder.addOutputState(state, SimpleContract.ID)
        txBuilder.addAttachment(attachmentId)
        txBuilder.setTimeWindow(serviceHub.clock.instant(), 100000)
        txBuilder.verify(serviceHub)

        val signedTransaction = serviceHub.signInitialTransaction(txBuilder)

        subFlow(net.corda.core.flows.FinalityFlow(signedTransaction, emptyList()))
        
        return attachmentId
    }
}
```

In this example, the `importAttachment` method is used to load the attachment and return the unique identifier (the secure hash) of the data. This hash is then added to the transaction using `addAttachment`. The transaction is built using a simple state and contract, signed, and then finalized on the ledger. This illustrates a basic procedure for incorporating arbitrary data into the transaction process via attachments. When the transaction is distributed to other nodes, they will verify the attachment hash, and if they do not already have it cached they will retrieve it using a specific messaging protocol.

**Example 2: Accessing an Attachment**

This second example details how to access and utilize an attached file within a Corda contract. In a real-world application the contract could need to parse a data file, or access some other external resource. I've found this useful when needing to create deterministic contracts, but also those with dynamic configuration.

```kotlin
import net.corda.core.contracts.Contract
import net.corda.core.contracts.ContractState
import net.corda.core.contracts.TransactionState
import net.corda.core.crypto.SecureHash
import net.corda.core.transactions.LedgerTransaction
import java.io.InputStream
import java.util.Scanner
import net.corda.core.identity.Party


data class ConfigState(val configHash: SecureHash, override val participants: List<Party> = emptyList()) : ContractState

class ConfigContract : Contract {
    companion object {
        const val ID = "com.example.ConfigContract"
    }

    override fun verify(tx: LedgerTransaction) {
        val configState = tx.outputsOfType<ConfigState>().single()

        val attachmentData: ByteArray = tx.attachments.openAttachment(configState.configHash)
            ?.readAllBytes()
            ?: throw IllegalArgumentException("Attachment not found: ${configState.configHash}")

        val contents: String = Scanner(attachmentData.inputStream()).useDelimiter("\\A").next()

        if(!contents.contains("Valid Configuration")){
            throw IllegalArgumentException("Configuration Failed")
        }
    }
}

fun LedgerTransaction.attachments() : net.corda.core.contracts.AttachmentStorage{
    return this.networkParameters.attachmentStorage
}
```

Here, the contract is retrieving a `ConfigState` from the transaction outputs, which has a hash referencing the configuration attachment. The contract accesses the attachment via the `openAttachment` method, reading the byte array, and converting it to a string using a `Scanner`.  A simple check is performed validating a string within the content, and the transaction is deemed valid only if the specific configuration data is found. The use of the `use` statement will close the `Scanner` automatically. This demonstrates how contract code can access, parse and make decisions based on attached data, maintaining a crucial connection between transaction data and the external context. Note that access to the attachment service requires getting the `AttachmentStorage` service from the `NetworkParameters` inside of a transaction.

**Example 3:  Sharing Attachments**

In this example I'm showing that attachments are distributed by hash, but the transaction will need to be available before a node can access the attachment. If a node already has the attachment it will not receive it again, saving on network traffic.

```kotlin
import net.corda.core.flows.FlowLogic
import net.corda.core.flows.InitiatingFlow
import net.corda.core.utilities.getOrThrow
import java.io.InputStream
import java.nio.file.Files
import java.nio.file.Paths
import net.corda.core.transactions.TransactionBuilder
import net.corda.core.contracts.TransactionState
import net.corda.core.node.services.VaultService
import net.corda.core.contracts.ContractState
import net.corda.core.identity.Party
import net.corda.core.node.StatesToRecord
import net.corda.core.crypto.SecureHash
import net.corda.core.flows.ReceiveFinalityFlow
import net.corda.core.flows.SendTransactionFlow

data class AttachmentState(val attachmentHash: SecureHash, override val participants: List<Party> = emptyList()) : ContractState

class AttachmentContract : net.corda.core.contracts.Contract {
    companion object {
        const val ID = "com.example.AttachmentContract"
    }
    override fun verify(tx: net.corda.core.transactions.LedgerTransaction) { }
}


@InitiatingFlow
class AttachmentInitiatorFlow(val filename: String, val otherParty: Party) : FlowLogic<SecureHash>() {

    override fun call(): SecureHash {
        val inputStream: InputStream = javaClass.classLoader.getResourceAsStream(filename)
            ?: throw IllegalArgumentException("File not found: $filename")

        val attachmentId: SecureHash = serviceHub.attachments.importAttachment(inputStream)

        val notary = serviceHub.networkMapCache.notaryIdentities.first()

        val state = AttachmentState(attachmentId, listOf(ourIdentity, otherParty))

        val txBuilder = TransactionBuilder(notary)
        txBuilder.addOutputState(state, AttachmentContract.ID)
        txBuilder.addAttachment(attachmentId)
        txBuilder.setTimeWindow(serviceHub.clock.instant(), 100000)
        txBuilder.verify(serviceHub)

        val signedTransaction = serviceHub.signInitialTransaction(txBuilder)

        val session = initiateFlow(otherParty)

        subFlow(SendTransactionFlow(session, signedTransaction))
        subFlow(net.corda.core.flows.FinalityFlow(signedTransaction, emptyList(), statesToRecord = StatesToRecord.ALL_VISIBLE))
        
        return attachmentId
    }
}


class AttachmentResponderFlow(val otherParty: Party) : FlowLogic<Unit>(){
     override fun call() {
        val session = initiateFlow(otherParty)
        val tx = subFlow(ReceiveFinalityFlow(session))

        //At this point the responder will have access to the attachment if they don't already
        val txFromVault = serviceHub.validatedTransactions.getTransaction(tx.id) ?: throw Exception("Transaction not found")
        val output = txFromVault.tx.outputsOfType(AttachmentState::class.java).single()

        val attachmentData: ByteArray = txFromVault.tx.attachments.openAttachment(output.attachmentHash)
            ?.readAllBytes()
            ?: throw IllegalArgumentException("Attachment not found: ${output.attachmentHash}")

        val contents: String = String(attachmentData)

        println("Received attachment: ${contents.subSequence(0, 100)}")

     }
}
```

In this example, `AttachmentInitiatorFlow` imports an attachment and creates a transaction. It then uses `SendTransactionFlow` to send the transaction to another node which executes the responder flow `AttachmentResponderFlow`. The responder receives the transaction using `ReceiveFinalityFlow`, and then retrieves the transaction from the vault to extract the attachment hash. It then calls `openAttachment` to load the data, demonstrating how attachments are shared between nodes. This code illustrates an end-to-end flow showing the distributed nature of the attachment process. If the receiving node already had the attachment by hash, the file will not be sent by the sending node, however, the transaction itself is necessary for the receiving node to discover the correct hash of the attachment.

Regarding resources, I would recommend focusing on the official Corda documentation, which details the specific functions within the `AttachmentStorage` service, such as `importAttachment`, `openAttachment`, and associated data structures. The CorDapp documentation is particularly useful when exploring common attachment use cases such as contract deployment. Additionally, reviewing the Corda training materials offers detailed walkthroughs of example use cases with best practices, especially with the latest versions of the platform. Corda samples, found on Github, are an invaluable resource for exploring specific attachment scenarios such as loading large amounts of data into an attachment. These samples provide executable code that showcases best practices, and patterns for integrating attachments in real-world application. Careful study of these resources is crucial for understanding and effectively implementing attachments in Corda.

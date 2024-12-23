---
title: "Can a Corda application's initiator and responder flows be executed on separate nodes?"
date: "2024-12-23"
id: "can-a-corda-applications-initiator-and-responder-flows-be-executed-on-separate-nodes"
---

Alright, let's unpack this interesting question about Corda flows and their execution across nodes. From my past experiences building distributed ledger applications, I’ve seen firsthand how crucial understanding flow execution patterns is. It's not a straightforward "yes" or "no" – it's more nuanced. To be clear, a Corda application, or 'CorDapp,' absolutely can have its initiator and responder flows executed on separate nodes, and in fact, that’s often the intended and more common deployment configuration. The whole point of Corda is distributed consensus across a network.

However, let's dive into the mechanisms that enable this, and the constraints we face, rather than just treating it as an assumed capability. The core idea is that initiator flows, typically started by a client interacting with one node, set the business logic in motion. Then, via messaging on the Corda network, responder flows are triggered on one or more peer nodes.

It's not just a passive listener-responder setup, though. It's very intentional and transaction-oriented. The initiator doesn't just send instructions. It proposes a transaction with all the required data and potentially, custom logic, which then must be validated and agreed upon by the responder(s) before it's committed to the ledger. The key here is that both the initiator and the responder flows are operating under specific security rules enforced by the Corda platform, especially when it comes to ledger updates. They're not just independently doing their own thing, they're engaged in a coordinated effort to record a transaction.

A crucial part of this is the `InitiatingFlow` and `@InitiatedBy` annotations in your CorDapp code. The former marks the class that begins a workflow (initiator), and the latter specifies the corresponding flow class that responds when invoked by the initiating flow on a peer node. This pairing allows Corda to establish the connection between the two sides of the process. It's worth also emphasizing that the responder flow usually has access to a `FlowSession` which allows it to communicate back with the initiator.

Let's look at a basic example. Imagine a scenario where one party (Node A) wants to issue an asset to another party (Node B).

```java
// Initiator Flow on Node A
@InitiatingFlow
@StartableByRPC
public class IssueAssetFlow extends FlowLogic<SignedTransaction> {

    private final Party recipient;
    private final Amount<Currency> amount;

    public IssueAssetFlow(Party recipient, Amount<Currency> amount) {
        this.recipient = recipient;
        this.amount = amount;
    }

    @Suspendable
    @Override
    public SignedTransaction call() throws FlowException {
        // 1. Create the state
        Asset asset = new Asset(getOurIdentity(), recipient, amount);

        // 2. Create the command
        Command<AssetContract.Commands.Issue> command = new Command<>(new AssetContract.Commands.Issue(), getOurIdentity().getOwningKey());

        // 3. Build transaction
        TransactionBuilder txBuilder = new TransactionBuilder(getNotary());
        txBuilder.addOutputState(asset, AssetContract.ID);
        txBuilder.addCommand(command);

        // 4. Sign the transaction locally
        SignedTransaction partSignedTx = getServiceHub().signInitialTransaction(txBuilder);

        // 5. Open a session with the recipient
        FlowSession recipientSession = initiateFlow(recipient);

        // 6. Send the partially signed transaction to the recipient
        SignedTransaction fullySignedTx = subFlow(new CollectSignaturesFlow(partSignedTx, Arrays.asList(recipientSession)));

         // 7. Finalize and record transaction
        return subFlow(new FinalityFlow(fullySignedTx, recipientSession));
    }
}
```

Now, let's look at the corresponding responder flow running on Node B:

```java
// Responder Flow on Node B
@InitiatedBy(IssueAssetFlow.class)
public class IssueAssetResponderFlow extends FlowLogic<SignedTransaction> {

    private final FlowSession counterpartySession;

    public IssueAssetResponderFlow(FlowSession counterpartySession) {
        this.counterpartySession = counterpartySession;
    }


    @Suspendable
    @Override
    public SignedTransaction call() throws FlowException {
        // 1. Receive partially signed transaction
        SignedTransaction fullySignedTx = subFlow(new SignTransactionFlow(counterpartySession) {
            @Override
            protected void checkTransaction(SignedTransaction stx) throws FlowException {
                // Implement custom transaction checks for the responder
            }
        });

        // 2. Finalize and record transaction
        return subFlow(new ReceiveFinalityFlow(counterpartySession, fullySignedTx.getId()));
    }
}
```

Notice the separation: `IssueAssetFlow` resides on the initiator node, and `IssueAssetResponderFlow` is on the recipient. The connection is facilitated by the `InitiatedBy` annotation and `FlowSession` object, allowing secure two-way communication and the collaborative construction of a valid, agreed-upon transaction.

Another point to consider is the use of `CollectSignaturesFlow`. In the example, the initiator doesn't just "tell" the responder to sign the transaction. Rather, the flow mechanics establish a clear choreography where the initiator first prepares the transaction, and then initiates the signing process on the recipient's node. The `CollectSignaturesFlow` further emphasizes that this process is a collaborative one, involving multiple parties signing and thus approving the transaction. The same holds true for the `FinalityFlow`. It is called at both nodes to record the transaction locally.

A slightly different case involves needing a response before completing the initial flow. This happens frequently in workflows where certain data depends on the responder. Let's say Node A wants to request pricing from Node B for a future trade.

```java
// Initiator Flow Node A
@InitiatingFlow
@StartableByRPC
public class RequestPriceFlow extends FlowLogic<Price> {

    private final Party counterparty;
    private final String product;

    public RequestPriceFlow(Party counterparty, String product) {
        this.counterparty = counterparty;
        this.product = product;
    }


    @Suspendable
    @Override
    public Price call() throws FlowException {
        // 1. Open a session with the recipient
        FlowSession session = initiateFlow(counterparty);

        // 2. Send request data to recipient
        session.send(product);

        // 3. Receive price and return it.
        return session.receive(Price.class).unwrap(data -> data);

    }
}

```

Here’s the responder:

```java
// Responder Flow Node B
@InitiatedBy(RequestPriceFlow.class)
public class RespondPriceFlow extends FlowLogic<Void> {

    private final FlowSession counterpartySession;

    public RespondPriceFlow(FlowSession counterpartySession) {
        this.counterpartySession = counterpartySession;
    }

    @Suspendable
    @Override
    public Void call() throws FlowException {

        // 1. Receive product details
       String product = counterpartySession.receive(String.class).unwrap(data -> data);


        // 2. Calculate the price based on a local logic
        Price price =  calculatePrice(product);


        // 3. Send the price back to initiator
        counterpartySession.send(price);

       return null;
    }


    private Price calculatePrice(String product){
        // some logic
        return new Price(new BigDecimal(100));
    }

}
```

In this example the initiator (`RequestPriceFlow`) sends the product identifier and then *waits* for the price. The responder (`RespondPriceFlow`) is on a different node, computes the price, and sends the result back. This bi-directional communication is perfectly valid in Corda, as the `FlowSession` facilitates the exchange of data between the two flows running on separate nodes.

This pattern is essential in distributed ledger systems where the data needed for a particular action might exist on another node.

To further solidify understanding, I'd strongly recommend diving into the Corda documentation (specifically, the section on flows) and the samples provided by R3. For a more academic perspective, the paper "Corda: A Distributed Ledger Platform for Financial Institutions" can provide insight into the design rationale behind the flow framework. Another good reference is the "Mastering Corda" book by R3, which delves deeper into these patterns and common development practices.

In conclusion, the capability to execute initiator and responder flows on different nodes is not just possible but fundamental to the entire Corda architecture. It is key to achieving a decentralized, collaborative system. It's not merely an incidental feature, but rather a carefully designed workflow that facilitates secure, verifiable, and distributed transaction processing. Understanding the specifics around annotations, flow sessions, and transactional flows is essential when developing robust CorDapps.

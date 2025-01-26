---
title: "How to set a memo for each output in a MsgMultiSend transaction on Terra?"
date: "2025-01-26"
id: "how-to-set-a-memo-for-each-output-in-a-msgmultisend-transaction-on-terra"
---

The fundamental challenge in annotating individual outputs within a `MsgMultiSend` transaction on the Terra blockchain stems from the design of the message itself. Unlike single transfer messages (`MsgSend`), `MsgMultiSend` consolidates multiple transfer operations into a single message, where the memo field applies to the entire transaction, not to specific outputs. Therefore, there is no native method provided by the Terra SDK to directly attach a unique memo to each individual output within a `MsgMultiSend`. My experience working on decentralized finance applications using Terra has highlighted the need for workarounds to achieve this kind of output-specific annotation.

The primary constraint is the structure of `MsgMultiSend`. It accepts an array of input and output objects, each containing an address and an amount of coins. However, it contains a single memo field, designed to apply to the entire transaction, not its components. Consequently, directly setting a unique memo for every output is not possible by simply modifying the fields of `MsgMultiSend`. Instead, the solution involves leveraging the existing memo field in a way that encodes the output-specific memo information. This encoding process requires parsing and interpretation at the receiver end, making this a convention rather than a hard-coded feature of the transaction.

The method I found most effective involves embedding the output-specific memo data within the primary transaction memo using a custom, structured format. My strategy has evolved from simplistic delimiters to a JSON-based approach for robustness and scalability. The memo will contain an ordered array of memo objects, corresponding to the ordered output array in the `MsgMultiSend` message. Each memo object would contain the necessary data for the corresponding output. While JSON can add overhead due to its verbosity, the increased clarity and maintainability offset that concern, particularly when parsing these memos in applications. This methodology enables associating custom context for each receiver.

Here's a demonstration of this strategy, beginning with the construction of the `MsgMultiSend` and the encoded memo:

```python
import json
from terra_sdk.client.lcd import LCDClient
from terra_sdk.key.mnemonic import MnemonicKey
from terra_sdk.core import Coins
from terra_sdk.core.bank import MsgMultiSend, Input, Output

# Setup for simplicity, use your own private key and LCD Client configuration.
terra = LCDClient(url="https://bombay-lcd.terra.dev", chain_id="bombay-12")
mnemonic = "your_mnemonic_here"  #Replace with your mnemonic
key = MnemonicKey(mnemonic=mnemonic)
wallet = terra.wallet(key)

# Defining recipients and their specific memos
recipients_info = [
    {"address": "terra1recipient1address", "amount": 1000000, "memo": "Payment for service A"},
    {"address": "terra1recipient2address", "amount": 2000000, "memo": "Reimbursement for travel"},
    {"address": "terra1recipient3address", "amount": 3000000, "memo": "Partial payment for product B"},
]

inputs = [Input(address=wallet.key.acc_address, coins=Coins.from_str("6000000uluna"))] #adjust input amount
outputs = []
memo_objects = []

for recipient in recipients_info:
    outputs.append(Output(address=recipient['address'], coins=Coins.from_str(f"{recipient['amount']}uluna")))
    memo_objects.append({"address": recipient["address"], "memo": recipient['memo']})

# Encode the custom memos into JSON string
encoded_memo = json.dumps(memo_objects)

# Create the message with encoded memos
msg = MsgMultiSend(inputs=inputs, outputs=outputs, memo=encoded_memo)

# The subsequent transaction creation and broadcasting are omitted for brevity.
# In actual implementation, the following steps would be followed:
# 1. create transaction body.
# 2. create transaction and sign it with private key.
# 3. broadcast it using LCD client.
print(f"Encoded Memo: {encoded_memo}")
print(f"MsgMultiSend message with encoded memos generated.")
```

In this example, the `recipients_info` list holds the data for each output, including the recipient address, amount, and a specific memo. We iterate through this list to construct both the output objects for the `MsgMultiSend` and the JSON-serializable `memo_objects` which will become the encoded memo. By generating the `encoded_memo` as a string representation of the `memo_objects` JSON object, it can be placed into the `MsgMultiSend` message.

The decoding process requires reconstructing the individual output memos from the transaction's memo. This is achieved by parsing the JSON string and then processing it according to the output addresses.

```python
import json

def decode_memos(transaction_memo, transaction_outputs):
    """
    Decodes the custom memo from a MsgMultiSend transaction.

    Args:
        transaction_memo (str): The memo string from the transaction.
        transaction_outputs (list): A list of Output objects from the transaction.

    Returns:
        dict: A dictionary mapping output addresses to their specific memos,
              or an empty dict if decoding fails or if no specific memos were present.
    """
    if not transaction_memo:
         return {}

    try:
       decoded_memo_objects = json.loads(transaction_memo)
       output_memos = {}

       for output in transaction_outputs:
         for memo_obj in decoded_memo_objects:
            if output.address == memo_obj['address']:
                output_memos[output.address] = memo_obj['memo']
                break

       return output_memos

    except (json.JSONDecodeError, KeyError) :
       # Handle cases where the memo is not a valid JSON, or does not contain address and memo keys.
       return {}

# Example usage of this function
# Assume 'transaction_memo' and 'transaction_outputs' were retrieved from a transaction
# Replace these placeholders with your actual values.
transaction_memo = '{"memo": [{"address": "terra1recipient1address", "memo": "Payment for service A"}, {"address": "terra1recipient2address", "memo": "Reimbursement for travel"}, {"address": "terra1recipient3address", "memo": "Partial payment for product B"}]}'
transaction_outputs = [
    Output(address="terra1recipient1address", coins=Coins.from_str("1000000uluna")),
    Output(address="terra1recipient2address", coins=Coins.from_str("2000000uluna")),
    Output(address="terra1recipient3address", coins=Coins.from_str("3000000uluna")),
    ]


output_specific_memos = decode_memos(transaction_memo, transaction_outputs)

if output_specific_memos:
  for address, memo in output_specific_memos.items():
    print(f"Address: {address}, Memo: {memo}")
else:
  print("No specific output memos found.")

```

This `decode_memos` function parses the JSON-encoded memo, iterates through the transaction outputs, and then extracts the corresponding memos based on the recipient address. I have found that having error handling in place when dealing with potentially malformed memos is critical for creating a robust implementation.

Finally, I want to illustrate a scenario where the memo might also include metadata related to the outputs, for instance, the sender's internal identifiers:

```python
import json
from terra_sdk.core import Coins
from terra_sdk.core.bank import MsgMultiSend, Input, Output

sender_address = "terra1senderaddress" # replace with your sender address

recipients_data = [
    {
        "address": "terra1recipient1address",
        "amount": 1000000,
        "memo": "Payment for service A",
        "sender_reference": "REF-001"
    },
    {
        "address": "terra1recipient2address",
        "amount": 2000000,
        "memo": "Reimbursement for travel",
        "sender_reference": "REF-002"
    },
    {
        "address": "terra1recipient3address",
        "amount": 3000000,
        "memo": "Partial payment for product B",
        "sender_reference": "REF-003"
    },
]

inputs = [Input(address=sender_address, coins=Coins.from_str("6000000uluna"))]
outputs = []
memo_objects = []

for recipient in recipients_data:
    outputs.append(Output(address=recipient["address"], coins=Coins.from_str(f"{recipient['amount']}uluna")))
    memo_objects.append({
        "address": recipient["address"],
        "memo": recipient["memo"],
        "sender_reference": recipient["sender_reference"]
    })

encoded_memo = json.dumps(memo_objects)
msg = MsgMultiSend(inputs=inputs, outputs=outputs, memo=encoded_memo)

print(f"Encoded Memo with sender reference: {encoded_memo}")
print(f"MsgMultiSend message with enhanced encoded memos generated.")
```

This expanded example shows how you can incorporate additional data elements alongside the basic memo string, in this case adding the `sender_reference`. This pattern provides flexibility to include any necessary metadata, as long as both the sending and receiving applications are aware of the memo’s structure and how to parse it.

While the described convention does add complexity, I've found it to be a pragmatic and functional way to overcome the inherent limitation of the `MsgMultiSend` message for more complex use cases involving nuanced transaction annotations on the Terra blockchain. For further exploration of transaction handling, I would recommend reviewing the Terra SDK documentation, official tutorials, and resources focusing on transaction construction and data parsing. Additionally, discussions within the Terra developer community can also provide helpful insights and alternative approaches. These resources, combined with a hands-on approach, provide the best way to master the intricacies of the Terra ecosystem.

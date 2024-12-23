---
title: "alternating bit protocol communication network?"
date: "2024-12-13"
id: "alternating-bit-protocol-communication-network"
---

 so you're asking about the alternating bit protocol yeah I've been down that road more times than I care to admit its like the first thing they throw at you when you get into network stuff and it seems simple but oh boy it can bite you real bad

So let's break it down you've got two entities Sender and Receiver and they're trying to have a reliable data transfer over an unreliable channel the channel is the key its like that old wifi router at your grandmas that randomly drops connection every five minutes or worse its like a network cable that your cat has chewed in 3 spots but lets go back to the protocol itself we need to make sure every packet gets there and we need to know if its the correct data packet right

The basic idea is that each data packet includes a single bit a sequence number or flag or whatever you want to call it and this bit alternates between 0 and 1 for each new packet sent. When the receiver gets a packet it acknowledges it with an ACK also carrying the expected sequence number then if that sequence number is what it expects all is good next packet etc if not something is wrong that packet got corrupted or its a duplicate lets see how it works in practice

For example here is pseudo code example for the sender end its as basic as it gets

```python
def sender(data_packets):
    seq_num = 0
    for packet in data_packets:
        while True:
            send_packet(packet, seq_num)
            ack = receive_ack() # Blocking operation wait for ACK
            if ack and ack.seq_num == seq_num:
               seq_num = 1 - seq_num  # Toggle seq number
               break # go next
            else:
                # Handle timeout or NACK retransmit
                pass
```

Now for the receiver side this is a bit more involved in order to see if the message is indeed what we expect it to be

```python
def receiver():
    expected_seq_num = 0
    while True:
        packet = receive_packet()
        if packet and packet.seq_num == expected_seq_num:
            send_ack(packet.seq_num)
            process_packet(packet.data)
            expected_seq_num = 1 - expected_seq_num # toggle for next
        else:
            send_ack(1-expected_seq_num) # NACK the message
            #Discard packet since it is invalid or duplicate

```

It seems like a really simple protocol right And yeah it is really simple but simple doesnt mean its not useful you will find it as the building blocks of a lot more complex protocols that do fancy things under the hood like TCP or even some custom protocols I had to deal with at some point when I was messing with embedded systems there I had very limited resources so I have had to come up with something similar and thats how I got to know this protocol like a good old friend

Now lets dive into a few scenarios where this protocol could fail and how to fix it or what are the constraints

One of the most common issues is packet loss If a data packet or an acknowledgment packet is lost in transit the sender might have to retransmit that message or worse it can end up in a deadlock it goes like this sender sends data packet with sequence number 0 it gets lost the sender never receives the ack so it keeps sending the same message over and over again and receiver never receives the first one to begin with and its just a loop of nothing no processing no progress. In order to deal with that we need to add timeouts if the sender does not receive an ACK in a certain timeframe it will assume that the packet was lost and it will send it again. In most cases the packet is just lost but what if a message arrives later? well then we need to have a sequence number to recognize what packet we are dealing with so we would simply discard that packet since its a duplicate

Another scenario is packet corruption due to errors in transmission like random bit flips. For that we need to add a checksum calculation to the message so that if the receiver can detect that a message is corrupted we simply discard it or nack it. This process however could be more involved computationally speaking so sometimes it might not be worth it

What about duplicate ACKs? yes thats a thing the receiver might send an ACK and it gets duplicated in transit the sender thinks ok I received an ACK so I can go next. But the receiver might be still waiting for this sequence number we need to deal with it by making sure that the sender just sends the next message if it receives a valid ACK with the expected sequence number otherwise it just re-sends the current one and we can add a timeout mechanism for this as well

Now some practical considerations.

This is a stop-and-wait protocol which means the sender sends a packet and waits for its acknowledgment before sending the next one. This is not efficient for high-bandwidth networks it's like ordering one item at a time in a restaurant when you have a whole group of people waiting to order. However if the messages are small this is pretty good since the implementation is trivial. For higher bandwidth you need to start considering the sliding window protocol which uses similar mechanisms but allows multiple packets to be in flight at the same time

The alternating bit protocol assumes that communication is bidirectional so the sender needs a way to listen for acknowledgments from the receiver. This is not a problem if you have a channel where you can just send and receive messages. However if the connection is unidirectional you will need a different method to handle the ACK messages like a separate data channel and that's when things start getting complex

Now regarding resources you might be interested in the good old books like "Computer Networks" by Andrew S Tanenbaum and David J. Wetherall or "Data and Computer Communications" by William Stallings. Those guys are the real deal their books are more than just theory you will find practical applications as well. Also look up RFC 791 for some insights about IP it also deals with some packet issues its a pretty good read if you want to dig in. Do not try to learn from online tutorials those can be useful but they will only get you so far most of the time they are incomplete or wrong. And lets be honest most of them are just copy pasted from a random blog anyway

I have a funny story about this protocol. I had this project back in uni you know that one where you had to build a network simulator I was messing around with the bit alternating implementation and it took me like two days to figure out why the whole system was just stuck. It turns out the the receiver was expecting sequence number 1 from the get go instead of zero thats right one of the simplest things that can happen to you, right at the start and I spend two days on that. Sometimes the smallest things bite you the hardest it is like looking for your car keys when you are already holding them (yeah I know its not very funny)

So yeah the alternating bit protocol is a nice entry point to the world of networking if you want to get started with the basics. It is simple but not perfect it can fail for all kinds of reasons but you can overcome them with simple logic and careful implementation and of course testing

Here's a final snippet showcasing a basic implementation in Go a language I sometimes use if I have to deal with multi-threaded code

```go
package main

import (
	"fmt"
	"time"
)

type Packet struct {
	SeqNum int
	Data    string
}

type ACK struct {
	SeqNum int
}

func sendPacket(data string, seqNum int, ch chan Packet) {
	packet := Packet{SeqNum: seqNum, Data: data}
    time.Sleep(100 * time.Millisecond)
    fmt.Println("Sender sending ",packet)
	ch <- packet
}

func receivePacket(ch chan Packet) Packet {
	packet := <-ch
    fmt.Println("Receiver received ",packet)
	return packet
}


func sendAck(seqNum int, ch chan ACK) {
    time.Sleep(100 * time.Millisecond)
    fmt.Println("Receiver sending ack ",ACK{SeqNum: seqNum})
	ch <- ACK{SeqNum: seqNum}
}

func receiveAck(ch chan ACK) ACK {
	ack := <-ch
    fmt.Println("Sender received ack ",ack)
	return ack
}

func main() {
	data := []string{"hello", "world", "from", "go"}
	dataChan := make(chan Packet)
	ackChan := make(chan ACK)
	seq := 0

	go func() { // Receiver
		expectedSeq := 0
		for {
			packet := receivePacket(dataChan)
			if packet.SeqNum == expectedSeq {
                fmt.Println("Receiver processing ",packet)
				sendAck(packet.SeqNum, ackChan)
				expectedSeq = 1 - expectedSeq
			}else{
				sendAck(1-expectedSeq, ackChan)
			}
		}
	}()

	for _, msg := range data { // Sender
		for {
			sendPacket(msg, seq, dataChan)
			ack := receiveAck(ackChan)
			if ack.SeqNum == seq {
				seq = 1 - seq
				break
			}
		}
	}

	fmt.Println("All messages sent successfully!")

}
```

Its basic but it does the job It will print all the messages and ACKs and show the progress of the messages being delivered This program assumes that no packet is lost.

So yeah thats pretty much it the alternating bit protocol its simple its not perfect but it's a cornerstone of many more complex systems out there if you have any further questions just let me know

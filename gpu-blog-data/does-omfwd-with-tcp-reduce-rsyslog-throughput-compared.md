---
title: "Does omfwd with TCP reduce rsyslog throughput compared to UDP?"
date: "2025-01-30"
id: "does-omfwd-with-tcp-reduce-rsyslog-throughput-compared"
---
Rsyslog's `omfwd` module, specifically when configured with TCP, introduces performance characteristics distinct from its UDP counterpart, primarily affecting throughput due to TCP's inherent connection management and reliability mechanisms. My prior experience optimizing high-volume logging pipelines reveals that while TCP offers guaranteed delivery and connection persistence, these features come at the cost of reduced raw throughput compared to UDP. UDP, in contrast, prioritizes speed, trading delivery guarantees for increased message processing rate. The core difference stems from TCP's requirement for connection establishment (handshake), error recovery (retransmissions), and connection termination (teardown) protocols, all of which add overhead.

A key factor impacting `omfwd` with TCP is the establishment of a persistent connection. For every new destination, a three-way handshake (SYN, SYN-ACK, ACK) is required. While generally efficient, in scenarios with rapid log source churn or a large number of logging endpoints, this repeated handshake process can contribute noticeably to overall performance degradation. Conversely, UDP is connectionless; log messages are simply transmitted without any prior setup, thus eliminating this overhead. The absence of guaranteed delivery in UDP means that some messages might be lost due to network congestion or other transient issues, but for applications where occasional data loss is tolerable, the speed advantage is often substantial.

Another significant performance bottleneck specific to TCP is congestion control. TCP's congestion control mechanisms (e.g., slow start, congestion avoidance) dynamically adjust transmission rates to prevent network overloads. These mechanisms can dramatically slow down throughput when network congestion is detected. Even without visible congestion, the initial slow start phase limits throughput, particularly in scenarios that require transmitting high volumes of log data quickly. These limitations are not present in the relatively "fire and forget" nature of UDP. While TCP's congestion control is crucial for general network stability, it imposes an overhead that impacts `omfwd`'s performance. Moreover, TCP's sequence numbers, acknowledgments, and retransmissions all require processing resources at both the sending and receiving ends, further decreasing the effective throughput achievable compared to UDP's simpler, connectionless approach.

The handling of errors also contributes to the throughput differential. TCP ensures message delivery via acknowledgements and retransmissions. If a TCP segment is lost, the sender will retransmit it, waiting for the acknowledgement. This retransmission process introduces delays and further reduces the effective data rate. While vital for ensuring data integrity, this reliability process is entirely absent in UDP. In a UDP configuration, messages are dispatched without confirmation, allowing for more data to be transmitted within the same timeframe, even with potential packet loss. If the receiving application or system cannot process the incoming stream, TCP backpressure mechanisms (e.g., reduced window size) cause the sender to slow transmission, something completely absent with UDP. Finally, TCP's connection termination process, which includes the four-way handshake, also contributes to overhead, albeit only at the end of a connection. When connections are frequently recycled (e.g., due to log rotation), these connection teardowns become an additional cost.

Here are some concrete examples and considerations:

**Example 1: Basic TCP Output**

```rsyslog
# rsyslog configuration snippet
$ModLoad imuxsock
$template TraditionalFormat,"%timegenerated% %HOSTNAME% %syslogfacility-text% %syslogseverity-text% %msg%\n"

*.*   @@192.168.1.100:514;TraditionalFormat
```

This simple example demonstrates sending all logs to a remote host at `192.168.1.100` on port 514 using TCP. The double at sign (`@@`) indicates TCP. Every log message sent via TCP to the remote server will undergo the full TCP protocol overhead: connection management, error checking, and potentially congestion avoidance. While generally reliable, this approach will achieve lower throughput than an equivalent UDP setup. This can be further complicated by the load on the target server processing this TCP data stream.

**Example 2: Basic UDP Output**

```rsyslog
# rsyslog configuration snippet
$ModLoad imuxsock
$template TraditionalFormat,"%timegenerated% %HOSTNAME% %syslogfacility-text% %syslogseverity-text% %msg%\n"

*.*   @192.168.1.100:514;TraditionalFormat
```

This example sends the same log messages to the same destination (IP and port), but this time using UDP. The single at sign (`@`) denotes UDP. Log messages will be sent without any connection establishment or delivery guarantees. This means higher overall throughput as it avoids TCP overhead, but at the expense of potential message loss. For systems where some data loss is acceptable, particularly when processing very high volumes of log data, this approach can be advantageous. Careful monitoring of lost message counters or employing a logging aggregation/analysis system that handles missing log entries is important.

**Example 3: TCP Output with Connection Re-use**

```rsyslog
# rsyslog configuration snippet
$ModLoad imuxsock
$template TraditionalFormat,"%timegenerated% %HOSTNAME% %syslogfacility-text% %syslogseverity-text% %msg%\n"

$ActionSendTCPConnectionReuseInterval 3600 # Re-use connections for 1 hour

*.*   @@192.168.1.100:514;TraditionalFormat
```

This third example shows a TCP setup with connection re-use configuration, where the underlying socket connection to the destination will be kept open (and reused for new logs) for an hour (3600 seconds). This approach is designed to mitigate the connection overhead associated with frequent reconnections. This configuration can improve performance compared to constantly establishing new connections, especially in a scenario where log rotations happen every few minutes, but it still does not solve congestion or reliability mechanism impact. If log source churn is too high, this approach will not avoid repeated connection establishment, and might not increase throughput significantly.

To elaborate further on resource recommendations, I advise reviewing the official rsyslog documentation for `omfwd` module. Particular attention should be given to the sections detailing TCP and UDP output formats, and their associated configuration parameters. Additionally, consult general networking texts that explain the differences between TCP and UDP at a protocol level. This understanding will provide context for the performance differences witnessed with `omfwd`. Network monitoring tools such as `tcpdump` and `iftop` can also provide invaluable real-time insights into the behavior of the underlying network protocols, thus allowing for data-driven optimization decisions. Lastly, reviewing best practices documentation on rsyslog tuning from various system administrators and cybersecurity resources is useful to avoid common configuration errors. Understanding the implications of each option becomes essential to achieving optimal performance based on specific business requirements and available infrastructure resources.

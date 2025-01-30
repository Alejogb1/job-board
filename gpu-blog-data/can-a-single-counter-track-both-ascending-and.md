---
title: "Can a single counter track both ascending and descending pulse streams?"
date: "2025-01-30"
id: "can-a-single-counter-track-both-ascending-and"
---
The inherent ambiguity of a "pulse stream" necessitates a precise definition before addressing the core question.  In my experience designing embedded systems for high-speed data acquisition, a "pulse stream" most accurately refers to a series of discrete, time-separated events, typically represented as transitions in a digital signal.  Therefore, determining whether a single counter can track both ascending and descending pulse streams hinges on how these streams are defined and presented to the counter.  A naïve approach will fail; a sophisticated approach, however, is achievable through careful signal conditioning and counter configuration.

The key lies in differentiating the ascending and descending edges. A standard counter increments solely on a rising edge. Tracking both requires either pre-processing the input signal or employing a counter with edge-select functionality.  Failing to distinguish the pulse direction will inevitably lead to erroneous counts, as the counter will simply accumulate all transitions, regardless of their direction.  This directly impacts the accuracy of the measurement, rendering the counter unsuitable for applications demanding precise bidirectional pulse counting.

**1.  Explanation: Differentiating Ascending and Descending Pulses**

To achieve accurate bidirectional counting, the input pulse stream must be conditioned to present distinct signals for ascending and descending transitions.  This can be accomplished using a combination of hardware and software techniques.  Hardware solutions often involve differentiating circuits, producing separate output signals for positive and negative-going edges.  Alternatively, software-based solutions utilize interrupt service routines (ISRs) triggered by edge-sensitive inputs.  The ISR then identifies the edge type and increments the appropriate counter variable.  In scenarios with limited interrupt capabilities, a state machine can effectively manage the detection of ascending and descending edges.  The implementation choice heavily depends on factors such as the required speed, hardware constraints, and the complexity of the microcontroller.  My work on the Helios Project, a high-precision satellite attitude control system, leveraged a custom hardware differentiator coupled with a dedicated counter module to achieve millisecond precision in bidirectional pulse counting from multiple gyro sensors.

**2. Code Examples and Commentary**

The following code examples illustrate different approaches to achieving bidirectional pulse counting.  Note that these are simplified illustrations and will need modification depending on the microcontroller architecture and specific peripherals used.

**Example 1:  Software-based Edge Detection with Two Counters (C)**

This approach utilizes two separate counters, one for ascending edges and one for descending edges.  The code relies on interrupt handling for edge detection.

```c
volatile unsigned long ascending_count = 0;
volatile unsigned long descending_count = 0;

// Interrupt Service Routine for rising edge
void ISR_RisingEdge(void) {
  ascending_count++;
}

// Interrupt Service Routine for falling edge
void ISR_FallingEdge(void) {
  descending_count++;
}

int main(void) {
  // Configure interrupt pins for rising and falling edge detection
  // ...

  // Enable interrupts
  // ...

  while(1) {
    // Main loop processing
    // ...
  }
  return 0;
}
```

**Commentary:** This method is straightforward but requires two counters and dedicated interrupt pins. It's best suited for environments where sufficient interrupt resources are available and precise individual counts are needed. It avoids the complexity of state machines and minimizes potential race conditions.


**Example 2:  State Machine Approach (C++)**

This example uses a state machine to track both edge types within a single counter.  The current state dictates the action taken upon detecting an edge.

```c++
enum State {IDLE, ASCENDING, DESCENDING};
State currentState = IDLE;
unsigned long totalCount = 0;

void processPulse(bool isRisingEdge) {
  switch (currentState) {
    case IDLE:
      if (isRisingEdge) {
        currentState = ASCENDING;
        totalCount++;
      } else {
        currentState = DESCENDING;
      }
      break;
    case ASCENDING:
      if (!isRisingEdge) {
        currentState = DESCENDING;
      }
      break;
    case DESCENDING:
      if (isRisingEdge) {
        currentState = ASCENDING;
        totalCount++;
      }
      break;
  }
}

int main(void) {
  // Initialize GPIO pins and interrupt handling
  // ...

  while(1) {
      // Read pulse input and call processPulse()
      // ...
  }
  return 0;
}
```

**Commentary:** This approach is more compact in terms of hardware requirements, utilizing only one counter. However, it introduces the complexity of managing the state machine and requires careful consideration of potential race conditions.  Its effectiveness depends heavily on the precise timing of pulse arrival and processing within the main loop.

**Example 3:  Hardware Differentiator and Single Counter (Assembly - illustrative)**

This demonstrates a conceptually simpler approach with a hypothetical hardware differentiator.  The code is illustrative and uses assembly-like pseudocode, as the exact implementation would depend heavily on the specific hardware architecture.

```assembly
; Assume a hardware differentiator provides two signals:
;   RISING_EDGE (set if rising edge detected)
;   FALLING_EDGE (set if falling edge detected)

; Counter register: COUNTER

loop:
    ; Check for rising edge
    test RISING_EDGE
    jz no_rising
    inc COUNTER    ; Increment counter on rising edge
no_rising:
    ; Check for falling edge (no action needed for this example)
    test FALLING_EDGE
    jz end_loop

end_loop:
    jmp loop
```

**Commentary:** This method relies on specialized hardware to handle edge detection. While highly efficient, it requires specific hardware support and is less portable compared to software-based solutions. The hardware differentiator pre-processes the signal, simplifying the software logic.  The clarity and speed of this method make it a strong contender for demanding high-frequency scenarios.


**3. Resource Recommendations**

For a deeper understanding of digital signal processing and microcontroller peripherals, I would recommend consulting a comprehensive textbook on embedded systems design.  Additionally, the official documentation for your chosen microcontroller's architecture and its peripherals will be invaluable.  Finally, exploration of advanced counter techniques like up/down counters and quadrature decoding can further enhance your understanding.  These resources will provide the necessary theoretical background and practical guidance required to successfully implement bidirectional pulse counting in your specific application.

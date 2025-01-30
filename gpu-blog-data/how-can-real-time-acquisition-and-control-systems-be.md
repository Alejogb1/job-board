---
title: "How can real-time acquisition and control systems be architected?"
date: "2025-01-30"
id: "how-can-real-time-acquisition-and-control-systems-be"
---
Real-time acquisition and control systems demand rigorous determinism; a missed deadline can have catastrophic consequences. I’ve spent the last decade architecting these systems across various industries, from high-speed manufacturing lines to critical infrastructure monitoring, and this constraint of time-predictability always sits at the forefront of design decisions. Achieving this determinism necessitates a holistic approach encompassing hardware selection, software design, and rigorous testing methodologies.

Fundamentally, a real-time acquisition and control system comprises three core layers: a sensing/actuation layer, a data processing layer, and a control/output layer. The sensing/actuation layer interfaces directly with the physical world; it houses sensors collecting data (temperature, pressure, position, etc.) and actuators that effect change (motors, valves, relays). The data processing layer receives the raw data, applies algorithms for filtering, transformation, and analysis, and ultimately derives the actionable information for control. Lastly, the control/output layer translates the processed information into commands for the actuators, thereby completing the feedback loop.

The architecture of each layer is driven by the real-time constraints of the overall system. At the sensing/actuation layer, hardware selection becomes critical. We opt for sensors and actuators with low latency and high reliability. Communication protocols employed here, such as EtherCAT or PROFINET, are chosen for their deterministic nature. Avoidance of inherently unpredictable technologies like USB or wireless connections in critical paths is paramount. Data acquisition must occur at precisely scheduled intervals with minimal jitter. This often involves hardware-based triggers to initiate data conversion, ensuring precise timing.

The data processing layer generally relies on a combination of dedicated hardware and specialized software. Here, a real-time operating system (RTOS) becomes essential. RTOSes provide mechanisms for deterministic task scheduling, ensuring high-priority tasks, such as critical control loops, always execute on time. Kernel preemption, a core feature of RTOSes, allows higher-priority threads to interrupt lower-priority ones, thus mitigating latency. Data processing algorithms must also be optimized for efficiency. In my experience, the trade-off between accuracy and speed is frequently the most challenging aspect. Complex models or transformations can be computationally expensive and may require optimization for real-time execution, often leading to the use of fixed-point arithmetic instead of floating-point, where precision is traded for determinism. This layer is also where data buffering is meticulously managed. Buffers must be of the appropriate size and managed by lock-free algorithms to minimize latency.

The control/output layer interprets the processed information and translates it into control signals. Similar to the sensing layer, deterministic communication protocols are a necessity. Additionally, fault tolerance is paramount. The system must handle failures gracefully, often by switching to a backup control scheme. This can involve redundancy in both hardware and software. For example, I’ve seen redundant controllers in operation, with only one being active while the other is a hot spare.

To illustrate this further, consider three practical examples.

**Example 1: A Simple Temperature Control System**

This example involves a temperature sensor reading the temperature of a chamber and a heater being controlled to keep it at a setpoint.

```c
// Simplified representation using pseudo-C for illustration within the RTOS
// Hardware-specific drivers are assumed.

// Global variables, protected through mutex if necessary
volatile float currentTemperature;
volatile float setpointTemperature = 25.0f;
volatile bool heaterState = false;

// Sampling task (high priority)
void readTemperatureTask() {
  while(1) {
      currentTemperature = readTemperatureSensor(); // Reads sensor via SPI or similar
      rtos_sleep(SAMPLING_PERIOD_MS);  // Sleeps for pre-configured sampling period
  }
}

// Control task (medium priority)
void controlTask() {
  while(1) {
    float error = setpointTemperature - currentTemperature;
    float output = calculatePID(error); // PID algorithm (omitted for brevity)
    if (output > 0) {
      heaterState = true;
      controlHeater(true); // Controls actuator (heater) via a GPIO or similar.
    } else {
      heaterState = false;
      controlHeater(false);
    }
    rtos_sleep(CONTROL_PERIOD_MS); // Sleeps for pre-configured control period
  }
}

// Main task, handles startup
int main() {
  initializeHardware(); // Initialize sensors, actuators
  createThread(readTemperatureTask, HIGH_PRIORITY); // Creates threads in the RTOS
  createThread(controlTask, MEDIUM_PRIORITY);
  startScheduler(); // Start the RTOS scheduler
  return 0;
}
```

In this example, the sensing is prioritized. The temperature is read at a higher frequency than the control loop's period. The control loop then calculates and applies corrections at its own frequency. The `rtos_sleep` function provides a deterministic delay, ensuring predictable timing.

**Example 2: High-Speed Motor Control System**

This system involves an encoder providing feedback on motor position, a current sensor measuring the motor current, and a motor driver controlling the motor speed and position.

```cpp
// Simplified example using C++ for illustration (assuming RTOS/drivers)

class MotorControl {
public:
  MotorControl(float kp, float ki, float kd) : kp_(kp), ki_(ki), kd_(kd), lastError_(0), integral_(0) {}

  void update(float setpoint, float currentPosition, float currentCurrent) {
    float error = setpoint - currentPosition;
    integral_ += error;
    float derivative = error - lastError_;
    float output = kp_ * error + ki_ * integral_ + kd_ * derivative;
    lastError_ = error;

    // Apply current control limitations
    output = constrain(output, -MAX_CURRENT, MAX_CURRENT);

    controlMotorDriver(output); // Send control signal to motor driver
  }
  
private:
  void controlMotorDriver(float current); // Motor driver specific function
  float kp_;
  float ki_;
  float kd_;
  float lastError_;
  float integral_;
};

// Main control loop
void motorControlTask() {
  MotorControl motorControl(KP_VALUE, KI_VALUE, KD_VALUE); // Instantiated once only
  while(true){
      float currentPos = readEncoder(); // Read encoder position via hardware interface
      float currentCurrent = readMotorCurrent(); // Read current via hardware interface
      motorControl.update(SETPOINT_POS, currentPos, currentCurrent);
      rtos_sleep(MOTOR_CONTROL_PERIOD_MS);
  }
}
```

In this example, object-oriented principles are applied, showcasing a simple PID controller. Both encoder readings and current measurements are required before the controller can update its output. Again the use of deterministic sleeping within the RTOS is critical to maintaining time deadlines.

**Example 3: A Multi-Sensor Data Acquisition System**

This illustrates a situation where multiple sensors feed data to a central processing unit.

```python
# Simplified Python example using asyncio for concurrency illustration

import asyncio
import time

# Example Sensor Classes
class TemperatureSensor:
    def __init__(self, sensor_id):
        self.sensor_id = sensor_id
        
    async def read(self):
        await asyncio.sleep(0.001) # Simulate sensor read latency
        return self.sensor_id, (time.time() * 100) % 100 # Fake temperature
        
class PressureSensor:
    def __init__(self, sensor_id):
       self.sensor_id = sensor_id

    async def read(self):
      await asyncio.sleep(0.002) # Simulate sensor read latency
      return self.sensor_id, (time.time() * 50) % 50 # Fake pressure

async def collect_data(sensor):
  while True:
    data = await sensor.read()
    print(f"Sensor {data[0]} Data: {data[1]}") # Pretend to perform further analysis.
    await asyncio.sleep(0.005) # Simulate a fixed collection rate

async def main():
    temp_sensor1 = TemperatureSensor("TEMP_1")
    temp_sensor2 = TemperatureSensor("TEMP_2")
    press_sensor = PressureSensor("PRESS_1")
    
    tasks = [
      asyncio.create_task(collect_data(temp_sensor1)),
      asyncio.create_task(collect_data(temp_sensor2)),
      asyncio.create_task(collect_data(press_sensor))
      ]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
  asyncio.run(main())
```

This example uses asyncio to simulate the concurrency of multiple sensors. While this illustrates concurrency using asynchronous patterns, it should be noted that Python is generally not suitable for hard real-time systems without an external RTOS due to its garbage collection and GIL. In real scenarios with an RTOS, each sensor read and data processing would be running on a dedicated thread with deterministic scheduling. In our example, all of this is simply simulated via python with non deterministic scheduling.

In summary, designing real-time acquisition and control systems demands careful consideration of timing requirements. Hardware selection, RTOS integration, optimized algorithms, and deterministic communication protocols all contribute to achieving predictability. Fault tolerance, robust testing procedures, and a thorough understanding of the application-specific needs are critical to reliable system operation.

For further understanding, I would recommend exploring books on real-time operating systems (specifically ones focused on embedded systems) and embedded systems design.  Study the documentation of real-time communication protocols such as EtherCAT and CANbus. Also, detailed readings into classical control theory, particularly PID controller design, can be hugely beneficial when constructing these kinds of systems. It would also be advantageous to read literature concerning specific real-time architectures such as VxWorks or QNX. Finally, reviewing the datasheet of any microcontroller you intend to use is critical before embarking upon development.

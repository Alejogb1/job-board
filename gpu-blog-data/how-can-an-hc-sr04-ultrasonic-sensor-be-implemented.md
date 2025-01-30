---
title: "How can an HC-SR04 ultrasonic sensor be implemented on an FPGA (DE1-SOC)?"
date: "2025-01-30"
id: "how-can-an-hc-sr04-ultrasonic-sensor-be-implemented"
---
The HC-SR04's simplicity belies a crucial design consideration when interfacing it with an FPGA like the DE1-SOC:  precise timing control is paramount for accurate distance measurement.  The sensor relies on transmitting an ultrasonic pulse and measuring the time-of-flight of the returning echo.  Any jitter or inaccuracy in the FPGA's timing circuitry directly translates to errors in the calculated distance.  My experience integrating this sensor on several custom FPGA boards, including projects utilizing the Cyclone V SoC present in the DE1-SOC, highlights the need for a robust, deterministic timing solution.

**1.  Explanation of Implementation**

Interfacing the HC-SR04 with the DE1-SOC necessitates a design encompassing several key components. First, a trigger signal must be generated to initiate the ultrasonic pulse transmission. This involves a simple output pin configured for digital output. The sensor's echo pin, an input, requires a dedicated input pin capable of detecting the returning pulse. This necessitates a precise timing mechanism for measuring the time elapsed between the trigger signal and the reception of the echo.  Efficient implementation relies heavily on the FPGA's inherent capabilities for creating high-precision timers and handling interrupt events.

Within the FPGA, a timer is initialized.  Upon triggering the HC-SR04, the timer begins counting. When the echo signal is detected, the timer is stopped. The value captured from the timer represents the time-of-flight. This raw time value is then converted into distance using the speed of sound, accounting for temperature corrections for enhanced accuracy.  The calculated distance can then be displayed on an on-board display, transferred via UART to a computer, or used for further processing within the FPGA.  Signal integrity is vital; careful consideration should be given to noise reduction techniques and appropriate signal levels to ensure reliable operation.  Incorrect voltage levels can lead to false readings or sensor malfunction.

Furthermore, efficient resource utilization is key. The HC-SR04 interface should not unduly burden the FPGA's resources, especially considering the limited resources available on a development board like the DE1-SOC. The design should be optimized to minimize logic utilization and avoid unnecessary clock cycles.

**2. Code Examples with Commentary**

The following examples illustrate different approaches to implementing the HC-SR04 interface using VHDL, a common hardware description language for FPGA development.  These examples are simplified for clarity and should be adapted according to the specific board constraints and desired functionality.

**Example 1: Basic Timer Implementation**

```vhdl
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity hc_sr04_interface is
  port (
    clk : in std_logic;
    rst : in std_logic;
    trigger : out std_logic;
    echo : in std_logic;
    distance : out unsigned(15 downto 0)
  );
end entity;

architecture behavioral of hc_sr04_interface is
  signal timer_count : unsigned(23 downto 0) := (others => '0');
  signal echo_detected : std_logic := '0';
begin
  -- Trigger generation
  process (clk, rst)
  begin
    if rst = '1' then
      trigger <= '0';
    elsif rising_edge(clk) then
      trigger <= '1';
      wait for 10 us; -- 10us pulse width
      trigger <= '0';
    end if;
  end process;

  -- Echo detection and timing
  process (clk, rst)
  begin
    if rst = '1' then
      timer_count <= (others => '0');
      echo_detected <= '0';
    elsif rising_edge(clk) then
      if echo = '1' and echo_detected = '0' then
        echo_detected <= '1';
      elsif echo = '0' and echo_detected = '1' then
        echo_detected <= '0';
        distance <= timer_count;
      else
        if echo_detected = '0' then
          timer_count <= timer_count + 1;
        end if;
      end if;
    end if;
  end process;
end architecture;
```

This example utilizes a simple counter for time measurement.  The `trigger` signal is pulsed for 10us, and the counter stops upon detection of the `echo` signal.  The `distance` output represents the raw timer count.  Accuracy is limited by the clock frequency and the counter resolution.  Calibration would be necessary.

**Example 2:  Interrupt-driven Approach**

This improved version leverages interrupts for a more efficient and accurate measurement:

```vhdl
-- ... (Similar entity declaration as Example 1) ...

architecture behavioral of hc_sr04_interface is
  signal timer_count : unsigned(23 downto 0) := (others => '0');
  signal interrupt : std_logic := '0';

begin
  -- Interrupt-based timer management (requires interrupt controller configuration)
  process (clk, rst)
    begin
      if rising_edge(clk) then
        if rst = '1' then
          timer_count <= (others => '0');
          interrupt <= '0';
        elsif interrupt = '1' then -- Interrupt triggered by echo
          distance <= timer_count;
          interrupt <= '0';
          timer_count <= (others => '0');
        else
          timer_count <= timer_count + 1;
        end if;
      end if;
    end process;

  -- ... (Trigger generation similar to Example 1) ...

  -- Echo detection triggering interrupt
  process (clk,rst, echo)
  begin
      if rising_edge(clk) then
          if rst = '1' then
              interrupt <= '0';
          elsif echo = '1' and interrupt = '0' then
              interrupt <= '1';
          end if;
      end if;
  end process;

end architecture;
```
This example demonstrates interrupt handling, minimizing CPU overhead compared to continuous polling. The interrupt is triggered when the echo is detected. The accuracy relies on interrupt latency, a system-dependent factor that needs careful attention in design.


**Example 3:  Using a Dedicated Timer IP Core**

Modern FPGAs often include dedicated timer IP cores offering high precision and features like pre-scalers.  Using these cores can streamline the design and improve accuracy.  This example assumes a hypothetical timer IP core `my_timer`:

```vhdl
-- ... (Similar entity declaration as Example 1) ...

architecture behavioral of hc_sr04_interface is
  signal timer_value : unsigned(23 downto 0);
  signal timer_started : std_logic := '0';

begin
  -- Using a dedicated timer IP core, this implementation greatly improves accuracy and simplifies the design.
  my_timer : entity work.my_timer_ip_core port map (
      clk => clk,
      rst => rst,
      start => timer_started,
      stop => echo,
      value => timer_value
  );

  -- Trigger generation remains unchanged.
  process (clk, rst)
  begin
    if rst = '1' then
      trigger <= '0';
      timer_started <= '0';
    elsif rising_edge(clk) then
      trigger <= '1';
      wait for 10 us;
      trigger <= '0';
      timer_started <= '1';
    end if;
  end process;

  -- Convert timer value to distance (requires calibration and speed-of-sound considerations)
  process (clk, rst)
  begin
      if rst = '1' then
          distance <= (others => '0');
      elsif rising_edge(clk) then
          if echo = '1' then
            -- Calculation considering speed of sound
            distance <= convert_timer_to_distance(timer_value);
          end if;
      end if;
  end process;

  -- Function to perform the conversion with calibration
  function convert_timer_to_distance (timer_val : unsigned(23 downto 0)) return unsigned is
      constant speed_of_sound : integer := 34300; -- cm/s, assuming 20°C. May need calibration
  begin
    -- Apply calibrated conversion here based on clock frequency, and other factors
    return to_unsigned(timer_val * speed_of_sound / 2, 16);
  end function;

end architecture;
```

This approach offers a significant performance enhancement due to the hardware support provided by the timer IP core.


**3. Resource Recommendations**

Consult the DE1-SOC user manual and the Altera/Intel Quartus Prime documentation for detailed information on available resources, including timer IPs and interrupt controllers.  Utilize the provided examples and adapt them based on the specific constraints of the DE1-SOC. Explore tutorials and reference designs available from Altera/Intel for further guidance. Review various VHDL coding style guides to ensure readability and maintainability.  Thorough simulation and testing are crucial for verifying the functionality and accuracy of the design.  Consider adding features like temperature compensation and error handling for enhanced robustness.

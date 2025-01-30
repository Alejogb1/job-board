---
title: "How can image data be stored in an FPGA for real-time video processing?"
date: "2025-01-30"
id: "how-can-image-data-be-stored-in-an"
---
In high-speed video processing applications, the choice of image data storage within an FPGA significantly impacts performance and resource utilization. Directly storing an entire video frame within on-chip Block RAM (BRAM) is often impractical due to BRAM’s limited capacity, especially for high-resolution and high frame rate video. Therefore, alternative strategies, utilizing external memory interfaces and clever data management schemes, are essential for efficient real-time processing.

My experience with designing embedded vision systems has highlighted the need for careful consideration of the memory hierarchy when dealing with video data. Typically, an incoming video stream, often received through a parallel or serial interface, is first staged in a small buffer within the FPGA, which acts as a temporary storage. From this buffer, data can be dispatched to one or several destinations: an external memory device for long term storage, a processing pipeline within the FPGA for real-time analysis, or a combination of both. The decision of how and where to move this data depends heavily on the overall system constraints, such as required processing latency, data bandwidth limitations, and the overall cost of the system.

The primary challenge arises from the trade-off between latency and memory capacity. On-chip BRAM, while providing very low latency access, cannot store large frames. External memory, such as DDR3 or DDR4 SDRAM, offers much greater capacity but introduces higher access latency due to the external interface overhead. Therefore, data transfer must be meticulously managed to ensure a continuous flow of data to the processing units and avoid underflows or overflows. Techniques such as line buffering and ping-pong buffering are employed to mitigate the latency impact of using external memory.

Line buffering, the first example, involves buffering only a single row or a few rows of the image within the FPGA. These lines are read sequentially, one after the other. Once a line is read from the external memory or input interface, the data for that line is loaded into the local on-chip buffer and immediately consumed by the processing block. This mechanism keeps a continuous stream of data without waiting for the entire frame to be loaded. It is particularly useful for image operations that can be executed on a line-by-line basis, such as edge detection algorithms based on horizontal gradients.

```vhdl
-- VHDL implementation of a line buffer for a single row of pixels

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity line_buffer is
    generic (
        DATA_WIDTH : natural := 8; -- Pixel data width
        LINE_LENGTH : natural := 640 -- Number of pixels in a line
    );
    port (
        clk : in std_logic;
        reset : in std_logic;
        data_in : in std_logic_vector(DATA_WIDTH - 1 downto 0);
        valid_in : in std_logic;
        data_out : out std_logic_vector(DATA_WIDTH - 1 downto 0);
        valid_out : out std_logic
    );
end entity line_buffer;

architecture behavioral of line_buffer is

    type line_array is array (0 to LINE_LENGTH -1) of std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal line_buffer_reg : line_array;
    signal write_ptr : integer range 0 to LINE_LENGTH - 1 := 0;
    signal buffer_full : std_logic := '0';

begin

    process(clk)
    begin
        if rising_edge(clk) then
            if reset = '1' then
                write_ptr <= 0;
                buffer_full <= '0';
            elsif valid_in = '1' then
                line_buffer_reg(write_ptr) <= data_in;
                write_ptr <= write_ptr + 1;
                if write_ptr = LINE_LENGTH - 1 then
                    buffer_full <= '1';
                end if;
            end if;
        end if;
    end process;

    data_out <= line_buffer_reg(write_ptr - 1);
    valid_out <= buffer_full;


end architecture behavioral;

```

This example provides a basic VHDL implementation of a line buffer. It uses a register array (`line_buffer_reg`) to hold one line of pixel data. The `write_ptr` keeps track of the current write position within this buffer and updates each clock cycle, providing an output with valid data after a full line has been loaded. This design can be extended to handle more lines, creating a small on-chip buffer for multiple rows.

Another crucial technique is ping-pong buffering. This approach uses two memory banks to enable concurrent read and write operations. While one buffer is being written with incoming data, the other is simultaneously being read by the processing unit. Once the current writing buffer is full, and all data is read from the reading buffer, the roles of the two buffers are swapped. This allows for continuous data flow and avoids stalls while the external memory is being accessed. This approach is highly valuable for frame-based image processing algorithms which require access to the entire frame before execution.

```vhdl
-- VHDL implementation of a ping-pong buffer

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity ping_pong_buffer is
    generic (
        DATA_WIDTH : natural := 8; -- Pixel data width
        FRAME_SIZE : natural := 720*480 -- Total pixels in a frame
    );
    port (
        clk : in std_logic;
        reset : in std_logic;
        data_in : in std_logic_vector(DATA_WIDTH - 1 downto 0);
        valid_in : in std_logic;
        data_out : out std_logic_vector(DATA_WIDTH - 1 downto 0);
        valid_out : out std_logic
    );
end entity ping_pong_buffer;

architecture behavioral of ping_pong_buffer is

    type frame_array is array (0 to FRAME_SIZE -1) of std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal buffer_A : frame_array;
    signal buffer_B : frame_array;
    signal write_ptr : integer range 0 to FRAME_SIZE - 1 := 0;
    signal read_ptr : integer range 0 to FRAME_SIZE - 1 := 0;
    signal active_buffer : std_logic := '0';  -- 0 for A, 1 for B
    signal buffer_full : std_logic := '0';

begin

    process(clk)
    begin
        if rising_edge(clk) then
            if reset = '1' then
                write_ptr <= 0;
                read_ptr <= 0;
                active_buffer <= '0';
                buffer_full <= '0';
            elsif valid_in = '1' then
                if active_buffer = '0' then
                    buffer_A(write_ptr) <= data_in;
                else
                    buffer_B(write_ptr) <= data_in;
                end if;
                write_ptr <= write_ptr + 1;
                if write_ptr = FRAME_SIZE - 1 then
                    buffer_full <= '1';
                    write_ptr <= 0;
                    active_buffer <= not active_buffer; -- Swap buffers
                end if;
            elsif buffer_full = '1' then
                if active_buffer = '1' then
                    data_out <= buffer_A(read_ptr);
                else
                    data_out <= buffer_B(read_ptr);
                end if;

                read_ptr <= read_ptr + 1;
                if read_ptr = FRAME_SIZE - 1 then
                    buffer_full <= '0';
                    read_ptr <= 0;
                end if;
            end if;
        end if;
    end process;

   valid_out <= buffer_full;

end architecture behavioral;

```

This VHDL example illustrates the operation of a ping-pong buffer. It utilizes two large arrays, `buffer_A` and `buffer_B`, representing the two memory buffers. The `active_buffer` signal determines which buffer is currently being written, and the `write_ptr` tracks the writing position in the active buffer. Once a buffer is full, the buffer is swapped and reading starts from the previously written buffer, utilizing the `read_ptr`. This methodology permits asynchronous data writing and reading.

Finally, another common approach combines both line buffering and ping-pong buffering.  This involves using line buffers within a ping-pong system. While one set of line buffers is processing the lines of a previous frame, the other set is receiving data from the next frame from the external memory. This hybrid method efficiently leverages both on-chip and external memory, enhancing overall throughput. For this, the ping-pong logic would remain similar, while each buffer would internally be composed of several line buffers.  This configuration allows you to apply line based operations to the image data, while continuously receiving the incoming video data, allowing for seamless data processing.

```vhdl
-- VHDL implementation of ping-pong buffers with line buffers
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;


entity hybrid_buffer is
    generic (
        DATA_WIDTH : natural := 8;
        LINE_LENGTH : natural := 640;
		NUM_LINES : natural := 10;
        FRAME_HEIGHT : natural := 480;
        FRAME_WIDTH : natural := 720
    );
    port (
        clk : in std_logic;
        reset : in std_logic;
        data_in : in std_logic_vector(DATA_WIDTH - 1 downto 0);
        valid_in : in std_logic;
        data_out : out std_logic_vector(DATA_WIDTH - 1 downto 0);
        valid_out : out std_logic
    );
end entity hybrid_buffer;

architecture behavioral of hybrid_buffer is

    type line_array is array (0 to LINE_LENGTH - 1) of std_logic_vector(DATA_WIDTH - 1 downto 0);
	type line_buffer_array is array (0 to NUM_LINES - 1) of line_array;

	signal buffer_A : line_buffer_array;
	signal buffer_B : line_buffer_array;

    signal write_line_ptr : integer range 0 to NUM_LINES - 1 := 0;
	signal write_pixel_ptr : integer range 0 to LINE_LENGTH - 1 := 0;
    signal read_line_ptr : integer range 0 to NUM_LINES - 1 := 0;
	signal read_pixel_ptr : integer range 0 to LINE_LENGTH - 1 := 0;
    signal active_buffer : std_logic := '0';  -- 0 for A, 1 for B
	signal current_frame_line : integer range 0 to FRAME_HEIGHT -1 := 0;
	signal buffer_full : std_logic := '0';

begin

    process(clk)
    begin
        if rising_edge(clk) then
            if reset = '1' then
                write_line_ptr <= 0;
				write_pixel_ptr <= 0;
                read_line_ptr <= 0;
				read_pixel_ptr <= 0;
                active_buffer <= '0';
				current_frame_line <= 0;
				buffer_full <= '0';
            elsif valid_in = '1' then
				if active_buffer = '0' then
					buffer_A(write_line_ptr)(write_pixel_ptr) <= data_in;
				else
					buffer_B(write_line_ptr)(write_pixel_ptr) <= data_in;
				end if;
				write_pixel_ptr <= write_pixel_ptr + 1;
				if write_pixel_ptr = LINE_LENGTH then
					write_pixel_ptr <= 0;
					write_line_ptr <= write_line_ptr + 1;
					if write_line_ptr = NUM_LINES - 1 then
						write_line_ptr <= 0;
						current_frame_line <= current_frame_line + NUM_LINES;
						if current_frame_line >= FRAME_HEIGHT then
							buffer_full <= '1';
							active_buffer <= not active_buffer;
							current_frame_line <= 0;
						end if;
					end if;
				end if;
            elsif buffer_full = '1' then
				if active_buffer = '1' then
                    data_out <= buffer_A(read_line_ptr)(read_pixel_ptr);
                else
                    data_out <= buffer_B(read_line_ptr)(read_pixel_ptr);
                end if;
				read_pixel_ptr <= read_pixel_ptr + 1;
				if read_pixel_ptr = LINE_LENGTH then
					read_pixel_ptr <= 0;
					read_line_ptr <= read_line_ptr + 1;
					if read_line_ptr = NUM_LINES - 1 then
						read_line_ptr <= 0;
						buffer_full <= '0';
					end if;
				end if;
			end if;
        end if;
    end process;

	valid_out <= buffer_full;


end architecture behavioral;
```
This final VHDL example demonstrates a more complex hybrid buffering scheme.  It combines the ping-pong buffer structure from the prior example with the line buffering concept. The `buffer_A` and `buffer_B` signals represent two banks of line buffers. Data is written to line buffers, controlled by a line pointer and a pixel pointer. The reading process operates similarly, but on the opposite buffer. The `current_frame_line` variable allows the system to keep track of the incoming data, and swap buffers as a frame is read.

In conclusion, effective image data storage within an FPGA for real-time video processing necessitates a careful balance between on-chip and external memory resources. Techniques such as line buffering, ping-pong buffering, and hybrid approaches are used to mitigate latency and ensure a continuous stream of data to processing units. The specific implementation depends upon system parameters, but the fundamental principles of efficient memory management remain crucial.

For further information, I recommend reviewing literature on FPGA architectures with specific emphasis on memory subsystems, as well as advanced digital design texts that discuss memory access optimization. Understanding both the architectural aspects of FPGAs and general memory management techniques will significantly enhance your capability to implement real-time vision applications. Specific vendor documentation regarding available memory resources (BRAM, DDR controllers) should also be consulted to gain deep insight into optimization potential.

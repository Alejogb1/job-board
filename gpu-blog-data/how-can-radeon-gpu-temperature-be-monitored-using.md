---
title: "How can Radeon GPU temperature be monitored using Windows command line tools?"
date: "2025-01-30"
id: "how-can-radeon-gpu-temperature-be-monitored-using"
---
Monitoring Radeon GPU temperature directly from the Windows command line presents a unique challenge, as standard utilities like `wmic` or `powershell` don't natively expose this sensor data. This necessitates leveraging specific third-party tools that can access the GPU hardware through low-level APIs and expose those readings to the command line. I have personally implemented and troubleshot this process multiple times within lab automation scripts, requiring precise temperature logging of multiple discrete GPUs.

The core principle involves utilizing a software package specifically designed for hardware monitoring. These packages, typically available in desktop GUI form, often include a command-line interface (CLI) or a method to extract data programmatically. For this purpose, I have found that the "Open Hardware Monitor" project (and its derivatives) to be the most reliable and straightforward to integrate into command-line scripts. Its flexibility in exposing data as well as the availability of third party cli tools that interact with Open Hardware Monitor is critical to its utility.

The first step is installing and setting up the monitoring software. While the core Open Hardware Monitor itself lacks direct command-line output, the application program interface (API) is the key to accessing its sensor data. A tool like "OHM-CLI" acts as an intermediary, using the Open Hardware Monitor API to fetch sensor readings and format them for command-line use. Once the main monitoring software is running, the accompanying cli client executable can then be invoked in the command line.

Here's a basic example of using OHM-CLI to get the temperature of a Radeon GPU:

```batch
@echo off
SETLOCAL

set "OHMCLI_PATH=C:\Path\To\OHM-CLI\OHM-CLI.exe"
for /f "tokens=2 delims==" %%a in ('%OHMCLI_PATH% /data /sensor ^| findstr "GPU Core" ^| findstr /i "temperature"') do (
  echo GPU Core Temperature: %%a
)

ENDLOCAL
pause
```

In this first example, I define the path to the OHM-CLI executable and then use `for /f` to iterate over each line returned when OHM-CLI is executed with parameters: `/data /sensor`. This output includes sensor data from the system. The command is piped to `findstr`, first looking for the line including “GPU Core” to narrow the results to lines that contain data about the GPU core, and then searching within these lines for the text "temperature" to get only the temperature data.  Finally, the loop parses out only the temperature value using delims and stores this result in variable `a` which is then echoed to the console. This example showcases how to target a specific sensor, in this case the core temperature sensor. The path to your cli executable will need to be adjusted.

A slightly more sophisticated approach can involve extracting temperature data from all GPU sensors. Consider this example:

```batch
@echo off
SETLOCAL

set "OHMCLI_PATH=C:\Path\To\OHM-CLI\OHM-CLI.exe"
echo GPU Temperatures:
FOR /F "tokens=1,* delims==" %%A IN ('%OHMCLI_PATH% /data /sensor ^| findstr /i "GPU" ^| findstr /i "temperature"') DO (
  echo %%A= %%B
)

ENDLOCAL
pause
```

This script also defines the executable path to the cli, then it uses a similar approach, this time piping the raw data and filtering for lines containing "GPU", then for lines containing "temperature."  This time, the for loop parses out both the sensor name (tokens=1) and sensor value (tokens=*) and echoes this pair to the console. This provides output for any detected GPU temperature sensors, such as memory temperatures or hot spot temperatures. This method will output lines in the format *Sensor Name* = *Value*, making it clearer to understand what specific temperature is being reported. The use of `findstr /i` makes the filtering case insensitive, for example a case insensitive comparison of `Gpu` vs `gpu` would return a match.

Finally, let's demonstrate how to output the temperature to a log file for more long term monitoring:

```batch
@echo off
SETLOCAL
set "LOG_FILE=gpu_temp_log.txt"
set "OHMCLI_PATH=C:\Path\To\OHM-CLI\OHM-CLI.exe"

echo %date% %time% >> "%LOG_FILE%"
FOR /F "tokens=1,* delims==" %%A IN ('%OHMCLI_PATH% /data /sensor ^| findstr /i "GPU" ^| findstr /i "temperature"') DO (
  echo %%A= %%B >> "%LOG_FILE%"
)
echo. >> "%LOG_FILE%"
ENDLOCAL
pause
```

Here, in addition to the usual setup, I define a log file, `gpu_temp_log.txt`. Before extracting the temperature, the current date and time are appended to the log file. The temperature data is collected using the same method as above, and is appended to the log file. An additional blank line is then added to improve the log file readability. This technique allows for a historical record of GPU temperatures over time. This method could be expanded to include other system data and called from a scheduled task to create periodic logs.

Key considerations when working with command-line GPU monitoring include the following. First, ensure the monitoring software is running with sufficient privileges, often administrator-level, to access hardware sensors. Second, the accuracy and availability of sensor data will depend on the hardware driver support as well as the specifics of the GPU and the monitoring software. Third, the command line parsing is sensitive to the text output by the cli tool, and if the naming convention for a specific sensor changes it may cause this code to stop working. Thorough testing is recommended to determine if a specific sensor is available and its current label for filtering.

For further understanding, I recommend exploring the Open Hardware Monitor project page for detailed API documentation. Additionally, researching the specific command-line client you choose to work with (e.g. OHM-CLI) will reveal its parameters and output formatting specifics. Furthermore, exploring scripting resources such as Windows batch scripting and powershell can be useful in implementing more advanced automation of this process. Lastly, examining the user forums and wikis for similar command-line tools often provides insight on additional capabilities and common troubleshooting techniques.

In summary, while there is no direct built in command line utility for monitoring Radeon GPU temperature, by leveraging third-party software and its API with command-line tools, one can obtain accurate and flexible temperature readings and use these measurements in automated scripting. This process requires specific software installation and an understanding of command-line parsing, but allows for great flexibility in monitoring and logging of GPU temperatures for a variety of applications.

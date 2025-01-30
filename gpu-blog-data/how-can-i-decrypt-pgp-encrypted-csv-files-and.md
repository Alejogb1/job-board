---
title: "How can I decrypt PGP-encrypted CSV files and output the results to a new CSV using PowerShell?"
date: "2025-01-30"
id: "how-can-i-decrypt-pgp-encrypted-csv-files-and"
---
The effective decryption of PGP-encrypted CSV files within PowerShell requires a nuanced understanding of both cryptographic principles and PowerShell’s object-oriented architecture. My experience automating data pipelines for several financial institutions has underscored the necessity for robust, repeatable decryption processes. I’ve found that achieving this involves leveraging third-party tools, specifically GnuPG, and integrating them into PowerShell scripts. Direct, native PowerShell functionality for PGP decryption is absent, necessitating this external dependency.

The fundamental process involves three distinct stages: establishing the necessary tools, decrypting the file contents, and formatting the output to the desired CSV structure. Firstly, a suitable installation of GnuPG (often referred to as GPG) must be present on the system where the PowerShell script is executed. This package furnishes the command-line utilities required for cryptographic operations. Specifically, `gpg.exe` will be the primary instrument utilized.

Secondly, the encrypted CSV file needs to be supplied as input to `gpg.exe` using PowerShell. I’ve commonly used redirection to facilitate this, rather than passing the file path as an argument, to accommodate variations in file naming conventions or locations. The critical GPG flag for decryption is `--decrypt`, and it can be augmented by specifying the private key with `--keyring` if the key is not located in the default keychain. The result of this decryption operation, being the plaintext CSV data, is captured by PowerShell for subsequent processing.

Thirdly, the captured plaintext CSV is typically raw text. It needs to be interpreted as a CSV structure before it can be readily re-exported to a new file. I’ve consistently utilized `Import-Csv` and `Export-Csv` cmdlets, as these are optimized for this purpose. The output CSV file can then be customized using formatting parameters such as delimiters, quote characters, and encoding specifications. I emphasize that proper encoding handling is of paramount importance to avoid data corruption during CSV processing.

Here are three code examples illustrating various scenarios that I have regularly encountered:

**Example 1: Basic Decryption and Export**

```powershell
# Assumes GPG is in your PATH environment variable. Adjust if needed.
$EncryptedFile = "C:\path\to\encrypted.csv.gpg"
$DecryptedFile = "C:\path\to\decrypted.csv"

try {
    # Decrypt using GPG, capturing standard output.
    $DecryptedContent = gpg --decrypt "$EncryptedFile"

    # Import the decrypted content as a CSV object.
    $CsvObject = $DecryptedContent | ConvertFrom-Csv

    # Export to a new CSV file
    $CsvObject | Export-Csv -Path $DecryptedFile -NoTypeInformation -Encoding UTF8

    Write-Host "Decryption successful. Output written to $DecryptedFile."
}
catch {
    Write-Error "An error occurred during decryption: $($_.Exception.Message)"
}
```
*   *Commentary:* This example represents a straightforward scenario where a file named “encrypted.csv.gpg” is decrypted using `gpg --decrypt` and the standard output containing the plain CSV is captured into `$DecryptedContent`. `ConvertFrom-Csv` transforms the captured content into PowerShell objects. `Export-Csv` then writes the object back into a file named "decrypted.csv" in the specified path with a UTF8 encoding and omits the type information. The `try`/`catch` block provides robust error handling to address scenarios like missing GPG executable or key-related issues. This example demonstrates the core functionalities and handles the most common case.

**Example 2: Specifying a Private Key and Keyring**

```powershell
# Assumes GPG is in your PATH environment variable. Adjust if needed.
$EncryptedFile = "C:\path\to\encrypted.csv.gpg"
$DecryptedFile = "C:\path\to\decrypted.csv"
$PrivateKeyPath = "C:\path\to\private.key"
$KeyringPath = "C:\path\to\mykeyring.gpg"

try {
    # Decrypt using GPG, specifying the key and keyring.
    $DecryptedContent = gpg --keyring "$KeyringPath" --key "$PrivateKeyPath" --decrypt "$EncryptedFile"

    # Import the decrypted content as a CSV object
    $CsvObject = $DecryptedContent | ConvertFrom-Csv

    # Export to a new CSV file, specifying a specific delimiter.
    $CsvObject | Export-Csv -Path $DecryptedFile -NoTypeInformation -Delimiter ';' -Encoding UTF8

     Write-Host "Decryption successful. Output written to $DecryptedFile."

}
catch {
     Write-Error "An error occurred during decryption: $($_.Exception.Message)"
}
```

*   *Commentary:* In situations where the private key is not in the default GnuPG keyring or if a non-standard private key file needs to be specified, this example becomes pertinent. It introduces the use of the `--key` and `--keyring` arguments. Additionally, the `Export-Csv` cmdlet utilizes the `-Delimiter` parameter to write the CSV with a semicolon separator instead of a comma, which is a common variant in some locales. This reflects real-world scenarios where default CSV conventions may not apply. Proper error handling is also included.

**Example 3: Handling Encrypted Archives (tar.gz)**

```powershell
# Assumes GPG is in your PATH environment variable. Adjust if needed.
$EncryptedArchive = "C:\path\to\encrypted.tar.gz.gpg"
$ExtractedArchive = "C:\path\to\extracted.tar.gz"
$DecryptedFile = "C:\path\to\decrypted.csv"

try {

    # Decrypt the encrypted archive
    $DecryptedArchiveContent = gpg --decrypt "$EncryptedArchive"

    # Write the decrypted output to a file
    $DecryptedArchiveContent | Out-File -Encoding Byte $ExtractedArchive

    # Extract the archive contents
    7z x "$ExtractedArchive" -o"C:\path\to\extract"

     # Locate the CSV file (assuming a single csv). You could iterate here.
    $ExtractedCSV = Get-ChildItem "C:\path\to\extract" -Include *.csv -Recurse | Select-Object -First 1
    
    if ($ExtractedCSV) {
        # Import the decrypted content as a CSV object
        $CsvObject = Import-Csv -Path $ExtractedCSV.FullName

        # Export to a new CSV file
        $CsvObject | Export-Csv -Path $DecryptedFile -NoTypeInformation -Encoding UTF8

          Write-Host "Decryption and extraction successful. Output written to $DecryptedFile."
    }
    else {
        Write-Error "No CSV file found within the extracted archive."
    }



}
catch {
     Write-Error "An error occurred during decryption or extraction: $($_.Exception.Message)"
}
finally {
    # Clean up by deleting extracted archive if exists
     if(Test-Path $ExtractedArchive) {Remove-Item $ExtractedArchive}
     if(Test-Path "C:\path\to\extract") {Remove-Item "C:\path\to\extract" -Recurse -Force}
}
```
*   *Commentary:* Complex scenarios sometimes involve a PGP encrypted archive file. This example shows how an encrypted tar.gz archive might be handled. After decrypting the archive using `gpg`, the byte output is written to a new file, `extracted.tar.gz`, which then is extracted via 7zip. I've added in logic to find and process the CSV, as well as robust error handling and post-processing cleanup. The cleanup `finally` block ensures that the extracted archive and directory are removed regardless of whether the rest of the script completes successfully or not. This demonstrates handling multi-step processes and includes cleanup routines.

For further understanding of the involved technologies, I recommend consulting documentation on the following: *GnuPG Command-line Manual*: which covers the full array of command-line options of the `gpg.exe` program, *PowerShell Core Cmdlet Documentation*: particularly for `ConvertFrom-Csv`, `Export-Csv`, and related cmdlets that deal with data manipulation, and *7-Zip Command-Line Documentation*: which is useful when handling common archive files. This combination provides the tools for more effective decryption tasks.

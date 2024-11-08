---
title: "Can't Open My .dat File!  Help!  😩"
date: '2024-11-08'
id: 'can-t-open-my-dat-file-help'
---

```java
public class InputFile {
    public static void main(String[] args) {
        BufferedReader inputStream = null;
        Scanner keys = new Scanner(System.in);

        System.out.print("Enter file name (with .txt extension): ");
        String inFileName = keys.next();

        try {
            inputStream = new BufferedReader(new FileReader(inFileName));
            System.out.println("The file " + inFileName + " contains the following lines:");
            String inFileString = inputStream.readLine();
            List<Integer> numbers = new ArrayList<>();

            while (inFileString != null) {
                System.out.println(inFileString);
                numbers.add(Integer.parseInt(inFileString)); // Parse and store integers
                inFileString = inputStream.readLine();
            }
            inputStream.close();

            String fileName = "numbers.dat";
            try (ObjectOutputStream outputStream = new ObjectOutputStream(new FileOutputStream(fileName))) {
                for (int number : numbers) {
                    outputStream.writeInt(number); // Write to .dat file
                }
            } catch (FileNotFoundException e) {
                System.out.println("Problem opening file.");
            } catch (IOException e) {
                System.out.println("Problem with output to the file.");
            }

            try (ObjectInputStream inputStream2 = new ObjectInputStream(new FileInputStream(fileName))) {
                System.out.println("The file being read yields:");
                int anInteger = inputStream2.readInt();
                while (anInteger >= 0) {
                    System.out.println(anInteger);
                    anInteger = inputStream2.readInt();
                }
            } catch (FileNotFoundException e) {
                System.out.println("Problem with opening the file.");
            } catch (EOFException e) {
                System.out.println("Problem reading the file.");
            } catch (IOException e) {
                System.out.println("There was a problem reading the file.");
            }
        } catch (FileNotFoundException e) {
            System.out.println(inFileName + " not found! Try Again.");
        } catch (IOException e) {
            System.out.println(e.getMessage());
        } catch (NumberFormatException e) {
            System.out.println("Invalid number format in input file.");
        }
    }
}
```

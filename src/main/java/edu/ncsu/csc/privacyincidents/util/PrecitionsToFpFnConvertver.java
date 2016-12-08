package edu.ncsu.csc.privacyincidents.util;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class PrecitionsToFpFnConvertver {

  public static void main(String[] args) throws FileNotFoundException, IOException {
    String predictionsFile = args[0];
    String originalCsvFile = args[1];
    
    List<String> originalFilenames = new ArrayList<String>();
    
    try (BufferedReader reader = new BufferedReader(new FileReader(originalCsvFile))) {
      String line = reader.readLine(); // Skip the header
      
      if (line == null) {
        throw new IllegalArgumentException(
            "The input file (" + originalCsvFile + ") seems to be empty");
      }
      
      while ((line = reader.readLine()) != null) {    
         String shorterLine = line.substring(0, 500);
         originalFilenames.add(shorterLine.split(",")[0]);
      }
    }
    
    try (BufferedReader reader = new BufferedReader(new FileReader(predictionsFile))) {
      String line;
      int i = 0;
      while ((line = reader.readLine()) != null) {
        if (line.contains("+")) {
          // Example line:   1278        2:1        1:0   +   1
          String[] lineParts = line.trim().split("\\s+");       

          if (lineParts[1].equals("1:0")) {
            System.out.println("FN: " + originalFilenames.get(i));
          } else if (lineParts[1].equals("2:1")) {
            System.out.println("FP: " + originalFilenames.get(i));
          }
        }
        i++;
      }
    }
  }

}

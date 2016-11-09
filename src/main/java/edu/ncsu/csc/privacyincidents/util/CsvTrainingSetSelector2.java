package edu.ncsu.csc.privacyincidents.util;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

public class CsvTrainingSetSelector2 {

  Map<String, Set<Long>> classToRowIndices = new HashMap<>();

  public static void main(String[] args) throws IOException {
    String inCsvFilename = args[0];
    String outCsvFilename = args[1];

    int classColumnIndex = 1; // Starting from 0
    int maxSubStrLengthHavingClassCol = 500;
    
    // String classNamesAndSizesToPick = "0:195,4:38,3:117,2:40";
    String classNamesAndSizesToPick = args[2];
    
    CsvTrainingSetSelector2 selector = new CsvTrainingSetSelector2();
    selector.readClassIndices(inCsvFilename, classColumnIndex, maxSubStrLengthHavingClassCol);

    Set<Long> rowIndicesToSelect = selector.selectRowIndices(classNamesAndSizesToPick);

    selector.writeSelectedRowsToCsv(inCsvFilename, outCsvFilename, rowIndicesToSelect);
  }

  public void readClassIndices(String csvFilename, int classColumnIndex,
      int maxSubStrLengthHavingClassCol) throws IOException {
    try (BufferedReader br = new BufferedReader(new FileReader(csvFilename))) {
      long recordNumber = 0;
      String line;
      while ((line = br.readLine()) != null) {
        recordNumber++;
        
        // Skip header and empty lines, if any
        if (recordNumber == 1 || line.trim().isEmpty()) {
          continue;
        }
        
        String shorterLine = line.substring(0, maxSubStrLengthHavingClassCol);
        String className = shorterLine.split(",")[classColumnIndex];
        Set<Long> classRowIndices = classToRowIndices.get(className);
        if (classRowIndices == null) {
          classRowIndices = new HashSet<Long>();
        }
        classRowIndices.add(recordNumber);
        classToRowIndices.put(className, classRowIndices);
      }
    }    
  }

  public Set<Long> selectRowIndices(String classNamesAndSizes) {
    Set<Long> rowIndicesToSelect = new HashSet<Long>();

    String[] classSizes = classNamesAndSizes.split(",");
    for (String classNumSize : classSizes) {
      String className = classNumSize.split(":")[0];
      int classSize = Integer.parseInt(classNumSize.split(":")[1]);
      Set<Long> curClassIndices = classToRowIndices.get(className);
      if (curClassIndices.size() < classSize) {
        throw new IllegalStateException("Class name: " + className + " has fewer than " + classSize
            + " rows");
      }

      // TODO Randomizing may be required here in future
      int i = 0;
      for (Long curIndex : curClassIndices) {
        rowIndicesToSelect.add(curIndex);
        if (++i >= classSize) {
          break;
        }
      }
    }

    return rowIndicesToSelect;
  }

  public void writeSelectedRowsToCsv(String inCsvFilename, String outCsvFilename,
      Set<Long> rowsToSelect) throws FileNotFoundException, IOException {
    try (BufferedReader br = new BufferedReader(new FileReader(inCsvFilename));
        PrintWriter pw = new PrintWriter(new FileWriter(outCsvFilename))) {
      long recordNumber = 0;
      String line;
      while ((line = br.readLine()) != null) {
        recordNumber++;
        
        // Write header
        if (recordNumber == 1) {
          pw.println(line);
        }
        
        // Skip empty lines, if any
        if (line.trim().isEmpty()) {
          continue;
        }
        
        if (rowsToSelect.contains(recordNumber)) {
          pw.println(line);
        }
      }
    }    
  }
}

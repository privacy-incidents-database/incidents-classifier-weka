package edu.ncsu.csc.privacyincidents.util;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.Reader;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVPrinter;
import org.apache.commons.csv.CSVRecord;

public class CsvTrainingSetSelector {

  Map<String, Set<Long>> classToRowIndices = new HashMap<>();

  public static void main(String[] args) throws IOException {
    String inCsvFilename = args[0];
    String outCsvFilename = args[1];

    int classColumnIndex = 1; // Starting from 0
    
    // String classNamesAndSizesToPick = "0:195,4:38,3:117,2:40";
    String classNamesAndSizesToPick = args[2];
    
    CsvTrainingSetSelector selector = new CsvTrainingSetSelector();
    selector.readClassIndices(inCsvFilename, classColumnIndex);

    Set<Long> rowIndicesToSelect = selector.selectRowIndices(classNamesAndSizesToPick);

    selector.writeSelectedRowsToCsv(inCsvFilename, outCsvFilename, rowIndicesToSelect);
  }

  public void readClassIndices(String csvFilename, int classColumnIndex) throws IOException {
    try (Reader in = new FileReader(csvFilename)) {
      CSVParser inParser = CSVFormat.EXCEL.withHeader().parse(in);
      for (CSVRecord record : inParser.getRecords()) {
        String className = record.get(classColumnIndex);
        Set<Long> classRowIndices = classToRowIndices.get(className);
        if (classRowIndices == null) {
          classRowIndices = new HashSet<Long>();
        }
        classRowIndices.add(record.getRecordNumber());
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

    try (Reader in = new FileReader(inCsvFilename);
        CSVPrinter out = new CSVPrinter(new FileWriter(outCsvFilename),
            CSVFormat.EXCEL.withDelimiter(','))) {

      CSVParser inParser = CSVFormat.EXCEL.withHeader().parse(in);

      // Write header
      out.printRecord(inParser.getHeaderMap().keySet());

      for (CSVRecord record : inParser.getRecords()) {
        if (rowsToSelect.contains(record.getRecordNumber())) {
          for (int i = 0; i < record.size(); i++) {
            if (i == 1 && !record.get(i).equals("0")) {
              out.print("1");
            } else {
              out.print(record.get(i));
            }
          } 
          out.println();
        }
      }
    }
  }
}

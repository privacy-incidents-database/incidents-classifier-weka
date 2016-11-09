package edu.ncsu.csc.privacyincidents.util;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;

import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;

public class CsvLeaveOneOutSelector {

  private Map<String, Set<Long>> mClassToRowIndices = new HashMap<>();
  
  private Instances mTrainingData;
  
  private Instances mTestingData;
  
  private String mTestingDataFilename;
  
  private final FastVector mClassVals = new FastVector();

  public static void main(String[] args) throws IOException {
    String inCsvFilename = args[0];
    String outTrainingCsvFilename = args[1];

    int classColumnIndex = 1; // Starting from 0
    int maxSubStrLengthHavingClassCol = 500;
    
    // String classNamesAndSizesToPick = "0:195,4:38,3:117,2:40";
    String classNamesAndSizesToPick = args[2];
    
    String outTestingCsvFilename = args[3];
    long testingRecordNumber = Long.parseLong(args[4]);

    CsvLeaveOneOutSelector selector = new CsvLeaveOneOutSelector();
    
    selector.readClassIndices(inCsvFilename, classColumnIndex, maxSubStrLengthHavingClassCol,
        testingRecordNumber);

    Set<Long> rowIndicesToSelect = selector.selectRowIndices(classNamesAndSizesToPick);

    selector.writeSelectedRowsToCsv(inCsvFilename, outTrainingCsvFilename, rowIndicesToSelect,
        outTestingCsvFilename, testingRecordNumber);
  }
  
  public void readClassIndices(String csvFilename, int classColumnIndex,
      int maxSubStrLengthHavingClassCol, long testingRecordNumber) throws IOException {
    try (BufferedReader br = new BufferedReader(new FileReader(csvFilename))) {
      long recordNumber = -1;
      String line;
      while ((line = br.readLine()) != null) {
        recordNumber++;
        
        // Skip header and empty lines, if any
        if (recordNumber == 0 || line.trim().isEmpty()) {
          continue;
        }
        
        // Skip the testing record; add one to compensate for header
        if (recordNumber == testingRecordNumber) {
          continue;
        }
        
        String shorterLine = line.substring(0, maxSubStrLengthHavingClassCol);
        String className = shorterLine.split(",")[classColumnIndex];
        Set<Long> classRowIndices = mClassToRowIndices.get(className);
        if (classRowIndices == null) {
          classRowIndices = new HashSet<Long>();
        }
        classRowIndices.add(recordNumber);
        mClassToRowIndices.put(className, classRowIndices);
      }
    }    
  }

  public Set<Long> selectRowIndices(String classNamesAndSizes) {
    Set<Long> rowIndicesToSelect = new HashSet<Long>();

    String[] classSizes = classNamesAndSizes.split(",");
    for (String classNumSize : classSizes) {
      String className = classNumSize.split(":")[0];
      int classSize = Integer.parseInt(classNumSize.split(":")[1]);
      Set<Long> curClassIndices = mClassToRowIndices.get(className);
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

  public void writeSelectedRowsToCsv(String inCsvFilename, String outTrainingCsvFilename,
      Set<Long> rowsToSelect, String outTestingCsvFilename, long testingRecordNumber)
      throws FileNotFoundException, IOException {
    try (BufferedReader br = new BufferedReader(new FileReader(inCsvFilename));
        PrintWriter trainingPw = new PrintWriter(new FileWriter(outTrainingCsvFilename));
        PrintWriter testingPw = new PrintWriter(new FileWriter(outTestingCsvFilename))) {
      long recordNumber = -1;
      String line;
      while ((line = br.readLine()) != null) {
        recordNumber++;
        
        // Write header
        if (recordNumber == 0) {
          trainingPw.println(line);
          testingPw.println(line);
        }
                
        // Skip empty lines, if any
        if (line.trim().isEmpty()) {
          continue;
        }

        // Write to the testing file
        if (recordNumber == testingRecordNumber) {
          testingPw.println(line);
        } else if (rowsToSelect.contains(recordNumber)) {
          trainingPw.println(line);
        }
      }
    }    
  }
  
  public void setupTrainingAndTestingInstances(String inCsvFilename, int classColumnIndex,
      int maxSubStrLengthHavingClassCol, String classNamesAndSizesToPick, long testingRecordNumber)
      throws FileNotFoundException, IOException {

    readClassIndices(inCsvFilename, classColumnIndex, maxSubStrLengthHavingClassCol,
        testingRecordNumber);

    Set<Long> rowIndicesToSelect = selectRowIndices(classNamesAndSizesToPick);

    try (BufferedReader br = new BufferedReader(new FileReader(inCsvFilename))) {
      long recordNumber = -1;
      String line;
      while ((line = br.readLine()) != null) {
        recordNumber++;

        // Header
        if (recordNumber == 0) {
          setupAttributes(line);
          continue;
        }

        // Skip empty lines, if any
        if (line.trim().isEmpty()) {
          continue;
        }

        // Write to the testing and training data instances
        if (recordNumber == testingRecordNumber) {
          addOneInstance(line, mTestingData);
          String shorterLine = line.substring(0, maxSubStrLengthHavingClassCol);
          mTestingDataFilename = shorterLine.split(",")[0];
        } else if (rowIndicesToSelect.contains(recordNumber)) {
          addOneInstance(line, mTrainingData);
        }        
      }
      
      mTrainingData.setClassIndex(mTrainingData.numAttributes() - 1);
      mTestingData.setClassIndex(mTestingData.numAttributes() - 1);
    }
  }

  private void setupAttributes(String headerCsv) throws IOException {
    FastVector atts = new FastVector();

    CSVParser parser = CSVParser.parse(headerCsv, CSVFormat.EXCEL.withHeader());
    Map<String, Integer> headerMap = parser.getHeaderMap();
    for (String headerVal : headerMap.keySet()) {
      if (headerMap.get(headerVal) >= 2) {
        atts.addElement(new Attribute(headerVal));
      }
    }

    mClassVals.addElement("0"); // Privacy incident
    mClassVals.addElement("1"); // Nonprivacy incident
    atts.addElement(new Attribute("isPrivacyIncident", mClassVals));

    mTrainingData = new Instances("privacy-incidents-tfidf-training", atts, 0);
    mTestingData = new Instances("privacy-incidents-tfidf-testing", atts, 0);
  }
  
  private void addOneInstance(String oneRow, Instances data) throws IOException {
    CSVParser parser = CSVParser.parse(oneRow, CSVFormat.EXCEL);
    List<CSVRecord> records = parser.getRecords();
    if (records.size() > 1) {
      throw new IllegalStateException("Was not expecting more than one record here!");
    }

    CSVRecord record = records.get(0);

    double[] vals = new double[data.numAttributes()];
    int valIndex = 0;
    for (int i = 2; i < record.size(); i++) {
      vals[valIndex++] = Double.parseDouble(record.get(i));
    }

    vals[valIndex] = mClassVals.indexOf(record.get(1));
    data.add(new Instance(1.0, vals));
  }
  
  public Instances getTrainingInstances() {
    return mTrainingData;
  }
  
  public Instances getTestingInstances() {
    return mTestingData;
  }

  public String getTestingFilename() {
    return mTestingDataFilename;
  }
}

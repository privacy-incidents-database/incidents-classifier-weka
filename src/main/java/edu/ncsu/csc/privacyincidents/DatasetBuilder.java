package edu.ncsu.csc.privacyincidents;

import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.Reader;
import java.util.Map;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;

import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffSaver;

public class DatasetBuilder {
  
  private Instances data;
  
  private final FastVector classVals = new FastVector();
  
  public static void main(String[] args) throws IOException {
    String csvFilename = args[0];
    String arffFilename = args[1];

    DatasetBuilder featureBldr = new DatasetBuilder();
    featureBldr.buildDataset(csvFilename);
    featureBldr.saveArff(arffFilename);
  }

  public void buildDataset(String csvFilename) throws IOException {
    // Setup attributes
    setupAttributes(csvFilename);
    
    // Add data instances
    try (Reader in = new FileReader(csvFilename)) {
      Iterable<CSVRecord> records = CSVFormat.EXCEL.withHeader().parse(in);
      for (CSVRecord record : records) {
        double[] vals = new double[data.numAttributes()];
        int valIndex = 0;
        for (int i = 2; i < record.size(); i++) {
          vals[valIndex++] = Double.parseDouble(record.get(i));
        }
        
        vals[valIndex] = classVals.indexOf(record.get(1));
        data.add(new Instance(1.0, vals));
      }
    }
  }
  
  private void setupAttributes(String csvFilename) throws IOException {
    FastVector atts = new FastVector();
    try (Reader in = new FileReader(csvFilename)) {
      CSVParser parser = CSVFormat.EXCEL.withHeader().parse(in);
      Map<String, Integer> headerMap = parser.getHeaderMap();
      for (String headerVal : headerMap.keySet()) {
        if (headerMap.get(headerVal) >= 2) {
          atts.addElement(new Attribute(headerVal));
        }
      }
    }
    
    classVals.addElement("0"); // Nonprivacy incident
    classVals.addElement("1"); // Privacy incident
    atts.addElement(new Attribute("isPrivacyIncident", classVals));

    data = new Instances("privacy-incidents-tfidf", atts, 0);

  }

  private void saveArff(String arffFilename) throws IOException {
    ArffSaver saver = new ArffSaver();
    saver.setInstances(data);
    saver.setFile(new File(arffFilename));
    saver.writeBatch();
  }

}

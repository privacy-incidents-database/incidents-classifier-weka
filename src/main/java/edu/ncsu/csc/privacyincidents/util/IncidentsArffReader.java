package edu.ncsu.csc.privacyincidents.util;

import java.io.BufferedReader;
import java.io.FileReader;

import weka.core.Instances;
import edu.ncsu.csc.privacyincidents.util.StopWordsHandler;

public class IncidentsArffReader {
  public static Instances readArff(String arffFilename) throws Exception {
    BufferedReader reader = new BufferedReader(new FileReader(arffFilename));
    Instances data = new Instances(reader);
    reader.close();

    data.setClassIndex(data.numAttributes() - 1);

    Instances filteredData = StopWordsHandler.filterInstances(data);

    System.out.println("# Attributes = " + filteredData.numAttributes());

    return filteredData;
  }

}

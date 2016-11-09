package edu.ncsu.csc.privacyincidents.classification;

import java.io.FileWriter;
import java.io.PrintWriter;

import edu.ncsu.csc.privacyincidents.util.CsvLeaveOneOutSelector;
import weka.core.Instances;

public class LeaveOneOutEvaluator {
  public static void main(String[] args) throws Exception {
    String inCsvFilename = args[0];
    String classNamesAndSizesToPick = args[1];
    long numRowsInCsvFile = Long.parseLong(args[2]);

    String falsePositivesFilename = args[3];
    String falseNegativesFilename = args[4];

    try (PrintWriter fpWriter = new PrintWriter(new FileWriter(falsePositivesFilename));
        PrintWriter fNWriter = new PrintWriter(new FileWriter(falseNegativesFilename))) {
      for (long recordNum = 1; recordNum <= numRowsInCsvFile; recordNum++) {
        CsvLeaveOneOutSelector leaveOneOutSelector = new CsvLeaveOneOutSelector();
        leaveOneOutSelector.setupTrainingAndTestingInstances(inCsvFilename, 1, 500,
            classNamesAndSizesToPick, recordNum);

        Instances trainingData = leaveOneOutSelector.getTrainingInstances();
        // System.out.println("# Training Attributes = " +
        // trainingData.numAttributes());
        // System.out.println("# Training Instances = " +
        // trainingData.numInstances());
        // System.out.println("# Training Classes = " +
        // trainingData.numClasses());

        Instances testingData = leaveOneOutSelector.getTestingInstances();
        // System.out.println("# Testing Attributes = " +
        // testingData.numAttributes());
        // System.out.println("# Testing Instances = " +
        // testingData.numInstances());
        // System.out.println("# Testing Classes = " +
        // testingData.numClasses());
        
        PrivacyClassifier privacyClassifier = new PrivacyClassifier(new String[] { "-c", "SMO" });

        double pctCorrect = privacyClassifier.runTrainingAndTesting(trainingData, testingData);

        System.out.println(recordNum + " : " + leaveOneOutSelector.getTestingFilename() + " : "
            + testingData.instance(0).classValue() + " : " + pctCorrect);
        
        if (pctCorrect < 100.0) {
          if (testingData.instance(0).classValue() == 0.0) {
            fNWriter.println(leaveOneOutSelector.getTestingFilename());
          } else {
            fpWriter.println(leaveOneOutSelector.getTestingFilename());
          }
        }
      }
    }
  }
}

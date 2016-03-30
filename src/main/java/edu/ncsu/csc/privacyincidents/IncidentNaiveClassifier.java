package edu.ncsu.csc.privacyincidents;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Random;

import edu.ncsu.csc.privacyincidents.custom.NaivePrivacyClassifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;

public class IncidentNaiveClassifier {

  public static void main(String[] args) throws Exception {
    String arffFilename = args[0];

    BufferedReader reader = new BufferedReader(new FileReader(arffFilename));
    Instances data = new Instances(reader);
    reader.close();

    data.setClassIndex(data.numAttributes() - 1);

    // Instances filteredData = filterData(data);
    Instances filteredData = data;
    
    System.out.println("# Attributes = " + filteredData.numAttributes());
    
    NaivePrivacyClassifier naiveClassifier = new NaivePrivacyClassifier();
    naiveClassifier.buildClassifier(filteredData);
    
    Evaluation eval = new Evaluation(data);
    
    eval.crossValidateModel(naiveClassifier, filteredData, 10, new Random(1));
    
    System.out.println("Percent correct = " + eval.pctCorrect() + "; " + "Percent incorrect = "
        + eval.pctIncorrect());
    
    double precisionSum = 0;
    double recallSum = 0;
    double fMeasureSum = 0;
    
    for (int i = 0; i < 2; i++) {
      System.out.println("Class " + (i + 1));
      System.out.println("Precision " + eval.precision(i));
      precisionSum += eval.precision(i);
      
      System.out.println("Recall " + eval.recall(i));
      recallSum += eval.recall(i);
      
      System.out.println("F-measure " + eval.fMeasure(i));
      fMeasureSum += eval.fMeasure(i);
    }
    
    System.out.println("Avg. Precision " + precisionSum / 2);
    System.out.println("Avg. Recall " + recallSum / 2);
    System.out.println("Avg. F-measure " + fMeasureSum / 2);
  }
}

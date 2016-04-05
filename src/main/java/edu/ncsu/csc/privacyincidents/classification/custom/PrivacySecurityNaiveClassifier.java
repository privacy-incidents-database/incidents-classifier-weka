package edu.ncsu.csc.privacyincidents.classification.custom;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

public class PrivacySecurityNaiveClassifier extends Classifier {

  private static final long serialVersionUID = 1478544817603231136L;
  
  private Instances mInstances;
  
  private int privacyAttIndex = -1;
  private int securityAttIndex = -1;
  
  @Override
  public void buildClassifier(Instances data) throws Exception {
    mInstances = new Instances(data);
    for (int i = 0; i < mInstances.numAttributes(); i++) {
      Attribute att = mInstances.attribute(i);
      if (att.name().equals("privaci")) {
        privacyAttIndex = i;
      }
      if (att.name().equals("secur")) {
        securityAttIndex = i;
      }
    }
    
    if (privacyAttIndex == -1 || securityAttIndex == -1) {
      throw new IllegalArgumentException(
          "The input data does not contain a privacy or security attribute");
    }
  }

  @Override
  public double classifyInstance(Instance instance) {
    double privacyVal = instance.value(instance.attribute(privacyAttIndex));
    double securityVal = instance.value(instance.attribute(securityAttIndex));
    if (privacyVal == 0 && securityVal == 0) {
      if (Math.random() < 0.5) {
        return 0;
      } else {
        return 1;
      }
    }
    if (privacyVal > securityVal) {
      return 1;
    } else {
      return 0;
    }
  }
  
  public static void main(String[] args) throws Exception {
    String arffFilename = args[0];

    BufferedReader reader = new BufferedReader(new FileReader(arffFilename));
    Instances data = new Instances(reader);
    reader.close();

    data.setClassIndex(data.numAttributes() - 1);

    // Instances filteredData = filterData(data);
    Instances filteredData = data;
    
    System.out.println("# Attributes = " + filteredData.numAttributes());
    
    PrivacySecurityNaiveClassifier naiveClassifier = new PrivacySecurityNaiveClassifier();
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

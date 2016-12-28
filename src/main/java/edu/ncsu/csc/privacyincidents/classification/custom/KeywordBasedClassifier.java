package edu.ncsu.csc.privacyincidents.classification.custom;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

public class KeywordBasedClassifier extends Classifier {

  private static final long serialVersionUID = 1478544817603231136L;
  
  private Instances mInstances;
  
  private final List<String> mKeywords = new ArrayList<String>();
  private final List<Integer> mKeywordAttIndices = new ArrayList<Integer>();
  
  /**
   * Constructor: initialize with a list of keywords.
   * 
   * @param keywordsPsv Pipe (|) separated keyword list
   */
  public KeywordBasedClassifier(String keywordsPsv) {
    if (keywordsPsv == null || keywordsPsv.isEmpty()) {
      throw new IllegalArgumentException("Must provide at least one keyword");
    }
    
    for (String keyword : keywordsPsv.split("\\|")) {
      mKeywords.add(keyword.trim().toLowerCase());
    }
    
    System.out.println(mKeywords);
  }
  
  public KeywordBasedClassifier(List<String> keywords) {
    mKeywords.addAll(keywords);
    
    if (mKeywords.isEmpty()) {
      throw new IllegalArgumentException("Must provide at least one keyword");
    }
    
    System.out.println(mKeywords);
  }

  
  @Override
  public void buildClassifier(Instances data) throws Exception {
    mInstances = new Instances(data);
    for (int i = 0; i < mInstances.numAttributes(); i++) {
      Attribute att = mInstances.attribute(i);
      for (String keyword : mKeywords) {
        if (att.name().contains(keyword)) {
          mKeywordAttIndices.add(i);
        }
      }
    }
    
    if (mKeywordAttIndices.size() == 0) {
      throw new  IllegalArgumentException("The input data does not contain any keywords supplied");
    }
  }

  @Override
  public double classifyInstance(Instance instance) {
    for (Integer keywordAttIndex : mKeywordAttIndices) {
      if (instance.value(instance.attribute(keywordAttIndex)) > 0) {
        return 0;
      }
    }
    return 1;
  }
  
  public static void main(String[] args) throws Exception {
    String arffFilename = args[0];
    String keywordsPsv = args[1];

    BufferedReader reader = new BufferedReader(new FileReader(arffFilename));
    Instances data = new Instances(reader);
    reader.close();

    data.setClassIndex(data.numAttributes() - 1);

    // Instances filteredData = filterData(data);
    Instances filteredData = data;
    
    System.out.println("# Attributes = " + filteredData.numAttributes());
    
    KeywordBasedClassifier naiveClassifier = new KeywordBasedClassifier(keywordsPsv);
    naiveClassifier.buildClassifier(filteredData);
    
    Evaluation eval = new Evaluation(data);
    
    eval.crossValidateModel(naiveClassifier, filteredData, 10, new Random(1));
    
    System.out.println("Percent correct = " + eval.pctCorrect() + "; " + "Percent incorrect = "
        + eval.pctIncorrect());
    
    double[][] confusionMatrix = eval.confusionMatrix();
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 2; j++) {
        System.out.print(confusionMatrix[i][j] + "  ");
      }
      System.out.println();
    }

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

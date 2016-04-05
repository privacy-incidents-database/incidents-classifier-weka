package edu.ncsu.csc.privacyincidents.featureselection;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import edu.ncsu.csc.privacyincidents.util.StopWordsHandler;
import weka.classifiers.Evaluation;
import weka.classifiers.rules.OneR;
import weka.core.Instances;

public class IterativeOneRFeatureSelector {

  public static void main(String[] args) throws Exception {
    String arffFilename = args[0];

    BufferedReader reader = new BufferedReader(new FileReader(arffFilename));
    Instances data = new Instances(reader);
    reader.close();

    data.setClassIndex(data.numAttributes() - 1);
    
    List<String> removeList = new ArrayList<String>();

    for (int i = 0; i < 100; i++) {
      Instances filteredData = StopWordsHandler.filterInstances(data, removeList);
      String nextAttribute = getNextRuleAttribute(filteredData);
      removeList.add(nextAttribute);
    }
  }

  private static String getNextRuleAttribute(Instances data) throws Exception {
    OneR oneRClassifier = new OneR();
    oneRClassifier.setOptions(weka.core.Utils.splitOptions("-B 6"));

    oneRClassifier.buildClassifier(data);

    String[] classifierOutput = oneRClassifier.toString().split(":");
    String nextAttribute = classifierOutput[0];
    
    OneR oneREvalClassifier = new OneR();
    oneREvalClassifier.setOptions(weka.core.Utils.splitOptions("-B 6"));

    Evaluation eval = new Evaluation(data);
    eval.crossValidateModel(oneREvalClassifier, data, 10, new Random(1));
    System.out.println(nextAttribute + ": " + eval.pctCorrect() + ", " + eval.pctIncorrect());

    return nextAttribute;
  }
}

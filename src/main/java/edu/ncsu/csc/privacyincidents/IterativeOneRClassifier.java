package edu.ncsu.csc.privacyincidents;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.List;
import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.rules.OneR;
import weka.core.Instances;
import weka.core.Attribute;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

public class IterativeOneRClassifier {

  public static void main(String[] args) throws Exception {
    String arffFilename = args[0];

    BufferedReader reader = new BufferedReader(new FileReader(arffFilename));
    Instances data = new Instances(reader);
    reader.close();

    List<String> removeList = new ArrayList<String>();
    removeList.add("nytimes.com");
    removeList.add("york");
    removeList.add("upgrad");
    removeList.add("explor");
    removeList.add("love");
    removeList.add("longer");
    removeList.add("no");
    removeList.add("browser");
    removeList.add("hear");
    removeList.add("support");
    removeList.add("internet");
    removeList.add("earlier");
    removeList.add("access");
    removeList.add("version");
    removeList.add("print");
    removeList.add("privaci");
    removeList.add("secur");

    data.setClassIndex(data.numAttributes() - 1);

    for (int i = 0; i < 100; i++) {
      Instances filteredData = filterData(data, removeList);
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

  private static Instances filterData(Instances data, List<String> removeList)
      throws Exception {
    Remove remove = new Remove();

    @SuppressWarnings("unchecked")
    Enumeration<Attribute> attributes = data.enumerateAttributes();
    int i = 1;
    StringBuffer attributeIndices = new StringBuffer();
    while (attributes.hasMoreElements()) {
      Attribute attribute = attributes.nextElement();
      if (removeList.contains(attribute.name())) {
        attributeIndices.append(i + ",");
      }
      i++;
    }
    if (attributeIndices.length() > 0) {
      // Remove last comma
      attributeIndices.replace(attributeIndices.length() - 1, attributeIndices.length(), "");
      remove.setAttributeIndices(attributeIndices.toString());
    }

    remove.setInputFormat(data);

    Instances filteredData = Filter.useFilter(data, remove);

    return filteredData;
  }
}

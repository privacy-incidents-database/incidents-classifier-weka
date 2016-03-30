package edu.ncsu.csc.privacyincidents;

import java.io.BufferedReader;
import java.io.FileReader;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.List;
import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO;
import weka.core.Attribute;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

public class IncidentClassifier {

  public static void main(String[] args) throws Exception {
    String arffFilename = args[0];

    BufferedReader reader = new BufferedReader(new FileReader(arffFilename));
    Instances data = new Instances(reader);
    reader.close();

    data.setClassIndex(data.numAttributes() - 1);

    Instances filteredData = filterData(data);
    
    System.out.println("# Attributes = " + filteredData.numAttributes());
    
    SMO smoClassifier = new weka.classifiers.functions.SMO();
    smoClassifier
        .setOptions(weka.core.Utils
            .splitOptions("-C 1.0 -L 0.0010 -P 1.0E-12 -N 0 -V -1 -W 1 -K \"weka.classifiers.functions.supportVector.PolyKernel -C 250007 -E 1.0\""));
        
    Evaluation eval = new Evaluation(data);
    
    eval.crossValidateModel(smoClassifier, filteredData, 10, new Random(1));
    
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

  private static Instances filterData(Instances data) throws Exception {
    List<String> stopWords = Files
        .readAllLines(Paths.get(ClassLoader.getSystemResource("stopwords.txt").toURI()),
            Charset.forName("utf-8"));
    
    List<String> attributeNamesToFilter = new ArrayList<String>();
    for (String stopWord : stopWords) {
      if (!stopWord.startsWith("%")) {
        attributeNamesToFilter.add(stopWord);
      }
    }
    
    System.out.println("# Stop words = " + stopWords.size());
    
    Remove remove = new Remove();

    @SuppressWarnings("unchecked")
    Enumeration<Attribute> attributes = data.enumerateAttributes();
    int i = 1;
    StringBuffer attributeIndices = new StringBuffer();
    while (attributes.hasMoreElements()) {
      Attribute attribute = attributes.nextElement();
      if (attributeNamesToFilter.contains(attribute.name())) {
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

package edu.ncsu.csc.privacyincidents;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.List;
import java.util.Random;

import weka.classifiers.Evaluation;
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
    
    // create new instance of scheme
    weka.classifiers.functions.SMO scheme = new weka.classifiers.functions.SMO();
    // set options
    scheme
        .setOptions(weka.core.Utils
            .splitOptions("-C 1.0 -L 0.0010 -P 1.0E-12 -N 0 -V -1 -W 1 -K \"weka.classifiers.functions.supportVector.PolyKernel -C 250007 -E 1.0\""));
        
    Evaluation eval = new Evaluation(data);
    
    eval.crossValidateModel(scheme, filteredData, 10, new Random(1));
    
    System.out.println(eval.pctCorrect() + ", " + eval.pctIncorrect());
  }

  private static Instances filterData(Instances data) throws Exception {
    List<String> attributeNamesToFilter = new ArrayList<String>();
    attributeNamesToFilter.add("nytimes.com");
    attributeNamesToFilter.add("york");
    attributeNamesToFilter.add("upgrad");
    attributeNamesToFilter.add("explor");
    attributeNamesToFilter.add("love");
    attributeNamesToFilter.add("longer");
    attributeNamesToFilter.add("no");
    attributeNamesToFilter.add("browser");
    attributeNamesToFilter.add("hear");
    attributeNamesToFilter.add("support");
    attributeNamesToFilter.add("internet");
    attributeNamesToFilter.add("earlier");
    attributeNamesToFilter.add("access");
    attributeNamesToFilter.add("version");
    attributeNamesToFilter.add("print");
    attributeNamesToFilter.add("privaci");
    attributeNamesToFilter.add("secur");
    attributeNamesToFilter.add("headlin");
    attributeNamesToFilter.add("edit");
    attributeNamesToFilter.add("concern");
    attributeNamesToFilter.add("page");
    attributeNamesToFilter.add("pleas");
    attributeNamesToFilter.add("email");
    attributeNamesToFilter.add("articl");
    attributeNamesToFilter.add("time");
    
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

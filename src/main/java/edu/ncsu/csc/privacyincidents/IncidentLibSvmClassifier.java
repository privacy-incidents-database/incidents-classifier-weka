package edu.ncsu.csc.privacyincidents;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.List;

import weka.classifiers.functions.LibSVM;
import weka.core.Attribute;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

public class IncidentLibSvmClassifier {

  public static void main(String[] args) throws Exception {
    String arffFilename = args[0];

    BufferedReader reader = new BufferedReader(new FileReader(arffFilename));
    Instances data = new Instances(reader);
    reader.close();

    data.setClassIndex(data.numAttributes() - 1);

    Instances filteredData = filterData(data);
    
    LibSVM svmClassifier = new LibSVM();
    
    svmClassifier
        .setOptions(weka.core.Utils
            .splitOptions("-S 0 -K 0 -D 3 -G 0.0 -R 0.0 -N 0.5 -M 40.0 -C 1.0 -E 0.001 -P 0.1 -model /Users/mpradeep/Software/weka-3-7-12 -seed 1"));

    svmClassifier.buildClassifier(filteredData);
        
    System.out.println(svmClassifier.toString());
    
    System.out.println(svmClassifier.getWeights());
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

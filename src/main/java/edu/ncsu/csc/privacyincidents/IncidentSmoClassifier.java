package edu.ncsu.csc.privacyincidents;

import java.io.BufferedReader;
import java.io.FileReader;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.google.common.base.Functions;
import com.google.common.collect.ImmutableSortedMap;
import com.google.common.collect.Ordering;

import weka.classifiers.functions.SMO;
import weka.core.Attribute;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

public class IncidentSmoClassifier {

  public static void main(String[] args) throws Exception {
    String arffFilename = args[0];

    BufferedReader reader = new BufferedReader(new FileReader(arffFilename));
    Instances data = new Instances(reader);
    reader.close();

    data.setClassIndex(data.numAttributes() - 1);

    Instances filteredData = filterData(data);
    
    SMO smoClassifier = new weka.classifiers.functions.SMO();
    smoClassifier
        .setOptions(weka.core.Utils
            .splitOptions("-C 1.0 -L 0.0010 -P 1.0E-12 -N 0 -V -1 -W 1 -K \"weka.classifiers.functions.supportVector.PolyKernel -C 250007 -E 1.0\""));
    
    smoClassifier.buildClassifier(filteredData);
    
    Map<String, Double> positiveCoeffs = new HashMap<String, Double>();
    Map<String, Double> negativeCoeffs = new HashMap<String, Double>();
    
    String[][][] attributeNames = smoClassifier.attributeNames();
    double[][][] sparseWts = smoClassifier.sparseWeights();
    
    for (int i = 0; i < sparseWts.length; i++) {
      double[][] sparseWtsFirstLevel = sparseWts[i];
      for (int j = 0; j < sparseWtsFirstLevel.length; j++) {
        double[] sparseWtsSecondLevel = sparseWtsFirstLevel[j];
        if (sparseWtsSecondLevel != null) {
          for (int k = 0; k < sparseWtsSecondLevel.length; k++) {
            if (sparseWtsSecondLevel[k] >= 0) {
              positiveCoeffs.put(attributeNames[i][j][k], sparseWtsSecondLevel[k]);
            } else {
              negativeCoeffs.put(attributeNames[i][j][k], sparseWtsSecondLevel[k]);
            }
          }
        }
      }
    }
    
    final Ordering<String> reverseValuesAndNaturalKeysOrdering = Ordering.natural().reverse()
        .nullsLast().onResultOf(Functions.forMap(positiveCoeffs, null))
        .compound(Ordering.natural());
    
    ImmutableSortedMap<String, Double> positiveSortedCoeffs = ImmutableSortedMap.copyOf(
        positiveCoeffs, reverseValuesAndNaturalKeysOrdering);
    
    int i = 0;
    for (String att : positiveSortedCoeffs.keySet()) {
      System.out.println(att + ", " + positiveSortedCoeffs.get(att));
      if (i++ >= 50) {
        break;
      }
    }
    System.out.println("====================================");
    
    final Ordering<String> valuesAndNaturalKeysOrdering = Ordering.natural().nullsLast()
        .onResultOf(Functions.forMap(negativeCoeffs, null)).compound(Ordering.natural());

    ImmutableSortedMap<String, Double> negativeSortedCoeffs = ImmutableSortedMap.copyOf(
        negativeCoeffs, valuesAndNaturalKeysOrdering);

    int j = 0;
    for (String att : negativeSortedCoeffs.keySet()) {
      System.out.println(att + ", " + negativeSortedCoeffs.get(att));
      if (j++ >= 50) {
        break;
      }
    }
  }

  private static Instances filterData(Instances data) throws Exception {
    List<String> attributeNamesToFilter = new ArrayList<String>();
    attributeNamesToFilter.addAll(Files.readAllLines(
        Paths.get(ClassLoader.getSystemResource("stopwords.txt").toURI()),
        Charset.forName("utf-8")));
    System.out.println("# Stop words = " + attributeNamesToFilter.size());
    
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

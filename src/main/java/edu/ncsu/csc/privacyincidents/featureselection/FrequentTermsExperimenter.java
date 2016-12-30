package edu.ncsu.csc.privacyincidents.featureselection;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import edu.ncsu.csc.privacyincidents.classification.PrivacyClassifier;
import weka.core.Instances;

public class FrequentTermsExperimenter {

  public static void main(String[] args) throws Exception {
    String[] outputStats = { "class1Precision", "class1Recall", "class1FMeasure", 
        "class2Precision", "class2Recall", "class2FMeasure", 
        "avgPrecision", "avgRecall", "avgFMeasure" };

    List<String> keywords = readKeywords(args[0]); 
    
    Instances data;
    try (BufferedReader br = new BufferedReader(new FileReader(args[1]))) {
      data = new Instances(br);
      data.setClassIndex(data.numAttributes() - 1);
    }
    
    PrivacyClassifier classifier = new PrivacyClassifier();
    
    try (PrintWriter pw = new PrintWriter(new BufferedWriter(new FileWriter(args[2])))) {
      int base = 2;
      for (int i = 0; Math.pow(base, i) < keywords.size(); i++) {
        List<String> subKeywords = keywords.subList(0, (int) (Math.pow(base, i)));
        Map<String, Double> stats = classifier.runCrossValidationOnKeywordBasedClassifier(data, subKeywords);
        pw.print(Math.pow(base, i));
        for (String outputStat : outputStats) {
          pw.print("," + stats.get(outputStat));
        }
        pw.println();
      }

      Map<String, Double> stats = classifier.runCrossValidationOnKeywordBasedClassifier(data, keywords);
      pw.print(keywords.size());
      for (String outputStat : outputStats) {
        pw.print("," + stats.get(outputStat));
      }
      pw.println();
    }
  }

  private static List<String> readKeywords(String keywordsFilename) throws IOException {
    List<String> keywords = new ArrayList<String>();
    try (BufferedReader br = new BufferedReader(new FileReader(keywordsFilename))) {
      String line;
      while ((line = br.readLine()) != null) {
        String listOfKeywords = line.split(":")[1].trim();
        // the list is enclosed in []
        listOfKeywords = listOfKeywords.substring(1, listOfKeywords.length() - 1);        
        for (String keyword : listOfKeywords.split(",")) {
          keywords.add(keyword.trim());
        }
      }
    }
    return keywords;
  }
}

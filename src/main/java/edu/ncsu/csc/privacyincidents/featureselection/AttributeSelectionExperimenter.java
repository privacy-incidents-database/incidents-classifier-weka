package edu.ncsu.csc.privacyincidents.featureselection;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.Map;

import edu.ncsu.csc.privacyincidents.classification.PrivacyClassifier;
import weka.attributeSelection.ASEvaluation;
import weka.attributeSelection.ChiSquaredAttributeEval;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.Ranker;
import weka.core.Instances;
import weka.filters.Filter;

public class AttributeSelectionExperimenter {

  private final Instances mData;

  private final ASEvaluation mEval;

  private static enum AttributeSelectorName {
    INFO_GAIN, CHI_SQUARED
  };

  public AttributeSelectionExperimenter(String arffFile, String attributeSelectorName)
      throws Exception {
    try (BufferedReader reader = new BufferedReader(new FileReader(arffFile))) {
      mData = new Instances(reader);
      mData.setClassIndex(mData.numAttributes() - 1);
    }
    mEval = getAttributeSelector(attributeSelectorName);
  }

  private ASEvaluation getAttributeSelector(String attributeSelectorName) {
    AttributeSelectorName selectorName = AttributeSelectorName.valueOf(attributeSelectorName);
    switch (selectorName) {
    case INFO_GAIN:
      return new InfoGainAttributeEval();
    case CHI_SQUARED:
      return new ChiSquaredAttributeEval();
    default:
      throw new IllegalStateException(
          selectorName + " is not a supported attribute selector; this should not have happened");
    }
  }

  private Instances filterData(Integer numAttributesToSelect) throws Exception {
    weka.filters.supervised.attribute.AttributeSelection attributeSelector = new weka.filters.supervised.attribute.AttributeSelection();

    Ranker ranker = new Ranker();
    ranker.setNumToSelect(Math.min(numAttributesToSelect, mData.numAttributes() - 1));

    attributeSelector.setEvaluator(mEval);
    attributeSelector.setInputFormat(mData);
    attributeSelector.setSearch(ranker);

    Instances selectedData = Filter.useFilter(mData, attributeSelector);
    return selectedData;
  }

  public static void main(String[] args) throws Exception {
    if (args.length !=4) {
      throw new IllegalArgumentException("Send four arguments...");
    }
    
    String[] outputStats = { "class1Precision", "class1Recall", "class1FMeasure", 
        "class2Precision", "class2Recall", "class2FMeasure", 
        "avgPrecision", "avgRecall", "avgFMeasure" };
    
    AttributeSelectionExperimenter attSelectionExperimenter = new AttributeSelectionExperimenter(
        args[0], args[1]);
    
    int base = 2;
    
    try (PrintWriter pw = new PrintWriter(new BufferedWriter(new FileWriter(args[3])))) {
      for (int i = 0; Math.pow(base, i) < (attSelectionExperimenter.mData.numAttributes() - 1); i++) {
        Instances selectedData = attSelectionExperimenter.filterData((int) Math.pow(base, i));
        PrivacyClassifier classifier = new PrivacyClassifier();
        Map<String, Double> stats = classifier.runCrossValidation(selectedData, args[2]);
        pw.print(Math.pow(base, i));
        for (String outputStat : outputStats) {
          pw.print("," + stats.get(outputStat));
        }
        pw.println();
      }
      
      PrivacyClassifier classifier = new PrivacyClassifier();
      Map<String, Double> stats = classifier.runCrossValidation(attSelectionExperimenter.mData,
          args[2]);
      pw.print((attSelectionExperimenter.mData.numAttributes() - 1));
      for (String outputStat : outputStats) {
        pw.print("," + stats.get(outputStat));        
      }
      pw.println();
    }
  }
}

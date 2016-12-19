package edu.ncsu.csc.privacyincidents.featureselection;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;

import weka.attributeSelection.ASEvaluation;
import weka.attributeSelection.AttributeSelection;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.Ranker;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.filters.Filter;

public class AttributeSelector {

  private Instances mData;

  private static enum AttributeSelectorName {
    InfoGain
  };

  public AttributeSelector(String arffFile) throws IOException {
    try (BufferedReader reader = new BufferedReader(new FileReader(arffFile))) {
      mData = new Instances(reader);
      mData.setClassIndex(mData.numAttributes() - 1);
    }
  }

  private ASEvaluation getAttributeSelector(String attributeSelectorName) {
    AttributeSelectorName selectorName = AttributeSelectorName.valueOf(attributeSelectorName);
    switch (selectorName) {
    case InfoGain:
      return new InfoGainAttributeEval();
    default:
      throw new IllegalStateException(
          selectorName + " is not a supported attribute selector; this should not have happened");
    }
  }

  private void selectAttributes(String attributeSelectorName, Integer numAttributesToSelect,
      String rankedAttributesFilename) throws Exception {
    ASEvaluation eval = getAttributeSelector(attributeSelectorName);

    AttributeSelection attributeSelector = new AttributeSelection();

    Ranker ranker = new Ranker();
    ranker.setNumToSelect(Math.min(numAttributesToSelect, mData.numAttributes() - 1));

    attributeSelector.setEvaluator(eval);
    attributeSelector.setSearch(ranker);
    attributeSelector.SelectAttributes(mData);

    double[][] rankedAttribuesArray = attributeSelector.rankedAttributes();
    try (PrintWriter writer = new PrintWriter(
        new BufferedWriter(new FileWriter(rankedAttributesFilename)))) {
      for (int i = 0; i < rankedAttribuesArray.length; i++) {
        writer.println(mData.attribute((int) rankedAttribuesArray[i][0]).name() + ","
            + rankedAttribuesArray[i][1]);
      }
    }

    // The following logic was too slow; so, I wrote another function
    /*
    int[] indices = attributeSelector.selectedAttributes();
    List<Integer> indicesList = new ArrayList<Integer>();
    for (int index : indices) {
      indicesList.add(index);
    }

    Instances selectedData = new Instances(mData);
    for (int i = selectedData.numAttributes() - 1; i >= 0; i--) {
      if (!indicesList.contains(i)) {
        selectedData.deleteAttributeAt(i);
      }
    }

    ArffSaver saver = new ArffSaver();
    saver.setInstances(selectedData);
    saver.setFile(new File(selectedAttributesArffFilename));
    saver.writeBatch();
    */
  }

  private void filterData(String attributeSelectorName, Integer numAttributesToSelect,
      String selectedAttributesArffFilename) throws Exception {
    ASEvaluation eval = getAttributeSelector(attributeSelectorName);

    weka.filters.supervised.attribute.AttributeSelection attributeSelector = new weka.filters.supervised.attribute.AttributeSelection();

    Ranker ranker = new Ranker();
    ranker.setNumToSelect(Math.min(numAttributesToSelect, mData.numAttributes() - 1));

    attributeSelector.setEvaluator(eval);
    attributeSelector.setSearch(ranker);
    attributeSelector.setInputFormat(mData);

    Instances selectedData = Filter.useFilter(mData, attributeSelector);
    ArffSaver saver = new ArffSaver();
    saver.setInstances(selectedData);
    saver.setFile(new File(selectedAttributesArffFilename));
    saver.writeBatch();
  }

  public static void main(String[] args) throws Exception {
    if (args.length != 5) {
      throw new IllegalArgumentException("You must send five arguments...");
    }

    AttributeSelector attSelector = new AttributeSelector(args[0]);
    attSelector.selectAttributes(args[1], Integer.parseInt(args[2]), args[3]);
    attSelector.filterData(args[1], Integer.parseInt(args[2]), args[4]);
  }
}

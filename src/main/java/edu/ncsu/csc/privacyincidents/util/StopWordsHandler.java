package edu.ncsu.csc.privacyincidents.util;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.URISyntaxException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.List;

import weka.core.Attribute;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

public class StopWordsHandler {
  
  private final List<String> stopWords = new ArrayList<String>();
  
  public StopWordsHandler(String stopwordFilename) throws IOException, URISyntaxException {
    String line = null;
    try (BufferedReader br = new BufferedReader(
        new InputStreamReader(StopWordsHandler.class.getResourceAsStream(stopwordFilename)))) {
      while ((line = br.readLine()) != null) {
        if (!line.startsWith("//"))
          stopWords.add(line);
      }
    }
    System.out.println("# Stop words = " + stopWords.size());
  }

  public StopWordsHandler(List<String> stopwordFilenames) throws IOException, URISyntaxException {
    for (String stopwordFilename : stopwordFilenames) {
      String line = null;
      try (BufferedReader br = new BufferedReader(
          new InputStreamReader(StopWordsHandler.class.getResourceAsStream(stopwordFilename)))) {
        while ((line = br.readLine()) != null) {
          if (!line.startsWith("//"))
            stopWords.add(line);
        }
      }
    }
    System.out.println("# Stop words = " + stopWords.size());
  }
  
  public boolean isStopword(String word) {
    return stopWords.contains(word.trim().toLowerCase());
  }
  
  @Deprecated // TODO: Create nonstatic methods for these
  public static Instances filterInstances(Instances data) throws Exception {
    List<String> stopWords = Files
        .readAllLines(Paths.get(ClassLoader.getSystemResource("stopwords.txt").toURI()),
            Charset.forName("utf-8"));
    
    List<String> attributeNamesToFilter = new ArrayList<String>();
    for (String stopWord : stopWords) {
      if (!stopWord.startsWith("//")) {
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
  
  @Deprecated // TODO: Create nonstatic methods for these
  public static Instances filterInstances(Instances data, List<String> additionalStopWords)
      throws Exception {
    
    List<String> stopWords = Files
        .readAllLines(Paths.get(ClassLoader.getSystemResource("stopwords.txt").toURI()),
            Charset.forName("utf-8"));
    
    List<String> attributeNamesToFilter = new ArrayList<String>();
    for (String stopWord : stopWords) {
      if (!stopWord.startsWith("//")) {
        attributeNamesToFilter.add(stopWord);
      }
    }

    for (String additionalWord : additionalStopWords) {
      attributeNamesToFilter.add(additionalWord);
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

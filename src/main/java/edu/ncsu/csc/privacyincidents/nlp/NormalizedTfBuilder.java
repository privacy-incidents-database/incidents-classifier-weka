package edu.ncsu.csc.privacyincidents.nlp;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Set;
import java.util.SortedSet;
import java.util.TreeSet;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVPrinter;

import com.google.common.collect.HashBasedTable;
import com.google.common.collect.Table;

import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreAnnotations.LemmaAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.PartOfSpeechAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.util.CoreMap;

public class NormalizedTfBuilder implements AutoCloseable {

  private StanfordCoreNLP mNlpPipeline;

  private Map<Integer, File> mPositiveFilesMap = new HashMap<Integer, File>();

  private Map<Integer, File> mNegativeFilesMap = new HashMap<Integer, File>();

  // For TF, count terms in each document
  private Table<Integer, String, Integer> docTermCounts = HashBasedTable.create();

  // TF table
  private Table<Integer, String, Double> tf = HashBasedTable.create();

  public NormalizedTfBuilder() {
    // Open NLP pipleline
    Properties nlpProps = new Properties();
    nlpProps.setProperty("annotators", "tokenize, ssplit, pos, lemma");
    mNlpPipeline = new StanfordCoreNLP(nlpProps);
  }

  public void close() throws Exception {
    StanfordCoreNLP.clearAnnotatorPool();
  }

  public static void main(String[] args)
      throws FileNotFoundException, ClassNotFoundException, IOException, Exception {

    String positiveFilesTopLevelDirs = args[0];
    String negativeFilesTopLevelDirs = args[1];
    String outTfCsvFilename = args[2];

    try (NormalizedTfBuilder tfIdfBldr = new NormalizedTfBuilder()) {

      tfIdfBldr.readFilenames(positiveFilesTopLevelDirs, negativeFilesTopLevelDirs);

      tfIdfBldr.readFilesAndUpdateCount();

      tfIdfBldr.buildNormalizedTf();
      
      tfIdfBldr.writeTfToCsv(outTfCsvFilename);
    }
  }

  private void readFilenames(String positiveFilesTopLevelDirs, String negativeFilesTopLevelDirs) {
    String DIRNAME_SEPARATOR = ":";

    Set<File> positiveFiles = new HashSet<File>();
    String[] positveFilesTopLevelDirsArray = positiveFilesTopLevelDirs.split(DIRNAME_SEPARATOR);
    for (int i = 0; i < positveFilesTopLevelDirsArray.length; i++) {
      positiveFiles.addAll(listFilesForDirectory(new File(positveFilesTopLevelDirsArray[i])));
    }

    System.out.println("Number of positive files: " + positiveFiles.size());

    Set<File> negativeFiles = new HashSet<File>();
    String[] negativeFilesTopLevelDirsArray = negativeFilesTopLevelDirs.split(DIRNAME_SEPARATOR);
    for (int i = 0; i < negativeFilesTopLevelDirsArray.length; i++) {
      negativeFiles.addAll(listFilesForDirectory(new File(negativeFilesTopLevelDirsArray[i])));
    }
    System.out.println("Number of negative files: " + negativeFiles.size());

    int id = 0;
    for (File positivefile : positiveFiles) {
      mPositiveFilesMap.put(id++, positivefile);
    }
    for (File negativefile : negativeFiles) {
      mNegativeFilesMap.put(id++, negativefile);
    }
  }

  private Set<File> listFilesForDirectory(final File directory) {
    if (!directory.isDirectory()) {
      throw new IllegalArgumentException(directory.getName() + " is not a directory");
    }

    Set<File> fileNames = new HashSet<File>();

    for (final File fileEntry : directory.listFiles()) {
      if (fileEntry.isDirectory()) {
        fileNames.addAll(listFilesForDirectory(fileEntry));
      } else {
        fileNames.add(fileEntry);
      }
    }
    return fileNames;
  }
  
  private void readFilesAndUpdateCount() throws IOException {
    for (Integer posFileID : mPositiveFilesMap.keySet()) {
      File posFile = mPositiveFilesMap.get(posFileID);
      updateCounts(posFileID, posFile);
    }

    for (Integer negFileID : mNegativeFilesMap.keySet()) {
      File posFile = mNegativeFilesMap.get(negFileID);
      updateCounts(negFileID, posFile);
    }
  }

  private void updateCounts(Integer fileId, File file) throws IOException {
    byte[] encoded = Files.readAllBytes(file.toPath());
    String fileContents = new String(encoded);

    List<String> bagOfWords = getBagOfWords(fileContents);

    for (String word : bagOfWords) {
      Integer wordCount = docTermCounts.get(fileId, word);
      if (wordCount == null) {
        docTermCounts.put(fileId, word, 1);
      } else {
        docTermCounts.put(fileId, word, wordCount + 1);
      }
    }
  }

  private List<String> getBagOfWords(String text) {
    List<String> words = new ArrayList<String>();

    Annotation annotation = new Annotation(text);
    mNlpPipeline.annotate(annotation);

    for (CoreMap sentence : annotation.get(CoreAnnotations.SentencesAnnotation.class)) {
      for (CoreLabel token : sentence.get(TokensAnnotation.class)) {
        String pos = token.get(PartOfSpeechAnnotation.class);
        if (pos.startsWith("N") || pos.startsWith("V") 
            || pos.startsWith("J") || pos.startsWith("R")) {
          String lemma = token.get(LemmaAnnotation.class);
          words.add(lemma.toLowerCase());
        }
      }
    }

    return words;
  }

  private void buildNormalizedTf() {
    // Change term counts to normalized term frequencies
    for (Integer docId : docTermCounts.rowKeySet()) {
      Map<String, Integer> termCounts = docTermCounts.row(docId);
      Integer maxCount = Collections.max(termCounts.values());
      for (String term : termCounts.keySet()) {
        double tF = 0.4 + (0.6 * termCounts.get(term) / maxCount);
        tf.put(docId, term, tF);
      }
    }
  }
    
  private void writeTfToCsv(String outTfCsvFilename) throws IOException {
    try (CSVPrinter tfOut = new CSVPrinter(new FileWriter(outTfCsvFilename),
        CSVFormat.EXCEL.withDelimiter(','))) {

      SortedSet<String> sortedWords = new TreeSet<String>();
      sortedWords.addAll(tf.columnKeySet());
      
      // Write header
      tfOut.print("fileName");
      tfOut.print("isPrivacy");
      for (String word : sortedWords) {
        tfOut.print(word);
      }
      tfOut.println();
      
      // Write positive files
      for (Integer posFileId : mPositiveFilesMap.keySet()) {
        tfOut.print(mPositiveFilesMap.get(posFileId));
        tfOut.print("0");
        
        for (String word : sortedWords) {
          Double tfScore = tf.get(posFileId, word);
          if (tfScore == null) {
            tfScore = 0.0;
          }
          tfOut.print(tfScore);          
        }
        tfOut.println();
      }
      
      // Write negative files
      for (Integer negFileId : mNegativeFilesMap.keySet()) {
        tfOut.print(mNegativeFilesMap.get(negFileId));
        tfOut.print("1");
        
        for (String word : sortedWords) {
          Double tfScore = tf.get(negFileId, word);
          if (tfScore == null) {
            tfScore = 0.0;
          }
          tfOut.print(tfScore * tfScore);
        }
        tfOut.println();
      }
    }
  }
}

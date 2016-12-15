package edu.ncsu.csc.privacyincidents.nlp;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.net.URISyntaxException;
import java.nio.file.Files;
import java.util.ArrayList;
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

import edu.ncsu.csc.privacyincidents.util.StopWordsHandler;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreAnnotations.LemmaAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.PartOfSpeechAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.StringUtils;

public class TfIdfBuilder implements AutoCloseable {

  private StanfordCoreNLP mNlpPipeline;
  
  private StopWordsHandler mStopWordsHandler;

  private Map<Integer, File> mPositiveFilesMap = new HashMap<Integer, File>();

  private Map<Integer, File> mNegativeFilesMap = new HashMap<Integer, File>();

  // For TF, count terms in each document
  private Table<Integer, String, Integer> docTermCounts = HashBasedTable.create();

  // For IDF, count number of docs for each term
  private Map<String, Integer> termDocCounts = new HashMap<String, Integer>();

  // TF table
  private Table<Integer, String, Double> tf = HashBasedTable.create();
  
  // IDF table
  private Table<Integer, String, Double> idf = HashBasedTable.create();

  // TF-IDF table // Don't need this
  // private Table<Integer, String, Double> tfIdf = HashBasedTable.create();


  public TfIdfBuilder() throws IOException, URISyntaxException {
    // Open NLP pipleline
    Properties nlpProps = new Properties();
    nlpProps.setProperty("annotators", "tokenize, ssplit, pos, lemma");
    mNlpPipeline = new StanfordCoreNLP(nlpProps);
    
    mStopWordsHandler = new StopWordsHandler("english-stopwords.txt");
  }

  public void close() throws Exception {
    StanfordCoreNLP.clearAnnotatorPool();
  }

  public static void main(String[] args)
      throws FileNotFoundException, ClassNotFoundException, IOException, Exception {

    String positiveFilesTopLevelDirs = args[0];
    String negativeFilesTopLevelDirs = args[1];
    String outTfCsvFilename = args[2];
    String outIdfCsvFilename = args[3];
    String outTfIdfCsvFilename = args[4];

    try (TfIdfBuilder tfIdfBldr = new TfIdfBuilder()) {

      tfIdfBldr.readFilenames(positiveFilesTopLevelDirs, negativeFilesTopLevelDirs);

      tfIdfBldr.readFiles();

      tfIdfBldr.buildTfIdf();
      
      tfIdfBldr.writeToCsv(outTfCsvFilename, outIdfCsvFilename, outTfIdfCsvFilename);
    }
  }

  private void readFilenames(String positiveFilesTopLevelDirs, String negativeFilesTopLevelDirs) {
    String DIRNAME_SEPARATOR = ":";

    Set<File> positiveFiles = new HashSet<File>();
    String[] positveFilesTopLevelDirsArray = positiveFilesTopLevelDirs.split(DIRNAME_SEPARATOR);
    for (int i = 0; i < positveFilesTopLevelDirsArray.length; i++) {
      System.out.println(positveFilesTopLevelDirsArray[i]);
      positiveFiles.addAll(listFilesForDirectory(new File(positveFilesTopLevelDirsArray[i])));
    }

    System.out.println("Number of positive files: " + positiveFiles.size());

    Set<File> negativeFiles = new HashSet<File>();
    String[] negativeFilesTopLevelDirsArray = negativeFilesTopLevelDirs.split(DIRNAME_SEPARATOR);
    for (int i = 0; i < negativeFilesTopLevelDirsArray.length; i++) {
      System.out.println(negativeFilesTopLevelDirsArray[i]);
      negativeFiles.addAll(listFilesForDirectory(new File(negativeFilesTopLevelDirsArray[i])));
    }
    System.out.println("Number of negative files: " + negativeFiles.size());

    int id = 0;
    for (File positivefile : positiveFiles) {
      // System.out.println("Positive file: " + positivefile);
      mPositiveFilesMap.put(id++, positivefile);
    }
    for (File negativefile : negativeFiles) {
      // System.out.println("Negative file: " + negativefile);
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
  
  private void readFiles() throws IOException {
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

    Set<String> bagOfUniqueWords = new HashSet<String>();
    bagOfUniqueWords.addAll(bagOfWords);

    for (String word : bagOfUniqueWords) {
      Integer docCount = termDocCounts.get(word);
      if (docCount == null) {
        termDocCounts.put(word, 1);
      } else {
        termDocCounts.put(word, docCount + 1);
      }
    }
  }

  private List<String> getBagOfWords(String text) {
    List<String> words = new ArrayList<String>();

    Annotation annotation = new Annotation(text);
    mNlpPipeline.annotate(annotation);

    for (CoreMap sentence : annotation.get(CoreAnnotations.SentencesAnnotation.class)) {
      for (CoreLabel token : sentence.get(TokensAnnotation.class)) {
        // Discard words that only have punctuations, numerals
        if (StringUtils.isPunct(token.originalText())
            || StringUtils.isNumeric(token.originalText())) {
          continue;
        }
        
        // This seems redundant given the following check on isAlpha(), but
        // somehow this seems to increase prediction accuracy. But, this could
        // just be due to the fact that I randomly select the negative set and
        // results might differ slightly from run to run.
        if (!StringUtils.isAlphanumeric(token.originalText())) {
          continue;
        }
                
        // Discard non alphabetic words. This may be too restrictive! Consider
        // adding a list of allowed alphanumeric words such as "49ers"
        if (!StringUtils.isAlpha(token.originalText())) {
          // Keep hashtags. This didn't help accuracy; so, commenting it
          /*if (!token.originalText().startsWith("#")) {
            continue;
          }*/
          continue;
        }        
        
        /*
         * Long words are likely not natural, e.g., I found instances such as
         * januaryfebrauary...december in the dataset. Longest common word has
         * length 20 --
         * https://web.archive.org/web/20090427054251/http://www.maltron.com/
         * words/words-longest-modern.html
         */
        if (token.originalText().length() > 20) {
          continue;
        }

        String pos = token.get(PartOfSpeechAnnotation.class);
        // System.out.println(token + ": " + pos);
        if (pos.startsWith("N") || pos.startsWith("V") 
            || pos.startsWith("J") || pos.startsWith("R")) {
          String lemma = token.get(LemmaAnnotation.class);
          
          // Remove word if it is a stop word
          if (!mStopWordsHandler.isStopword(lemma)) {
            words.add(lemma.toLowerCase());
          }
        }
      }
    }

    return words;
  }

  private void buildTfIdf() {
    // Change term counts to term frequencies
    for (Integer docId : docTermCounts.rowKeySet()) {
      Map<String, Integer> termCounts = docTermCounts.row(docId);
      // Integer maxCount = Collections.max(termCounts.values());
      for (String term : termCounts.keySet()) {
        // double tF = 0.4 + (0.6 * termCounts.get(term) / maxCount);
        double tF = 1 + Math.log(termCounts.get(term));
        double iDF = Math.log((double) (mPositiveFilesMap.size() + mNegativeFilesMap.size())
            / termDocCounts.get(term));
        tf.put(docId, term, tF);
        idf.put(docId, term, iDF);
        // tfIdf.put(docId, term, tF * iDF); //Don't need this
      }
    }
  }
  
  private void writeToCsv(String outTfCsvFilename, String outIdfCsvFilename,
      String outTfIdfCsvFilename) throws IOException {
    writeTfToCsv(outTfCsvFilename);
    writeIdfToCsv(outIdfCsvFilename);
    writeTfIdfToCsv(outTfIdfCsvFilename);
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
          tfOut.print(tfScore);
        }
        tfOut.println();
      }
    }
  }
  
  private void writeIdfToCsv(String outIdfCsvFilename) throws IOException {
    try (CSVPrinter idfOut = new CSVPrinter(new FileWriter(outIdfCsvFilename),
        CSVFormat.EXCEL.withDelimiter(','))) {

      SortedSet<String> sortedWords = new TreeSet<String>();
      sortedWords.addAll(tf.columnKeySet());
      
      // Write header
      idfOut.print("fileName");
      idfOut.print("isPrivacy");
      for (String word : sortedWords) {
        idfOut.print(word);
      }
      idfOut.println();
      
      // Write positive files
      for (Integer posFileId : mPositiveFilesMap.keySet()) {
        idfOut.print(mPositiveFilesMap.get(posFileId));
        idfOut.print("0");
        
        for (String word : sortedWords) {
          Double idfScore = idf.get(posFileId, word);
          if (idfScore == null) {
            idfScore = 0.0;
          }
          idfOut.print(idfScore);          
        }
        idfOut.println();
      }
      
      // Write negative files
      for (Integer negFileId : mNegativeFilesMap.keySet()) {
        idfOut.print(mNegativeFilesMap.get(negFileId));
        idfOut.print("1");
        
        for (String word : sortedWords) {
          Double idfScore = idf.get(negFileId, word);
          if (idfScore == null) {
            idfScore = 0.0;
          }
          idfOut.print(idfScore);
        }
        idfOut.println();
      }
    }
  }

  private void writeTfIdfToCsv(String outTfIdfCsvFilename) throws IOException {
    try (CSVPrinter tfIdfOut = new CSVPrinter(new FileWriter(outTfIdfCsvFilename),
        CSVFormat.EXCEL.withDelimiter(','))) {

      SortedSet<String> sortedWords = new TreeSet<String>();
      sortedWords.addAll(tf.columnKeySet());
      
      // Write header
      tfIdfOut.print("fileName");
      tfIdfOut.print("isPrivacy");
      for (String word : sortedWords) {
        tfIdfOut.print(word);
      }
      tfIdfOut.println();
      
      // Write positive files
      for (Integer posFileId : mPositiveFilesMap.keySet()) {        
        tfIdfOut.print(mPositiveFilesMap.get(posFileId));
        tfIdfOut.print("0");

        for (String word : sortedWords) {
          Double tfScore = tf.get(posFileId, word);
          if (tfScore == null) {
            tfScore = 0.0;
          }
          
          Double idfScore = idf.get(posFileId, word);
          if (idfScore == null) {
            idfScore = 0.0;
          }
          
          tfIdfOut.print(tfScore * idfScore);
        }
        tfIdfOut.println();
      }
      
      // Write negative files
      for (Integer negFileId : mNegativeFilesMap.keySet()) {
        tfIdfOut.print(mNegativeFilesMap.get(negFileId));
        tfIdfOut.print("1");

        for (String word : sortedWords) {
          Double tfScore = tf.get(negFileId, word);
          if (tfScore == null) {
            tfScore = 0.0;
          }
          
          Double idfScore = idf.get(negFileId, word);
          if (idfScore == null) {
            idfScore = 0.0;
          }
          
          tfIdfOut.print(tfScore * idfScore);
        }
        tfIdfOut.println();
      }
    }
  }
}

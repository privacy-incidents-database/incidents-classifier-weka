package edu.ncsu.csc.privacyincidents.nlp;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.net.URISyntaxException;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Set;
import java.util.TreeMap;

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

public class FrrequentTermsComputer implements AutoCloseable {

  private StanfordCoreNLP mNlpPipeline;
  
  private StopWordsHandler mStopWordsHandler;

  private Map<Integer, File> mFilesMap = new HashMap<Integer, File>();

  // For TF, count terms in each document
  private Table<Integer, String, Integer> docTermCounts = HashBasedTable.create();

  public FrrequentTermsComputer() throws IOException, URISyntaxException {
    // Open NLP pipleline
    Properties nlpProps = new Properties();
    nlpProps.setProperty("annotators", "tokenize, ssplit, pos, lemma");
    mNlpPipeline = new StanfordCoreNLP(nlpProps);
    
    mStopWordsHandler = new StopWordsHandler("/english-stopwords.txt");
  }

  public void close() throws Exception {
    StanfordCoreNLP.clearAnnotatorPool();
  }

  public static void main(String[] args)
      throws FileNotFoundException, ClassNotFoundException, IOException, Exception {

    String topLevelDirs = args[0];
    String outCsvFilename = args[1];

    try (FrrequentTermsComputer tfIdfBldr = new FrrequentTermsComputer()) {

      tfIdfBldr.readFilenames(topLevelDirs);

      tfIdfBldr.readFilesAndUpdateCounts();
      
      tfIdfBldr.writeToCsv(outCsvFilename);
    }
  }

  private void readFilenames(String topLevelDirs) {
    String DIRNAME_SEPARATOR = ":";

    Set<File> files = new HashSet<File>();
    String[] topLevelDirsArray = topLevelDirs.split(DIRNAME_SEPARATOR);
    for (int i = 0; i < topLevelDirsArray.length; i++) {
      System.out.println(topLevelDirsArray[i]);
      files.addAll(listFilesForDirectory(new File(topLevelDirsArray[i])));
    }

    System.out.println("Number of files: " + files.size());

    int id = 0;
    for (File file : files) {
      mFilesMap.put(id++, file);
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
  
  private void readFilesAndUpdateCounts() throws IOException {
    for (Integer fileID : mFilesMap.keySet()) {
      File file = mFilesMap.get(fileID);
      updateCounts(fileID, file);
    }
  }

  private void updateCounts(Integer fileId, File file) throws IOException {
    byte[] encoded = Files.readAllBytes(file.toPath());
    String fileContents = new String(encoded);

    List<String> bagOfWords = getBagOfWords(fileContents);
    
    // getBagOfWords adds a null value between sentences
    bagOfWords.removeAll(Collections.singleton(null));

    for (String word : bagOfWords) {
      Integer wordCount = docTermCounts.get(fileId, word);
      if (wordCount == null) {
        docTermCounts.put(fileId, word, 1);
      } else {
        docTermCounts.put(fileId, word, wordCount + 1);
      }
    }
  }
  
  // TODO: This method is an exact copy of TfIdfBuilder.getBagOfWords(). Some
  // refactoring may help reduce code duplication.
  private List<String> getBagOfWords(String text) {
    List<String> words = new ArrayList<String>();

    Annotation annotation = new Annotation(text);
    mNlpPipeline.annotate(annotation);

    for (CoreMap sentence : annotation.get(CoreAnnotations.SentencesAnnotation.class)) {
      for (CoreLabel token : sentence.get(TokensAnnotation.class)) {
        // Discard words that only have punctuation, numerals
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
      words.add(null);
    }

    return words;
  }
  
  private void writeToCsv(String outCsvFilename) throws IOException {
    Map<Integer, List<String>> countsToWords = new TreeMap<Integer, List<String>>(
        Collections.reverseOrder());
    for (String word : docTermCounts.columnKeySet()) {
      Map<Integer, Integer> fileIdToWordCount = docTermCounts.column(word);
      Integer wordCount = 0;
      for (Integer fileWordCount : fileIdToWordCount.values()) {
        wordCount += fileWordCount;
      }
      List<String> wordsWithCurCount = countsToWords.get(wordCount);
      if (wordsWithCurCount == null) {
        wordsWithCurCount = new ArrayList<String>();
        countsToWords.put(wordCount, wordsWithCurCount);
      }
      wordsWithCurCount.add(word);
    }

    try (BufferedWriter writer = new BufferedWriter(new FileWriter(outCsvFilename))) {
      for (Integer count : countsToWords.keySet()) {
        writer.write(count + ": " + countsToWords.get(count) + "\n");
      }
    }
  }
}

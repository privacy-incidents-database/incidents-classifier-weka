package edu.ncsu.csc.privacyincidents.nlp;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.HashSet;
import java.util.Properties;
import java.util.Set;

import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.CoreAnnotations.LemmaAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.PartOfSpeechAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.util.CoreMap;

public class NLPTester implements AutoCloseable {

  private StanfordCoreNLP mNlpPipeline;
  
  private Stemmer mStemmer = new Stemmer();
  
  public NLPTester() {
    // Open NLP pipleline
    Properties nlpProps = new Properties();
    nlpProps.setProperty("annotators", "tokenize, ssplit, pos, lemma");
    mNlpPipeline = new StanfordCoreNLP(nlpProps);
  }
  
  @Override
  public void close() {
    StanfordCoreNLP.clearAnnotatorPool();
  }
  
  public void test(String text) {
    Set<String> allWords = new HashSet<String>();
    Set<String> lemmatizedWords = new HashSet<String>();
    Set<String> stemmedWords = new HashSet<String>();
    
    Annotation annotation = new Annotation(text);
    mNlpPipeline.annotate(annotation);

    for (CoreMap sentence : annotation.get(CoreAnnotations.SentencesAnnotation.class)) {
      for (CoreLabel token : sentence.get(TokensAnnotation.class)) {
        allWords.add(token.originalText());
        String pos = token.get(PartOfSpeechAnnotation.class);
        if (pos.startsWith("N") || pos.startsWith("J") || pos.startsWith("V")
            || pos.startsWith("R")) {
          String lemma = token.get(LemmaAnnotation.class);
          String stem = mStemmer.stem(token.originalText());
          System.out.println(token + " : " + pos + " : " + lemma + " : " + stem);
          lemmatizedWords.add(lemma);
          stemmedWords.add(stem);
        }
      }
    }

    System.out.println("Original count: " + allWords.size() + "; After lemma count: "
        + lemmatizedWords.size() + "; After stem count: " + stemmedWords.size());
  }
  
  public static void main(String[] args) throws IOException {
    String filename = args[0];
    
    byte[] encoded = Files.readAllBytes(Paths.get(filename));
    String fileContents = new String(encoded).toLowerCase();
    
    // String fileContents = "This bread is from a basket of breads";
    
    try (NLPTester nlpTester = new NLPTester()) {
      nlpTester.test(fileContents);
    }

  }

}

package edu.ncsu.csc.privacyincidents.classification;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.SequenceInputStream;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.Scanner;
import java.util.logging.Level;
import java.util.logging.Logger;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.LibSVM;
import weka.classifiers.functions.SMO;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.Range;
import edu.ncsu.csc.privacyincidents.classification.custom.KeywordBasedClassifier;

public class PrivacyClassifier {

  private static final Logger log = Logger.getLogger(PrivacyClassifier.class.getName());
  private String[] args = null;
  private Options options = new Options();

  // private String mKeywords;
  private List<String> mKeywords = new ArrayList<String>();
  
  private String[] mTrainingFiles;
  private String[] mTestingFiles;
  private String[] mCrossValidationFiles;
  
  private String outPredictionsFilename;
  
  // TODO: Throw an exception when an invalid classifier is used
  private static enum PrivacyClassifierName {
    KEYWORD_BASED, NAIVE_BAYES, SMO, LIB_SVM, RANDOM_FOREST
  };
  
  private PrivacyClassifierName mClassifierName;
  
  public PrivacyClassifier(String[] args) throws FileNotFoundException {
    this.args = args;
    options.addOption("h", "help", false, "show help");

    options.addOption("c", "classifier", true, "classifier to train");
    
    Option keywordOption = new Option("k", "keywordsFile", true,
        "name of the file containing keywords (one per line) for the NAIVE classifier");
    keywordOption.setArgs(1);
    options.addOption(keywordOption);
    
    Option trainOption = new Option("r", "train", true, "name of the arff file for training");
    trainOption.setArgs(10);
    options.addOption(trainOption);

    Option testOption = new Option("e", "test", true, "name of the arff file for testing");
    testOption.setArgs(10);
    options.addOption(testOption);

    Option cvOption = new Option("v", "crossvalidate", true,
        "name of the arff file for training and crossvalidation");
    cvOption.setArgs(10);
    options.addOption(cvOption);
    
    Option predictionsFilenameOption = new Option("o", "out", true,
        "name of the output file for storing predictions");
    options.addOption(predictionsFilenameOption);
    
    parse();
  }

  private void parse() throws FileNotFoundException {
    CommandLineParser parser = new DefaultParser();
    CommandLine cmd = null;
    try {
      cmd = parser.parse(options, args);

      if (cmd.hasOption("h")) {
        help();
      } 
      
      if (cmd.hasOption("c")) {
        mClassifierName = PrivacyClassifierName.valueOf(cmd.getOptionValue("c"));        
      } else {
        log.log(Level.SEVERE, "Must provide a classifier name");
        help();
      }
      
      /*
      if (mClassifierName == PrivacyClassifierName.KEYWORD_BASED) {
        if (cmd.hasOption("k")) {
          mKeywords = cmd.getOptionValue("k");
        } else {
          log.log(Level.SEVERE,
              "Must provide a pipe (|) separated list of keywords if using NAIVE classifier");
          help();            
        }
      }*/
      
      if (mClassifierName == PrivacyClassifierName.KEYWORD_BASED) {
        if (cmd.hasOption("k")) {
          String keywordsFilename = cmd.getOptionValue("k");
          try (Scanner s = new Scanner(new File(keywordsFilename))) {
            while (s.hasNext()) {
              String nextLine = s.next().trim().toLowerCase();
              if (!nextLine.isEmpty() && !nextLine.startsWith("#")) {
                mKeywords.add(nextLine);
              }
            }
          }
        } else {
          log.log(Level.SEVERE,
              "Must provide a keywords filename");
          help();            
        }
      }

      
      if (cmd.hasOption("v")) {
        mCrossValidationFiles = cmd.getOptionValues("v");
      } else if (cmd.hasOption("r")){
        mTrainingFiles = cmd.getOptionValues("r");
        if (cmd.hasOption("r")) {
          mTestingFiles = cmd.getOptionValues("e");
        } else {
          log.log(Level.SEVERE,
              "Provided training files but no testing files; use crossvalidation, instead");
          help();
        }
      }
      
      if (cmd.hasOption("o")) {
        outPredictionsFilename = cmd.getOptionValue("o");
      }
      
    } catch (ParseException e) {
      log.log(Level.SEVERE, "Failed to parse comand line arguments", e);
      help();
    }
  }

  private void help() {
    HelpFormatter formater = new HelpFormatter();
    formater.printHelp("PrivacyClassifier", options);
    System.exit(0);
  }
  
  private void run() throws Exception {
    if (mCrossValidationFiles != null) {
      runCrossValidation();
    } else if (mTrainingFiles != null) {
      runTrainingAndTesting();
    } else {
      log.log(Level.INFO, "Nothing to do");
    }
  }
  
  private void runCrossValidation() throws Exception {
    Instances data = readArff(mCrossValidationFiles);
    
    Classifier classifier = getClassifier(mClassifierName);
    
    Evaluation eval = new Evaluation(data);
    eval.crossValidateModel(classifier, data, 10, new Random(1));
    printEvalStatistics(eval);
  }
  
  private void printEvalStatistics(Evaluation eval) {
    System.out.println("Percent correct = " + eval.pctCorrect() + "; " + "Percent incorrect = "
        + eval.pctIncorrect());

    double[][] confusionMatrix = eval.confusionMatrix();
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 2; j++) {
        System.out.print(confusionMatrix[i][j] + "  ");
      }
      System.out.println();
    }
    
    double precisionSum = 0;
    double recallSum = 0;
    double fMeasureSum = 0;
    
    for (int i = 0; i < 2; i++) {
      System.out.println("Class " + (i + 1));
      System.out.println("Precision " + eval.precision(i));
      precisionSum += eval.precision(i);

      System.out.println("Recall " + eval.recall(i));
      recallSum += eval.recall(i);

      System.out.println("F-measure " + eval.fMeasure(i));
      fMeasureSum += eval.fMeasure(i);
    }

    System.out.println("Avg. Precision " + precisionSum / 2);
    System.out.println("Avg. Recall " + recallSum / 2);
    System.out.println("Avg. F-measure " + fMeasureSum / 2);
  }
  
  private double[] runTrainingAndTesting() throws Exception {
    Instances trainingData = readArff(mTrainingFiles);
    Instances testingData = readArff(mTestingFiles);
    
    Classifier classifier = getClassifier(mClassifierName);
    classifier.buildClassifier(trainingData);
    
    Evaluation eval = new Evaluation(trainingData);
    StringBuffer outBuffer = new StringBuffer();
    // Range attRange = new Range("1-2");
    Range attRange = null;
    Boolean printDist = new Boolean(false);
    double[] predictions = eval.evaluateModel(classifier, testingData, outBuffer, attRange,
        printDist);
    
    if (outPredictionsFilename == null || outPredictionsFilename.isEmpty()) {
      System.out.println(outBuffer.toString());
    } else {
      try (BufferedWriter writer = new BufferedWriter(new FileWriter(outPredictionsFilename))) {
        writer.append(outBuffer);
      }
    }
    
    printEvalStatistics(eval);
    
    return predictions;
  }
  
  public double runTrainingAndTesting(Instances trainingData, Instances testingData)
      throws Exception {
    Classifier classifier = getClassifier(mClassifierName);
    classifier.buildClassifier(trainingData);

    Evaluation eval = new Evaluation(trainingData);
    eval.evaluateModel(classifier, testingData);
    
    return eval.pctCorrect();
  }

  private Instances readArff(String[] arffFilenames) throws Exception {
    List<InputStream> inputStreams = new ArrayList<>();
    for (String arffFilename : arffFilenames) {
      FileInputStream fis = new FileInputStream(arffFilename);
      inputStreams.add(fis);
    }
    SequenceInputStream sis = new SequenceInputStream(Collections.enumeration(inputStreams));

    BufferedReader reader = new BufferedReader(new InputStreamReader(sis, Charset.forName("utf-8")));

    Instances data = new Instances(reader);

    reader.close();
    for (InputStream inputStream : inputStreams) {
      inputStream.close();
    }

    data.setClassIndex(data.numAttributes() - 1);

    System.out.println("# Attributes = " + data.numAttributes());
    System.out.println("# Instances = " + data.numInstances());
    System.out.println("# Classes = " + data.numClasses());

    return data;
  }
  
  private Classifier getClassifier(PrivacyClassifierName classifierName) throws Exception {
    switch (classifierName) {
    case KEYWORD_BASED:
      return new KeywordBasedClassifier(mKeywords);
    case NAIVE_BAYES:
      NaiveBayes nbClassifier = new NaiveBayes();
      return nbClassifier;
    case LIB_SVM:
      LibSVM svmClassifier = new LibSVM();
      svmClassifier
          .setOptions(weka.core.Utils
              .splitOptions("-S 0 -K 0 -D 3 -G 0.0 -R 0.0 -N 0.5 -M 40.0 -C 1.0 -E 0.001 -P 0.1 -model /Users/mpradeep/Software/weka-3-7-12 -seed 1"));
      return svmClassifier;
    case SMO:
      SMO smoClassifier = new weka.classifiers.functions.SMO();
      smoClassifier
          .setOptions(weka.core.Utils
              .splitOptions("-C 1.0 -L 0.0010 -P 1.0E-12 -N 0 -V -1 -W 1 -K \"weka.classifiers.functions.supportVector.PolyKernel -C 250007 -E 1.0\""));
      return smoClassifier;
    case RANDOM_FOREST:
      RandomForest rfClassifier = new RandomForest();
      rfClassifier.setMaxDepth(20);
      rfClassifier.setNumTrees(400);
      return rfClassifier;
    default:
      throw new IllegalArgumentException("Invalid classifier of name " + classifierName);
    }
  }

  public static void main(String[] args) throws Exception {
    PrivacyClassifier privacyClassifier = new PrivacyClassifier(args);
    privacyClassifier.run();
  }
}

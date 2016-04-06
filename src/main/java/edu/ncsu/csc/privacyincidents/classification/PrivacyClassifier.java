package edu.ncsu.csc.privacyincidents.classification;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.SequenceInputStream;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
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
import weka.classifiers.functions.LibSVM;
import weka.classifiers.functions.SMO;
import weka.core.Instances;
import weka.core.Range;
import edu.ncsu.csc.privacyincidents.util.StopWordsHandler;

public class PrivacyClassifier {

  private static final Logger log = Logger.getLogger(PrivacyClassifier.class.getName());
  private String[] args = null;
  private Options options = new Options();

  private String[] trainingFiles;
  private String[] testingFiles;
  private String[] crossValidationFiles;
  
  // TODO: Throw an exception when an invalid classifier is used
  private static enum PrivacyClassifierName {
    SMO, LibSVM
  };
  
  private PrivacyClassifierName mClassifierName;
  
  public PrivacyClassifier(String[] args) {
    this.args = args;
    options.addOption("h", "help", false, "show help");

    options.addOption("c", "classifier", true, "classifier to train");
    
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
    
    parse();
  }

  private void parse() {
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
      
      if (cmd.hasOption("v")) {
        crossValidationFiles = cmd.getOptionValues("v");
      } else if (cmd.hasOption("r")){
        trainingFiles = cmd.getOptionValues("r");
        if (cmd.hasOption("r")) {
          testingFiles = cmd.getOptionValues("e");
        } else {
          log.log(Level.SEVERE,
              "Provided training files but no testing files; use crossvalidation, instead");
          help();
        }
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
    if (crossValidationFiles != null) {
      runCrossValidation();
    } else if (trainingFiles != null) {
      runTrainingAndTesting();
    } else {
      log.log(Level.INFO, "Nothing to do");
    }
  }
  
  private void runCrossValidation() throws Exception {
    Instances data = readArff(crossValidationFiles);
    
    Classifier classifier = getClassifier(mClassifierName);
    
    Evaluation eval = new Evaluation(data);
    eval.crossValidateModel(classifier, data, 10, new Random(1));
    printEvalStatistics(eval);
  }
  
  private void printEvalStatistics(Evaluation eval) {
    System.out.println("Percent correct = " + eval.pctCorrect() + "; " + "Percent incorrect = "
        + eval.pctIncorrect());

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
    Instances trainingData = readArff(trainingFiles);
    Instances testingData = readArff(testingFiles);
    
    Classifier classifier = getClassifier(mClassifierName);
    classifier.buildClassifier(trainingData);
    
    Evaluation eval = new Evaluation(trainingData);
    StringBuffer outBuffer = new StringBuffer();
    // Range attRange = new Range("1-2");
    Range attRange = null;
    Boolean printDist = new Boolean(false);
    double[] predictions = eval.evaluateModel(classifier, testingData, outBuffer, attRange,
        printDist);
    System.out.println(outBuffer.toString());
    printEvalStatistics(eval);
    
    return predictions;
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

    Instances filteredData = StopWordsHandler.filterInstances(data);

    System.out.println("# Attributes = " + filteredData.numAttributes());
    System.out.println("# Instances = " + filteredData.numInstances());
    System.out.println("# Classes = " + filteredData.numClasses());

    return filteredData;
  }
  
  private Classifier getClassifier(PrivacyClassifierName classifierName) throws Exception {
    switch (classifierName) {
    case LibSVM:
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
    default:
      throw new IllegalArgumentException("Invalid classifier of name " + classifierName);
    }
  }

  public static void main(String[] args) throws Exception {
    PrivacyClassifier privacyClassifier = new PrivacyClassifier(args);
    privacyClassifier.run();
  }
}

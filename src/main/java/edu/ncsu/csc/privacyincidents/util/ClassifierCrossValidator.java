package edu.ncsu.csc.privacyincidents.util;

import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;

public class ClassifierCrossValidator {

  public static void crossValidate(Instances data, Classifier classifier, int numFolds)
      throws Exception {
    
    Evaluation eval = new Evaluation(data);

    eval.crossValidateModel(classifier, data, numFolds, new Random(1));

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
}

package edu.ncsu.csc.privacyincidents;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Random;

import weka.classifiers.Evaluation;
import weka.core.Instances;

public class IncidentClassifier {

  public static void main(String[] args) throws Exception {
    String arffFilename = args[0];

    BufferedReader reader = new BufferedReader(new FileReader(arffFilename));
    Instances data = new Instances(reader);
    reader.close();

    data.setClassIndex(data.numAttributes() - 1);

    // create new instance of scheme
    weka.classifiers.functions.SMO scheme = new weka.classifiers.functions.SMO();
    // set options
    scheme
        .setOptions(weka.core.Utils
            .splitOptions("-C 1.0 -L 0.0010 -P 1.0E-12 -N 0 -V -1 -W 1 -K \"weka.classifiers.functions.supportVector.PolyKernel -C 250007 -E 1.0\""));
        
    Evaluation eval = new Evaluation(data);
    
    eval.crossValidateModel(scheme, data, 10, new Random(1));
    
    System.out.println(eval.pctCorrect() + ", " + eval.pctIncorrect());
  }

}

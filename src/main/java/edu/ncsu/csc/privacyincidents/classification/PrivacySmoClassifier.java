package edu.ncsu.csc.privacyincidents.classification;

import edu.ncsu.csc.privacyincidents.util.ClassifierCrossValidator;
import edu.ncsu.csc.privacyincidents.util.IncidentsArffReader;
import weka.classifiers.functions.SMO;
import weka.core.Instances;

public class PrivacySmoClassifier {

  public static void main(String[] args) throws Exception {
    String arffFilename = args[0];
    Instances data = IncidentsArffReader.readArff(arffFilename);

    SMO smoClassifier = new weka.classifiers.functions.SMO();
    smoClassifier
        .setOptions(weka.core.Utils
            .splitOptions("-C 1.0 -L 0.0010 -P 1.0E-12 -N 0 -V -1 -W 1 -K \"weka.classifiers.functions.supportVector.PolyKernel -C 250007 -E 1.0\""));

    ClassifierCrossValidator.crossValidate(data, smoClassifier, 10);
  }
}

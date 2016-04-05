package edu.ncsu.csc.privacyincidents.classification;

import java.io.BufferedReader;
import java.io.FileReader;

import edu.ncsu.csc.privacyincidents.util.StopWordsHandler;
import weka.classifiers.functions.LibSVM;
import weka.core.Instances;

public class PrivacyLibSvmClassifier {

  public static void main(String[] args) throws Exception {
    String arffFilename = args[0];

    BufferedReader reader = new BufferedReader(new FileReader(arffFilename));
    Instances data = new Instances(reader);
    reader.close();

    data.setClassIndex(data.numAttributes() - 1);

    Instances filteredData = StopWordsHandler.filterInstances(data);
    
    LibSVM svmClassifier = new LibSVM();
    
    svmClassifier
        .setOptions(weka.core.Utils
            .splitOptions("-S 0 -K 0 -D 3 -G 0.0 -R 0.0 -N 0.5 -M 40.0 -C 1.0 -E 0.001 -P 0.1 -model /Users/mpradeep/Software/weka-3-7-12 -seed 1"));

    svmClassifier.buildClassifier(filteredData);
        
    System.out.println(svmClassifier.toString());
    
    System.out.println(svmClassifier.getWeights());
  }
}

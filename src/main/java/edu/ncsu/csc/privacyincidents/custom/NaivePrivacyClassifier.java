package edu.ncsu.csc.privacyincidents.custom;

import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

public class NaivePrivacyClassifier extends Classifier {

  private static final long serialVersionUID = 1478544817603231136L;
  
  private Instances mInstances;
  
  private int privacyAttIndex = -1; 
  
  @Override
  public void buildClassifier(Instances data) throws Exception {
    mInstances = new Instances(data);
    for (int i = 0; i < mInstances.numAttributes(); i++) {
      Attribute att = mInstances.attribute(i);
      if (att.name().equals("privaci")) {
        privacyAttIndex = i;
      }
    }
    
    if (privacyAttIndex == -1) {
      throw new  IllegalArgumentException("The input data does not contain a privacy attribute");
    }
  }

  @Override
  public double classifyInstance(Instance instance) {
    if (instance.value(instance.attribute(privacyAttIndex)) > 0) {
      return 1;
    }
    return 0;
  }
}

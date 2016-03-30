package edu.ncsu.csc.privacyincidents.custom;

import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

public class NaivePrivacySecurityClassifier extends Classifier {

  private static final long serialVersionUID = 1478544817603231136L;
  
  private Instances mInstances;
  
  private int privacyAttIndex = -1;
  private int securityAttIndex = -1;
  
  @Override
  public void buildClassifier(Instances data) throws Exception {
    mInstances = new Instances(data);
    for (int i = 0; i < mInstances.numAttributes(); i++) {
      Attribute att = mInstances.attribute(i);
      if (att.name().equals("privaci")) {
        privacyAttIndex = i;
      }
      if (att.name().equals("secur")) {
        securityAttIndex = i;
      }
    }
    
    if (privacyAttIndex == -1 || securityAttIndex == -1) {
      throw new IllegalArgumentException(
          "The input data does not contain a privacy or security attribute");
    }
  }

  @Override
  public double classifyInstance(Instance instance) {
    double privacyVal = instance.value(instance.attribute(privacyAttIndex));
    double securityVal = instance.value(instance.attribute(securityAttIndex));
    if (privacyVal == 0 && securityVal == 0) {
      if (Math.random() < 0.5) {
        return 0;
      } else {
        return 1;
      }
    }
    if (privacyVal > securityVal) {
      return 1;
    } else {
      return 0;
    }
  }
}

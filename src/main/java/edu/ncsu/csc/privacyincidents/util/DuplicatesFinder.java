package edu.ncsu.csc.privacyincidents.util;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.HashSet;
import java.util.Set;

public class DuplicatesFinder {
  public static Set<String> listFilesForDirectory(final File directory) {
    if (!directory.isDirectory()) {
      throw new IllegalArgumentException(directory.getName() + " is not a directory");
    }

    Set<String> fileNames = new HashSet<String>();

    for (final File fileEntry : directory.listFiles()) {
      if (fileEntry.isDirectory()) {
        fileNames.addAll(listFilesForDirectory(fileEntry));
      } else {
        fileNames.add(fileEntry.getName());
      }
    }
    return fileNames;
  }

  public static void main(String[] args)
      throws FileNotFoundException, ClassNotFoundException, IOException, Exception {

    String DIRNAME_SEPARATOR = ":";
    
    String positiveFilesTopLevelDirs = args[0]; // || separated
    String negativeFilesTopLevelDirs = args[1]; // || separated
    
    Set<String> positiveFiles = new HashSet<String>();
    String[] positveFilesTopLevelDirsArray = positiveFilesTopLevelDirs.split(DIRNAME_SEPARATOR);
    for (int i = 0; i < positveFilesTopLevelDirsArray.length; i++) {
      System.out.println(positveFilesTopLevelDirsArray[i]);
      positiveFiles.addAll(listFilesForDirectory(new File(positveFilesTopLevelDirsArray[i])));
    }
    
    System.out.println(positiveFiles.size());

    Set<String> negativeFiles = new HashSet<String>();
    String[] negativeFilesTopLevelDirsArray = negativeFilesTopLevelDirs.split(DIRNAME_SEPARATOR);
    for (int i = 0; i < negativeFilesTopLevelDirsArray.length; i++) {
      negativeFiles.addAll(listFilesForDirectory(new File(negativeFilesTopLevelDirsArray[i])));
    }
    
    System.out.println(negativeFiles.size());
    
    positiveFiles.retainAll(negativeFiles);
    
    System.out.println(positiveFiles.size());
    
    System.out.println(positiveFiles);
  }
}

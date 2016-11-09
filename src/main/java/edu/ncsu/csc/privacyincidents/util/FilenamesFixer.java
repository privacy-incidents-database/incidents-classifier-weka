package edu.ncsu.csc.privacyincidents.util;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * This is a one time utility for fixing a bug that was in TF-IDF builder. Just
 * left here in case we need to look back.
 * 
 * @author mpradeep
 *
 */
public class FilenamesFixer {
  private Map<Integer, File> mPositiveFilesMap = new HashMap<Integer, File>();

  private Map<Integer, File> mNegativeFilesMap = new HashMap<Integer, File>();

  private List<String> previousFiles = new ArrayList<String>();

  public static void main(String[] args) throws IOException {
    String posTopDirs = "/Users/mpradeep/git/incidents-collection/dat/positive";
    String negTopDirs = "/Users/mpradeep/git/incidents-collection/dat/negative";
    String prevFilename = "/Users/mpradeep/git/incidents-collection/dat/tf-idf.csv";
    String fileToFix = "/Users/mpradeep/git/incidents-collection/dat/old-false-positives.txt";
    String fixedFilename = "/Users/mpradeep/git/incidents-collection/dat/false-positives.txt";

    FilenamesFixer fixer = new FilenamesFixer();

    fixer.readFilenames(posTopDirs, negTopDirs);

    fixer.readPreviousFile(prevFilename);

    fixer.compareFiles();

    fixer.fixFiles(fileToFix, fixedFilename);
  }

  private void readFilenames(String positiveFilesTopLevelDirs, String negativeFilesTopLevelDirs) {
    String DIRNAME_SEPARATOR = ":";

    Set<File> positiveFiles = new HashSet<File>();
    String[] positveFilesTopLevelDirsArray = positiveFilesTopLevelDirs.split(DIRNAME_SEPARATOR);
    for (int i = 0; i < positveFilesTopLevelDirsArray.length; i++) {
      System.out.println(positveFilesTopLevelDirsArray[i]);
      positiveFiles.addAll(listFilesForDirectory(new File(positveFilesTopLevelDirsArray[i])));
    }

    System.out.println("Number of positive files: " + positiveFiles.size());

    Set<File> negativeFiles = new HashSet<File>();
    String[] negativeFilesTopLevelDirsArray = negativeFilesTopLevelDirs.split(DIRNAME_SEPARATOR);
    for (int i = 0; i < negativeFilesTopLevelDirsArray.length; i++) {
      negativeFiles.addAll(listFilesForDirectory(new File(negativeFilesTopLevelDirsArray[i])));
    }
    System.out.println("Number of negative files: " + negativeFiles.size());

    int id = 0;
    for (File positivefile : positiveFiles) {
      mPositiveFilesMap.put(id++, positivefile);
    }
    for (File negativefile : negativeFiles) {
      mNegativeFilesMap.put(id++, negativefile);
    }
  }

  private Set<File> listFilesForDirectory(final File directory) {
    if (!directory.isDirectory()) {
      throw new IllegalArgumentException(directory.getName() + " is not a directory");
    }

    Set<File> fileNames = new HashSet<File>();

    for (final File fileEntry : directory.listFiles()) {
      if (fileEntry.isDirectory()) {
        fileNames.addAll(listFilesForDirectory(fileEntry));
      } else {
        fileNames.add(fileEntry);
      }
    }
    return fileNames;
  }

  private void readPreviousFile(String filename) throws IOException {
    try (BufferedReader br = new BufferedReader(new FileReader(filename))) {
      String line;
      while ((line = br.readLine()) != null) {
        String shorterLine = line.substring(0, 500);
        String fileNameInside = shorterLine.split(",")[0];

        // Header
        if (fileNameInside.equals("fileName")) {
          continue;
        }

        if (!fileNameInside.trim().isEmpty()) {
          previousFiles.add(fileNameInside);
        }
      }
    }
  }

  private void compareFiles() {
    int listIndex = 0;
    for (Integer posFileId : mPositiveFilesMap.keySet()) {
      String prevFilaneme = previousFiles.get(listIndex);
      String curFilanem = "\"" + mPositiveFilesMap.get(posFileId).toString() + "\"";
      if (!prevFilaneme.equals(curFilanem)) {
        System.out.println(prevFilaneme + " : " + curFilanem);
      }
      listIndex++;
    }

    for (Integer negFileId : mNegativeFilesMap.keySet()) {
      String prevFilaneme = previousFiles.get(listIndex);
      String curFilanem = "\"" + mNegativeFilesMap.get(negFileId).toString() + "\"";
      if (!prevFilaneme.equals(curFilanem)) {
        System.out.println(prevFilaneme + " : " + curFilanem);
      }
      listIndex++;
    }
  }

  private void fixFiles(String oldFilename, String newFilename)
      throws FileNotFoundException, IOException {

    try (BufferedReader br = new BufferedReader(new FileReader(oldFilename));
        PrintWriter pw = new PrintWriter(new FileWriter(newFilename))) {
      String line;
      while ((line = br.readLine()) != null) {
        pw.println("\"" + mNegativeFilesMap.get(Integer.parseInt(line)) + "\"");
      }
    }
  }
}

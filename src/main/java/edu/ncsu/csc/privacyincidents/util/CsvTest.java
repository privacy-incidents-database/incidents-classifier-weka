package edu.ncsu.csc.privacyincidents.util;

import java.io.IOException;
import java.util.Map;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;

public class CsvTest {

  public static void main(String[] args) throws IOException {
    CSVParser parser = CSVParser.parse("\"Hello,World\",Hello,World!", CSVFormat.EXCEL);
    for (CSVRecord record : parser.getRecords()) {
      for (int i = 0; i < record.size(); i++)
      System.out.println(record.get(i));
    }
    /*Map<String, Integer> headerMap = parser.getHeaderMap();
    for (String headerVal : headerMap.keySet()) {
      System.out.println(headerVal + " : " + headerMap.get(headerVal));
    }*/
  }

}

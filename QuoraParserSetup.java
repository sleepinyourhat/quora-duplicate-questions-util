/*
Hastily assembled java utility to invoke the Stanford parser to get unlabeled binary parses for the Quora duplicate questions data.

To run:

Download Stanford CoreNLP and the Stanford Shift-Reduce parser model and place this file in the same directory as those .jar files. 
Edit the version numbers below and run.

javac QuoraParserSetup.java -cp ./stanford-srparser-2014-10-23-models.jar:./stanford-corenlp-3.5.1.jar:./stanford-corenlp-3.5.1-models.jar
java -cp ./stanford-srparser-2014-10-23-models.jar:./stanford-corenlp-3.5.1.jar:./stanford-corenlp-3.5.1-models.jar:. \
    QuoraParserSetup test_lines.txt ../quora_duplicate_questions.txt > ../quora_duplicate_questions_parsed.txt
 */

import java.util.*;
import java.util.Collection;
import java.util.List;
import java.io.StringReader;
import java.io.FileReader;
import java.io.BufferedReader;

import edu.stanford.nlp.parser.shiftreduce.ShiftReduceParser;
import edu.stanford.nlp.process.Tokenizer;
import edu.stanford.nlp.process.TokenizerFactory;
import edu.stanford.nlp.process.CoreLabelTokenFactory;
import edu.stanford.nlp.process.PTBTokenizer;
import edu.stanford.nlp.ling.*;
import edu.stanford.nlp.tagger.maxent.MaxentTagger;
import edu.stanford.nlp.trees.*;
import edu.stanford.nlp.parser.lexparser.*;
import edu.stanford.nlp.sentiment.CollapseUnaryTransformer;
import edu.stanford.nlp.util.Generics;


class QuoraParserSetup {
  
  public static void main(String[] args) {
    System.err.println("Loading parser...");
    ShiftReduceParser parser = ShiftReduceParser.loadModel("edu/stanford/nlp/models/srparser/englishSR.ser.gz");
    LexicalizedParser lp = LexicalizedParser.loadModel("edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz");

    System.err.println("Loading tagger...");
    MaxentTagger tagger = new MaxentTagger("edu/stanford/nlp/models/pos-tagger/english-left3words/english-left3words-distsim.tagger");

    System.err.println("Loading done.");

    TokenizerFactory<CoreLabel> tokenizerFactory =
      PTBTokenizer.factory(new CoreLabelTokenFactory(), "");
    CollapseUnaryTransformer transformer = new CollapseUnaryTransformer();
    TreeBinarizer binarizer = new TreeBinarizer(lp.getTLPParams().headFinder(), lp.treebankLanguagePack(),
                                                false, false, 0, false, false, 0.0, false, true, true);
    
    Map<String, String> parseCache = Generics.newHashMap();
    
    try {
      BufferedReader br = new BufferedReader(new FileReader(args[0]));
      String line = null;
      String[] inputRow = new String[6];
      String[] rowParses = new String[2];
      
      System.out.println("id\tqid1\tqid2\tquestion1\tquestion1_binary_parse\tquestion2\tquestion2_binary_parse\tis_duplicate");
      
      while ((line = br.readLine()) != null)
      {
        inputRow = line.split("\t", -1);
        if (inputRow.length < 6 || (inputRow[3].length() < 1)  || (inputRow[4].length() < 1) || inputRow[0].equals("id")) {
          continue;
        }     
        for (int i = 0; i < 2; i++) {
          if(parseCache.keySet().contains(inputRow[i + 3])) {
            rowParses[i] = parseCache.get(inputRow[i + 3]);
          } else {
            String input = inputRow[i + 3].replaceAll("₹", "Rs ");
            Tokenizer<CoreLabel> tokenizerInstance = tokenizerFactory.getTokenizer(new StringReader(input));
            List<CoreLabel> rawWords = tokenizerInstance.tokenize();
            List<TaggedWord> taggedWords = tagger.tagSentence(rawWords);
            Tree tree = parser.apply(taggedWords);
            Tree bin = binarizer.transformTree(tree);
            Tree collapsed = transformer.transformTree(bin);
            
            rowParses[i] = unlabeledPrint(collapsed);
            
            // Populate cache.
            parseCache.put(inputRow[i + 3], rowParses[i]);
          }
        }
        System.out.println(inputRow[0] + "\t" + inputRow[1] + "\t" + inputRow[2] + "\t" + inputRow[3] + "\t" + rowParses[0] + "\t" + inputRow[4] + "\t" + rowParses[1] + "\t" + inputRow[5]);
        
      }
    } catch (Exception e) {
      System.err.println(e);
    }
  }
  
  static String unlabeledPrint(Tree tree) {
    if (tree.isLeaf()) {
      String s = tree.nodeString();
      if (s.equals("(")) {
        s = "-LRB-";
      } else if (s.equals(")")) {
        s = "-RRB-";
      }
      return s;
    } else if (tree.isPreTerminal()) {
      for (Tree child : tree.children()) {
        return unlabeledPrint(child);
      }
      return "---";
    } else {
      String rv = "(";
      for (Tree child : tree.children()) {
        rv = rv + " " + unlabeledPrint(child);
      }
      return rv + " )";
    }
  }

  private QuoraParserSetup() {} // static methods only
}

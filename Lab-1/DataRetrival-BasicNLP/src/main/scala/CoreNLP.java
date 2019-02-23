import edu.stanford.nlp.simple.Document;
import edu.stanford.nlp.simple.Sentence;
import edu.stanford.nlp.util.Quadruple;

import java.util.Collection;
import java.util.Iterator;
import java.util.List;

public class CoreNLP {
    public static String getLemmatized(String question) {
        Document doc = new Document(question);
        String ret = "";
        for (Sentence sent : doc.sentences()) {
            for(int i = 0; i < sent.length()-1; i++){
                ret+= sent.lemma(i)+" ";
            }
            ret+= sent.lemma(sent.length()-1);
        }
        return ret;
    }
}

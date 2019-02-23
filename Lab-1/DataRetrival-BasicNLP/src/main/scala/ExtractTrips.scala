import org.apache.spark.{SparkConf, SparkContext}

import scala.io.Source
import java.io.{File, FileInputStream, FileOutputStream,PrintWriter}
/**
  * Created by Sai Sri Narne on 2-22-2019
  *
  * ExtractTrips obtains (question, answer, image_id) triplets for a set of keywords
  *
  */
object ExtractTrips {
  // Main function
  def main(args: Array[String]) {

    // INITIALIZE ENVIRONMENT
    // set directory of emulated hadoop home
    System.setProperty("hadoop.home.dir","C:\\winutils")
    // create new spark configuration
    val sparkConf = new SparkConf().setAppName("ExtractTrips").setMaster("local[*]")
    //create new spark context using spark configuration
    val sc=new SparkContext(sparkConf)

    // INITIALIZE VALUES
    val keywords = Array("lake", "pond", "fountain", "water", "pool")

    val questionFile = "Data/cocoqa/release/train/questions.txt"
    val answerFile = "Data/cocoqa/release/train/answers.txt"
    val imgIDFile = "Data/cocoqa/release/train/img_ids.txt"

    val questions = Source.fromFile(questionFile).getLines.toList
    val answers = Source.fromFile(answerFile).getLines.toList
    val imgIDs = Source.fromFile(imgIDFile).getLines.toList

    // Create Array to house triplets
    val trips = Array.ofDim[String](questions.length, 3)
    // Populate Array with triplets
    for( l <- 0 to (questions.length-1)){
      trips(l) = Array(questions(l), answers(l), imgIDs(l))
    }

    // Create RDD of triplets
    val triplets = sc.parallelize(trips)

    // Expand triplets to quintuplets
    // Result: <question, lemmatized_question, answer, lemmatized_answer, path_to_img>
    val lemmatizedTriplets = triplets.map(a => {
      (a(0),CoreNLP.getLemmatized(a(0)),a(1),CoreNLP.getLemmatized(a(1)),imgIDtoFilename(a(2)))
    })

    // Create Array to house pairs of <keyword, "triplet1 \n triplet2 \n ... \n tripletn">
    var tripsPerKeyword  = Array.ofDim[String](keywords.length, 2)

    // Filter triplets by keywords and insert pairs of
    // <keyword, "triplet1 \n ... \n tripletn"> into tripsPerKeyword
    for(i <- 0 to keywords.length-1){
      var curKey = keywords(i)
      tripsPerKeyword(i)(0) = curKey

      // Filter out triplets (RDD) which don't contain the given keyword
      // in either the lemmatized question or the lemmatized answer.
      var currentKeywordMatches = lemmatizedTriplets.filter(t => t._2.contains(curKey)||t._4.contains(curKey))
      // Create integer to hold the number of triplets found for the keyword
      var numTrips = 0
      // Create output directory for to the keyword
      new File("output/"+curKey).mkdirs

      // Transform RDD[(String1, String2, String3, String4, String5)] to RDD[StringFinal]
      // Resulting strings are of the form (comma delimited)
      // StringFinal = String1 + ", " + String2 + ", " +... + ", " + String5
      var t = currentKeywordMatches.map(a => {
        // While constructing this new RDD we transfer each image into the output directory
        var src = new File("C:\\Users\\Vishnu Sree Narne\\Downloads\\train2017\\train2017\\"+a._5)
        var dest = new File("output/"+curKey+"/"+a._5)
        new FileOutputStream(dest) getChannel() transferFrom(
          new FileInputStream(src) getChannel, 0, Long.MaxValue )
        // Then we simply conbine the Strings
        a._1+", "+a._2+", "+a._3+", "+a._4+", "+a._5+"\n"
      })

      // Create a varible to house the string "triplet1 \n ... \n tripletn"
      var curKeyTripsStr = ""
      // Populate the list of "triplet1 \n ... \n tripletn"
      t.collect().foreach(v => {
        numTrips+=1
        curKeyTripsStr+=v
      })

      tripsPerKeyword(i)(0) = curKey
      tripsPerKeyword(i)(1) = curKeyTripsStr

      // Write Triplets to a file
      var tripsFile = new File("output/"+curKey+"/"+curKey+".txt")
      var tripsWriter = new PrintWriter(tripsFile)
      tripsWriter.write(curKeyTripsStr)
      tripsWriter.close()

      // Write Stats to a file
      var statsFile = new File("output/"+curKey+"/"+curKey+"Stats.txt")
      var statsWriter = new PrintWriter(statsFile)
      var stats = "Total: " + numTrips
      statsWriter.write(stats)
      statsWriter.close()
    }
  }//End of Main function

  // img_id -> filename
  // Takes an img_id from Data/cocoqa/release/train/img_ids.txt as a String
  // Returns the filepath of the image in Data/
  def imgIDtoFilename(imgID: String): String = {
    var s = ""
    for(i <- 0 to 11-imgID.length()){
      s+="0"
    }
    s+=imgID+".jpg"
    s
  }//End of img_id -> filename

}
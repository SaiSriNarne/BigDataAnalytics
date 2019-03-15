import java.io.PrintStream

import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.mllib.feature.HashingTF
import org.apache.spark.{SparkConf, SparkContext}

object kM_Clustering {
  def main(args: Array[String]): Unit = {
    System.setProperty("hadoop.home.dir", "C:\\winutils")
    val sparkConf = new SparkConf().setAppName("SparkWordCount").setMaster("local[*]")
    val sc = new SparkContext(sparkConf)

    val features=sc.textFile("data\\wiki_data.txt")
      .map(f=>{
        val str=f.replaceAll(",","")
        val ff=f.split(" ")
        ff.drop(1).toSeq
      })
      val hashingTF=new HashingTF()

    val tf=hashingTF.transform(features)
    val kMeansModel=KMeans.train(tf,10,1000)

    val WSSSE = kMeansModel.computeCost(tf)
    println("Within Set Sum of Squared Errors = " + WSSSE)

    val clusters=kMeansModel.predict(tf)
    val out=new PrintStream("data\\results_wiki.csv")
    features.zip(clusters).collect().foreach(f=>{
       out.println(f._2+","+f._1.mkString(" "))
      })
  }

}

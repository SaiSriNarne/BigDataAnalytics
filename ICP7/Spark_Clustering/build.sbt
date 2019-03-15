
name := "Spark_Clustering"

version := "0.1"

scalaVersion := "2.11.8"

// https://mvnrepository.com/artifact/org.apache.spark/spark-core
libraryDependencies += "org.apache.spark" %% "spark-core" % "2.3.1"
// https://mvnrepository.com/artifact/org.apache.spark/spark-mllib
libraryDependencies += "org.apache.spark" %% "spark-mllib" % "2.3.1"

// https://mvnrepository.com/artifact/org.bytedeco.javacpp-presets/opencv
libraryDependencies += "org.bytedeco.javacpp-presets" % "opencv" % "4.0.1-1.4.4"

// https://mvnrepository.com/artifact/org.bytedeco/javacpp
libraryDependencies += "org.bytedeco" % "javacpp" % "1.4.4"

// https://mvnrepository.com/artifact/org.bytedeco/javacv
libraryDependencies += "org.bytedeco" % "javacv" % "1.4.4"


// https://mvnrepository.com/artifact/org.bytedeco.javacpp-presets/opencv-platform
libraryDependencies += "org.bytedeco.javacpp-presets" % "opencv-platform" % "4.0.1-1.4.4"

// https://mvnrepository.com/artifact/org.scalatest/scalatest
libraryDependencies += "org.scalatest" %% "scalatest" % "3.2.0-SNAP10" % Test

// https://mvnrepository.com/artifact/com.sun.jersey/jersey-core
libraryDependencies += "com.sun.jersey" % "jersey-core" % "1.19.4"

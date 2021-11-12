package logistic
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{Normalizer, StringIndexer, VectorAssembler}
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql._
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types._
import org.apache.spark.{SparkConf, SparkContext}

object diabetes {
  def main(args: Array[String]): Unit = {
    System.setProperty("hadoop.home.dir", "C:\\hadoop")
    val conf = new SparkConf().setMaster("local[*]").setAppName("Indian Diabetes")
    val sc = new SparkContext(conf)
    sc.setLogLevel("WARN")
    val spark = SparkSession.builder().getOrCreate()

    var df = spark.read.option("header", true)
      .csv("C:\\Users\\100_rabh\\IdeaProjects\\Mini_Project(ML)" +
        "\\src\\main\\scala\\logistic\\diabetes.csv")
    println("Dataframe of Indian Diabetes: ")
    df.show(10)
    println("Schema of each column: ")
    df.printSchema()

    val colNames = Array("Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
      "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome")
    for (colName <- colNames) {
              df = df.withColumn(colName, col(colName).cast("Double"))
      }

    println("Print Schema:")
    df.printSchema()

    // selecting particular columns which we want to scale for fitting in logistic regression
    val columns_to_scale = df.select( "Outcome","Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
      "Insulin", "BMI", "DiabetesPedigreeFunction", "Age").toDF( "Outcome","Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
      "Insulin", "BMI", "DiabetesPedigreeFunction", "Age")
    println("Columns to scale: ")
    columns_to_scale.show(5)

    // making features columns which will contains all the selected columns
    val assembler = new VectorAssembler()
      .setInputCols(Array("Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"))
      .setOutputCol("features")

    val df1 = assembler.transform(columns_to_scale)
    println("Separate features column which contains the remaining columns: ")
    df1.show(10)

    // min max Scaler
    val normalizer = new Normalizer()
      .setInputCol("features")
      .setOutputCol("normFeatures")
      .setP(2.0)
      .transform(df1)

    println("Normalized features column into normFeatures: ")
    normalizer.show(10)

    // Split the data into training and test sets (30% held out for testing).
    val Array(training, test) = normalizer.randomSplit(Array(0.7, 0.3))

    // Applying linear regression model
    val lr = new LogisticRegression()
      .setLabelCol("Outcome")
      .setFeaturesCol("normFeatures")
      .setMaxIter(10)
      .setRegParam(0.3)
      .setElasticNetParam(0.8)

    // Fit the model
    val lrModel = lr.fit(training)

    // print the output column and the input column
    println("Dataframe with input column and output column:")
    lrModel.transform(test)
      .select("features", "normFeatures", "Outcome", "prediction")
      .show(10)

    // Print the coefficients and intercept for linear regression
    println(s"Coefficients: ${lrModel.coefficients} \nIntercept: ${lrModel.intercept}")

    // Summarize the model over the training set and print out some metrics
    val trainingSummary = lrModel.binarySummary
    println(s"Total Iterations: ${trainingSummary.totalIterations}")
    println(s"Objective History: [${trainingSummary.objectiveHistory.mkString(",")}]")
    println(s"Accuracy: ${trainingSummary.accuracy}")
    println(s"False Positive Rate: ${trainingSummary.weightedFalsePositiveRate}")
    println(s"True Positive Rate: ${trainingSummary.weightedTruePositiveRate}")
    println(s"F Measure: ${trainingSummary.weightedFMeasure}")
    println(s"Precision: ${trainingSummary.weightedPrecision}")
    println(s"Recall: ${trainingSummary.weightedRecall}")
  }
}

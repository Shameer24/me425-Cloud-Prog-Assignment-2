import os
import sys
import boto3
from pyspark.sql import SparkSession
from pyspark.sql.types import FloatType, IntegerType
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder


# AWS Credentials Setup
def initialize_s3_client():
    access_key = os.getenv("ACCESSKey")
    secret_key = os.getenv("SECRETKey")
    return boto3.client("s3")

# Download Files from S3
def download_directory(bucket_name, folder_prefix, local_path):
    s3 = initialize_s3_client()
    paginator = s3.get_paginator("list_objects_v2")

    for page in paginator.paginate(Bucket=bucket_name, Prefix=folder_prefix):
        for obj in page.get("Contents", []):
            file_path = obj["Key"][len(folder_prefix):]
            local_file = os.path.join(local_path, file_path)
            os.makedirs(os.path.dirname(local_file), exist_ok=True)
            s3.download_file(bucket_name, obj["Key"], local_file)
            print(f"Downloaded: {obj['Key']} to {local_file}")


# Extract and Classify Columns
def identify_columns(dataframe, category_threshold=10, cardinality_threshold=20):
    categorical_columns, numerical_columns, high_cardinality_columns = [], [], []

    for column in dataframe.schema.fields:
        unique_values = dataframe.select(column.name).distinct().count()

        if str(column.dataType) == "StringType":
            if unique_values > cardinality_threshold:
                high_cardinality_columns.append(column.name)
            else:
                categorical_columns.append(column.name)
        elif unique_values < category_threshold:
            categorical_columns.append(column.name)
        else:
            numerical_columns.append(column.name)

    return categorical_columns, numerical_columns, high_cardinality_columns


# Configure Model Parameters
def setup_model_parameters(label_column):
    logistic_regression = LogisticRegression(featuresCol="scaled_features", labelCol=label_column)
    decision_tree = DecisionTreeClassifier(featuresCol="scaled_features", labelCol=label_column)

    logistic_params = ParamGridBuilder() \
        .addGrid(logistic_regression.maxIter, [50, 100]) \
        .addGrid(logistic_regression.regParam, [0.01, 0.1, 0.5]) \
        .addGrid(logistic_regression.elasticNetParam, [0.0, 0.5, 1.0]) \
        .build()

    return [
        ("LogisticRegression", logistic_regression, logistic_params)
    ]


# Evaluate Models
def evaluate_and_train_models(training_data, validation_data, features, label):
    assembler = VectorAssembler(inputCols=features, outputCol="features")
    scaler = StandardScaler(inputCol="features", outputCol="scaled_features")
    evaluator = MulticlassClassificationEvaluator(labelCol=label, metricName="f1")

    best_model = None
    highest_score = 0

    for model_name, model_instance, param_grid in setup_model_parameters(label):
        pipeline = Pipeline(stages=[assembler, scaler, model_instance])
        cross_validator = CrossValidator(
            estimator=pipeline,
            estimatorParamMaps=param_grid,
            evaluator=evaluator,
            numFolds=5
        )
        trained_model = cross_validator.fit(training_data)
        f1_score = evaluator.evaluate(trained_model.transform(validation_data))

        if f1_score > highest_score:
            highest_score = f1_score
            best_model = trained_model.bestModel

        print(f"{model_name} F1 Score: {f1_score:.2f}")

    return best_model


# Load Data from S3 and Apply Transformations
def load_data_from_s3(s3_key, spark_session, transform_function):
    s3_client = initialize_s3_client()
    response = s3_client.get_object(Bucket='cloud-assignment-2-me425', Key=s3_key)
    data_content = response['Body'].read().decode('utf-8').replace('"', '')
    data_rows = [tuple(row.split(';')) for row in data_content.strip().split('\r\n') if row]
    header = list(data_rows.pop(0))
    dataframe = spark_session.createDataFrame(data_rows, header)
    return transform_function(dataframe)

# Data Cleaning and Preparation
def transform_data(dataframe):
    float_columns = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar",
        "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
        "pH", "sulphates", "alcohol" ]

    for col in float_columns:
        dataframe = dataframe.withColumn(col, dataframe[col].cast(FloatType()))

    dataframe = dataframe.withColumn("quality", dataframe["quality"].cast(IntegerType()))
    return dataframe


# Predict New Data
def predict_on_new_data(test_key, spark_session, trained_model):
    test_data = load_data_from_s3(test_key, spark_session, transform_data)
    predictions = trained_model.transform(test_data)
    predictions.show()

    evaluatorF1 = MulticlassClassificationEvaluator(labelCol="quality", predictionCol="prediction", metricName="f1")
    f1Score = evaluatorF1.evaluate(predictions)
    print(f"f1Score {f1Score:.2f}")

    evaluator = MulticlassClassificationEvaluator(labelCol="quality", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    print(f"Prediction Accuracy: {accuracy:.2f}")


if __name__ == "__main__":
    spark = SparkSession.builder \
        .appName("Wine Quality Model Pipeline") \
        .config("spark.jars", "hadoop-aws-3.0.0.jar,aws-java-sdk-1.11.375.jar") \
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
        .getOrCreate()

    spark._jsc.hadoopConfiguration().set("fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
    spark._jsc.hadoopConfiguration().set("fs.s3a.access.key", os.getenv("ACCESSKey"))
    spark._jsc.hadoopConfiguration().set("fs.s3a.secret.key", os.getenv("SECRETKey"))

    train_key = 'TrainingDataset.csv'
    validation_key = 'ValidationDataset.csv'
    model_directory = 's3://cloud-assignment-2-me425/answermodel'

    train_df = load_data_from_s3(train_key, spark, transform_data)
    validation_df = load_data_from_s3(validation_key, spark, transform_data)

    categories, numericals, _ = identify_columns(train_df)
    feature_columns = [col for col in numericals if col != "quality"]

    if '--train' in sys.argv:
        best_trained_model = evaluate_and_train_models(train_df, validation_df, feature_columns, "quality")
        best_trained_model.write().overwrite().save(model_directory)

    if '--predict' in sys.argv:
        download_directory('cloud-assignment-2-me425', 'answermodel/', '/home/hadoop/answermodel')
        loaded_model = PipelineModel.load('/home/hadoop/answermodel')
        predict_on_new_data(validation_key, spark, loaded_model)

    spark.stop()

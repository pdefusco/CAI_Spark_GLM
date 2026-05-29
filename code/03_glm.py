#****************************************************************************
# (C) Cloudera, Inc. 2020-2026
#  All rights reserved.
#
#  Applicable Open Source License: GNU Affero General Public License v3.0
#
#  NOTE: Cloudera open source products are modular software products
#  made up of hundreds of individual components, each of which was
#  individually copyrighted.  Each Cloudera open source product is a
#  collective work under U.S. Copyright Law. Your license to use the
#  collective work is as provided in your written agreement with
#  Cloudera.  Used apart from the collective work, this file is
#  licensed for your use pursuant to the open source license
#  identified above.
#
#  This code is provided to you pursuant a written agreement with
#  (i) Cloudera, Inc. or (ii) a third-party authorized to distribute
#  this code. If you do not have a written agreement with Cloudera nor
#  with an authorized and properly licensed third party, you do not
#  have any rights to access nor to use this code.
#
#  Absent a written agreement with Cloudera, Inc. (“Cloudera”) to the
#  contrary, A) CLOUDERA PROVIDES THIS CODE TO YOU WITHOUT WARRANTIES OF ANY
#  KIND; (B) CLOUDERA DISCLAIMS ANY AND ALL EXPRESS AND IMPLIED
#  WARRANTIES WITH RESPECT TO THIS CODE, INCLUDING BUT NOT LIMITED TO
#  IMPLIED WARRANTIES OF TITLE, NON-INFRINGEMENT, MERCHANTABILITY AND
#  FITNESS FOR A PARTICULAR PURPOSE; (C) CLOUDERA IS NOT LIABLE TO YOU,
#  AND WILL NOT DEFEND, INDEMNIFY, NOR HOLD YOU HARMLESS FOR ANY CLAIMS
#  ARISING FROM OR RELATED TO THE CODE; AND (D)WITH RESPECT TO YOUR EXERCISE
#  OF ANY RIGHTS GRANTED TO YOU FOR THE CODE, CLOUDERA IS NOT LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, PUNITIVE OR
#  CONSEQUENTIAL DAMAGES INCLUDING, BUT NOT LIMITED TO, DAMAGES
#  RELATED TO LOST REVENUE, LOST PROFITS, LOSS OF INCOME, LOSS OF
#  BUSINESS ADVANTAGE OR UNAVAILABILITY, OR LOSS OR CORRUPTION OF
#  DATA.
#
# #  Author(s): Paul de Fusco
#***************************************************************************/

import mlflow
import os
import time
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    VectorAssembler,
    StandardScaler,
    FeatureHasher
)
from pyspark.ml.regression import GeneralizedLinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

import cml.data_v1 as cmldata


class FraudPoissonTrainer:

    def __init__(self, username, dbname, connection_name):
        self.username = username
        self.dbname = dbname
        self.connection_name = connection_name

    ##########################################################################
    # Timer Utility
    ##########################################################################
    def timer(self, name):
        """
        Context manager for timing code blocks
        """

        class TimerContext:
            def __enter__(self_inner):
                self_inner.start = time.time()
                print(f"\n[TIMER START] {name}")
                return self_inner

            def __exit__(self_inner, exc_type, exc_val, exc_tb):
                duration = time.time() - self_inner.start
                print(f"[TIMER END] {name} -> {duration:.2f} seconds\n")
                mlflow.log_param(f"Timer {name}", duration)

        return TimerContext()

    ##########################################################################
    # Spark Session
    ##########################################################################
    def createSparkConnection(self):
        from pyspark import SparkContext

        SparkContext.setSystemProperty("spark.executor.cores", "4")
        SparkContext.setSystemProperty("spark.executor.memory", "16g")

        conn = cmldata.get_connection(self.connection_name)
        return conn.get_spark_session()

    ##########################################################################
    # Load Data
    ##########################################################################
    def loadData(self, spark):
        table_name = f"{self.dbname}.transactions_{self.username}"
        print(f"Loading: {table_name}")
        return spark.read.format("iceberg").load(table_name)

    ##########################################################################
    # Prepare Data
    ##########################################################################
    def prepareData(self, df):
        df = df.dropna()
        df = df.withColumn("label", F.col("fraud_trx").cast("double"))
        df = df.filter(F.col("label") >= 0)
        return df

    ##########################################################################
    # Build Pipeline
    ##########################################################################
    def buildPipeline(self):

        numeric_features = [
            "age","credit_card_balance","bank_account_balance","mortgage_balance",
            "sec_bank_account_balance","savings_account_balance",
            "sec_savings_account_balance","total_est_nworth",
            "primary_loan_balance","secondary_loan_balance","uni_loan_balance",
            "longitude","latitude","transaction_amount","customer_score"
        ]

        categorical_features = [
            "customer_segment","account_type","transaction_type",
            "merchant_category","state","employment_status",
            "device_type","payment_channel","risk_region","card_network"
        ]

        hasher = FeatureHasher(
            inputCols=numeric_features + categorical_features,
            outputCol="hashed_features",
            numFeatures=2**12
        )

        scaler = StandardScaler(
            inputCol="hashed_features",
            outputCol="features",
            withStd=True,
            withMean=False
        )

        glm = GeneralizedLinearRegression(
            family="poisson",
            link="log",
            featuresCol="features",
            labelCol="label",
            predictionCol="prediction",
            maxIter=25,
            regParam=0.01,
            tol=1e-6
        )

        return Pipeline(stages=[hasher, scaler, glm])

    ##########################################################################
    # Train
    ##########################################################################
    def trainModel(self, pipeline, train_df):
        return pipeline.fit(train_df)

    ##########################################################################
    # Evaluate
    ##########################################################################
    def evaluateModel(self, model, test_df):

        predictions = model.transform(test_df)

        print("Predictions sample")
        predictions.select("label", "prediction").show(20, truncate=False)

        rmse = RegressionEvaluator(
            labelCol="label", predictionCol="prediction", metricName="rmse"
        ).evaluate(predictions)

        mae = RegressionEvaluator(
            labelCol="label", predictionCol="prediction", metricName="mae"
        ).evaluate(predictions)

        r2 = RegressionEvaluator(
            labelCol="label", predictionCol="prediction", metricName="r2"
        ).evaluate(predictions)

        print(f"RMSE: {rmse}")
        print(f"MAE:  {mae}")
        print(f"R2:   {r2}")

        mlflow.log_param("RMSE", rmse)
        mlflow.log_param("MAE", mae)
        mlflow.log_param("R2", r2)

        glm_model = model.stages[-1]
        print("\nGLM Coefficients")
        print(glm_model.coefficients)

        print("\nIntercept")
        print(glm_model.intercept)

    ##########################################################################
    # Run
    ##########################################################################
    def run(self):

        EXPERIMENT_NAME = "03_glm"
        mlflow.set_experiment(EXPERIMENT_NAME)

        with self.timer("FULL GLM PIPELINE"):
            spark = self.createSparkConnection()

            with self.timer("LOAD DATA"):
                df = self.loadData(spark)
                print("Row count:", df.count())
                df.printSchema()

            with self.timer("PREPARE DATA"):
                df = self.prepareData(df)

            train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

            print(f"Train: {train_df.count()}")
            print(f"Test: {test_df.count()}")

            pipeline = self.buildPipeline()

            with self.timer("MODEL TRAINING"):
                model = self.trainModel(pipeline, train_df)

            with self.timer("MODEL EVALUATION"):
                self.evaluateModel(model, test_df)

            print("Poisson GLM Training Complete")


##############################################################################
# MAIN
##############################################################################

def main():

    USERNAME = os.environ["PROJECT_OWNER"]
    DBNAME = os.environ["DBNAME_PREFIX"]
    CONNECTION_NAME = os.environ["SPARK_CONNECTION_NAME"]

    trainer = FraudPoissonTrainer(USERNAME, DBNAME, CONNECTION_NAME)
    trainer.run()


if __name__ == "__main__":
    main()

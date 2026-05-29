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

import os
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
    # Spark Session
    ##########################################################################

    def createSparkConnection(self):

        from pyspark import SparkContext

        SparkContext.setSystemProperty("spark.executor.cores", "2")
        SparkContext.setSystemProperty("spark.executor.memory", "4g")

        conn = cmldata.get_connection(self.connection_name)
        spark = conn.get_spark_session()

        return spark

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

        df = df.withColumn(
            "label",
            F.col("fraud_trx").cast("double")
        )

        df = df.filter(F.col("label") >= 0)

        return df

    ##########################################################################
    # Build Pipeline (Feature Hashing Version)
    ##########################################################################

    def buildPipeline(self):

        ######################################################################
        # Numeric Features
        ######################################################################

        numeric_features = [
            "age",
            "credit_card_balance",
            "bank_account_balance",
            "mortgage_balance",
            "sec_bank_account_balance",
            "savings_account_balance",
            "sec_savings_account_balance",
            "total_est_nworth",
            "primary_loan_balance",
            "secondary_loan_balance",
            "uni_loan_balance",
            "longitude",
            "latitude",
            "transaction_amount",
            "customer_score"
        ]

        ######################################################################
        # Categorical Features (HIGH CARDINALITY SAFE)
        ######################################################################

        categorical_features = [
            "customer_segment",
            "account_type",
            "transaction_type",
            "merchant_category",
            "state",
            "employment_status",
            "device_type",
            "payment_channel",
            "risk_region",
            "card_network"
        ]

        ######################################################################
        # Feature Hasher (replaces indexing + OHE)
        ######################################################################

        hasher = FeatureHasher(
            inputCols=numeric_features + categorical_features,
            outputCol="hashed_features",
            numFeatures=2**18
        )

        ######################################################################
        # Scaling (IMPORTANT: no mean centering for sparse vectors)
        ######################################################################

        scaler = StandardScaler(
            inputCol="hashed_features",
            outputCol="features",
            withStd=True,
            withMean=False
        )

        ######################################################################
        # Poisson GLM
        ######################################################################

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

        ######################################################################
        # Pipeline
        ######################################################################

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

        predictions.select("label", "prediction").show(20, truncate=False)

        rmse = RegressionEvaluator(
            labelCol="label",
            predictionCol="prediction",
            metricName="rmse"
        ).evaluate(predictions)

        mae = RegressionEvaluator(
            labelCol="label",
            predictionCol="prediction",
            metricName="mae"
        ).evaluate(predictions)

        r2 = RegressionEvaluator(
            labelCol="label",
            predictionCol="prediction",
            metricName="r2"
        ).evaluate(predictions)

        print(f"RMSE: {rmse}")
        print(f"MAE:  {mae}")
        print(f"R2:   {r2}")

        glm_model = model.stages[-1]

        print("\nGLM Coefficients")
        print(glm_model.coefficients)

        print("\nIntercept")
        print(glm_model.intercept)

    ##########################################################################
    # Run
    ##########################################################################

    def run(self):

        spark = self.createSparkConnection()

        df = self.loadData(spark)

        print("Dataset count:", df.count())
        df.printSchema()

        df = self.prepareData(df)

        train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

        print("Train:", train_df.count())
        print("Test:", test_df.count())

        pipeline = self.buildPipeline()

        model = self.trainModel(pipeline, train_df)

        self.evaluateModel(model, test_df)

        print("Training complete.")


##############################################################################
# Main
##############################################################################

def main():

    USERNAME = os.environ["PROJECT_OWNER"]
    DBNAME = os.environ["DBNAME_PREFIX"]
    CONNECTION_NAME = os.environ["SPARK_CONNECTION_NAME"]

    trainer = FraudPoissonTrainer(USERNAME, DBNAME, CONNECTION_NAME)
    trainer.run()


if __name__ == "__main__":
    main()

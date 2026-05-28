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
import numpy as np
import pandas as pd
from datetime import datetime
from pyspark.sql.types import IntegerType, FloatType
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import rand
import dbldatagen as dg
import dbldatagen.distributions as dist
from dbldatagen import FakerTextFactory, DataGenerator, fakerText
from faker.providers import bank, credit_card, currency

import cml.data_v1 as cmldata


class BankDataGen:

    """Class to Generate Banking Data"""

    def __init__(self, username, dbname, connectionName):

        self.username = username
        self.dbname = dbname
        self.connectionName = connectionName

    ##########################################################################
    # Generate Synthetic Dataset
    ##########################################################################

    def dataGen(
        self,
        spark,
        shuffle_partitions_requested=5,
        partitions_requested=2,
        data_rows=10000
    ):

        FakerTextUS = FakerTextFactory(
            locale=['en_US'],
            providers=[bank]
        )

        spark.conf.set(
            "spark.sql.shuffle.partitions",
            shuffle_partitions_requested
        )

        ######################################################################
        # Data Specification
        ######################################################################

        fakerDataspec = (

            DataGenerator(
                spark,
                rows=data_rows,
                partitions=partitions_requested
            )

            ##################################################################
            # Numeric Features
            ##################################################################

            .withColumn("age", "float", minValue=18, maxValue=90, random=True)
            .withColumn("credit_card_balance", "float", minValue=100, maxValue=30000, random=True)
            .withColumn("bank_account_balance", "float", minValue=0.01, maxValue=100000, random=True)
            .withColumn("mortgage_balance", "float", minValue=0.01, maxValue=1000000, random=True)
            .withColumn("sec_bank_account_balance", "float", minValue=0.01, maxValue=100000, random=True)
            .withColumn("savings_account_balance", "float", minValue=0.01, maxValue=500000, random=True)
            .withColumn("sec_savings_account_balance", "float", minValue=0.01, maxValue=500000, random=True)
            .withColumn("total_est_nworth", "float", minValue=10000, maxValue=500000, random=True)
            .withColumn("primary_loan_balance", "float", minValue=0.01, maxValue=5000, random=True)
            .withColumn("secondary_loan_balance", "float", minValue=0.01, maxValue=500000, random=True)
            .withColumn("uni_loan_balance", "float", minValue=0.01, maxValue=10000, random=True)
            .withColumn("longitude", "float", minValue=-180, maxValue=180, random=True)
            .withColumn("latitude", "float", minValue=-90, maxValue=90, random=True)
            .withColumn("transaction_amount", "float", minValue=0.01, maxValue=30000, random=True)

            ##################################################################
            # Label
            ##################################################################

            .withColumn("fraud_trx", "string",
                        values=["0", "1"],
                        weights=[6, 4],
                        random=True)

            ##################################################################
            # Categorical Features
            ##################################################################

            .withColumn("customer_segment", "string",
                        values=["retail", "private", "wealth", "commercial"],
                        weights=[60, 20, 10, 10],
                        random=True)

            .withColumn("account_type", "string",
                        values=["checking", "savings", "brokerage", "credit"],
                        weights=[50, 30, 10, 10],
                        random=True)

            .withColumn("transaction_type", "string",
                        values=["purchase", "wire", "withdrawal", "deposit", "transfer"],
                        weights=[50, 10, 10, 15, 15],
                        random=True)

            .withColumn("merchant_category", "string",
                        values=["retail", "travel", "restaurant", "electronics", "crypto", "gaming"],
                        weights=[35, 15, 20, 15, 5, 10],
                        random=True)

            .withColumn("state", "string",
                        values=["CA", "TX", "FL", "NY", "NV", "WA", "AZ"],
                        random=True)

            .withColumn("employment_status", "string",
                        values=["employed", "self_employed", "unemployed", "retired", "student"],
                        weights=[55, 15, 10, 10, 10],
                        random=True)

            .withColumn("device_type", "string",
                        values=["mobile", "desktop", "tablet"],
                        weights=[70, 25, 5],
                        random=True)

            .withColumn("payment_channel", "string",
                        values=["online", "branch", "atm", "pos"],
                        weights=[40, 10, 10, 40],
                        random=True)

            .withColumn("risk_region", "string",
                        values=["low", "medium", "high"],
                        weights=[60, 30, 10],
                        random=True)

            .withColumn("card_network", "string",
                        values=["visa", "mastercard", "amex", "discover"],
                        weights=[45, 35, 10, 10],
                        random=True)

        )

        ######################################################################
        # Build DataFrame
        ######################################################################

        df = fakerDataspec.build()

        ######################################################################
        # Feature Engineering
        ######################################################################

        df = df.withColumn(
            "fraud_trx",
            F.col("fraud_trx").cast(IntegerType())
        )

        df = df.withColumn(
            "customer_score",
            F.when(rand() < 0.20, rand())
             .otherwise(
                 F.col("fraud_trx")
                 * F.col("age")
                 * F.col("mortgage_balance")
             )
        )

        ######################################################################
        # Add Fraud Correlations
        ######################################################################

        df = df.withColumn(
            "customer_score",
            F.when(
                F.col("merchant_category") == "crypto",
                F.col("customer_score") * 1.5
            ).otherwise(F.col("customer_score"))
        )

        df = df.withColumn(
            "customer_score",
            F.when(
                F.col("risk_region") == "high",
                F.col("customer_score") * 1.25
            ).otherwise(F.col("customer_score"))
        )

        df = df.withColumn(
            "customer_score",
            F.col("customer_score").cast(FloatType())
        )

        return df

    ##########################################################################
    # Create Spark Connection
    ##########################################################################

    def createSparkConnection(self):

        from pyspark import SparkContext

        SparkContext.setSystemProperty('spark.executor.cores', '2')
        SparkContext.setSystemProperty('spark.executor.memory', '4g')

        conn = cmldata.get_connection(self.connectionName)
        spark = conn.get_spark_session()

        return spark

    ##########################################################################
    # Create Database
    ##########################################################################

    def createDatabase(self, spark):

        spark.sql(f"CREATE DATABASE IF NOT EXISTS {self.dbname}")

        print(f"SHOW DATABASES LIKE '{self.dbname}'")
        spark.sql(f"SHOW DATABASES LIKE '{self.dbname}'").show()

    ##########################################################################
    # Create Iceberg Table
    ##########################################################################

    def createOrReplace(self, df):

        (
            df.writeTo(f"{self.dbname}.transactions_{self.username}")
              .using("iceberg")
              .tableProperty("write.format.default", "parquet")
              .createOrReplace()
        )

    ##########################################################################
    # Validate Table
    ##########################################################################

    def validateTable(self, spark):

        print(f"SHOW TABLES FROM '{self.dbname}'")

        spark.sql(f"SHOW TABLES FROM {self.dbname}").show()

        print("SHOW ICEBERG METADATA")

        (
            spark.read.format("iceberg")
                 .load(f"{self.dbname}.transactions_{self.username}.snapshots")
                 .show(truncate=False)
        )


##############################################################################
# Main
##############################################################################

def main():

    USERNAME = os.environ["PROJECT_OWNER"]
    DBNAME = os.environ["DBNAME_PREFIX"]
    CONNECTION_NAME = os.environ["SPARK_CONNECTION_NAME"]

    ##########################################################################
    # Instantiate Generator
    ##########################################################################

    bdg = BankDataGen(
        USERNAME,
        DBNAME,
        CONNECTION_NAME
    )

    ##########################################################################
    # Create Spark Session
    ##########################################################################

    spark = bdg.createSparkConnection()

    ##########################################################################
    # Generate Dataset
    ##########################################################################

    df = bdg.dataGen(spark)

    ##########################################################################
    # Create Database
    ##########################################################################

    bdg.createDatabase(spark)

    ##########################################################################
    # Create Iceberg Table
    ##########################################################################

    bdg.createOrReplace(df)

    ##########################################################################
    # Validate
    ##########################################################################

    bdg.validateTable(spark)


if __name__ == '__main__':

    main()

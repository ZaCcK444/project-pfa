# src/train_models.py - Fixed version
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.feature import StringIndexer
import os
import shutil

def train_and_save_models():
    spark = SparkSession.builder \
        .appName("ModelTraining") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .getOrCreate()

    try:
        # Load data
        reviews = spark.read.parquet("data/cleaned_reviews.parquet")
        
        # Verify required columns exist
        required_columns = {'user_id', 'product_id', 'rating'}
        if not required_columns.issubset(set(reviews.columns)):
            missing = required_columns - set(reviews.columns)
            raise ValueError(f"Missing required columns: {missing}")

        # Create indexers
        user_indexer = StringIndexer(
            inputCol="user_id",
            outputCol="user_id_index"
        ).fit(reviews)

        product_indexer = StringIndexer(
            inputCol="product_id",
            outputCol="product_id_index"
        ).fit(reviews)

        # Transform data
        indexed_data = user_indexer.transform(reviews)
        indexed_data = product_indexer.transform(indexed_data)

        # Train ALS model
        als = ALS(
            maxIter=5,
            regParam=0.01,
            userCol="user_id_index",
            itemCol="product_id_index",
            ratingCol="rating",
            coldStartStrategy="drop",
            nonnegative=True
        )
        als_model = als.fit(indexed_data)

        # Save models
        model_dir = "models"
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)
        os.makedirs(model_dir)

        als_model.write().overwrite().save(os.path.join(model_dir, "als_model"))
        user_indexer.write().overwrite().save(os.path.join(model_dir, "user_indexer"))
        product_indexer.write().overwrite().save(os.path.join(model_dir, "product_indexer"))

        print("Models trained and saved successfully!")

    except Exception as e:
        print(f"ERROR: {str(e)}")
        raise
    finally:
        spark.stop()

if __name__ == "__main__":
    train_and_save_models()
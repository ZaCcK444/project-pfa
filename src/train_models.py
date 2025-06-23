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
        # Load data - using Windows path
        reviews = spark.read.parquet(os.path.join("data", "cleaned_reviews.parquet"))
        
        # Verify columns - must match EXACTLY what's in your data
        if "user_id" not in reviews.columns:
            raise ValueError("Column 'user_id' not found. Available columns: " + str(reviews.columns))

        # Create indexers with correct column names
        user_indexer = StringIndexer(
            inputCol="user_id",  # Must match your data exactly
            outputCol="user_id_index"  # This will be the new column name
        ).fit(reviews)

        product_indexer = StringIndexer(
            inputCol="product_id",  # Must match your data exactly
            outputCol="product_id_index"  # This will be the new column name
        ).fit(reviews)

        # Transform data
        indexed_data = user_indexer.transform(reviews)
        indexed_data = product_indexer.transform(indexed_data)

        # Train ALS model
        als = ALS(
            maxIter=5,
            regParam=0.01,
            userCol="user_id_index",  # Must match indexer output
            itemCol="product_id_index",  # Must match indexer output
            ratingCol="rating",
            coldStartStrategy="drop",
            nonnegative=True
        )
        als_model = als.fit(indexed_data)

        # Save models - Windows compatible paths
        model_dir = "models"
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)
        os.makedirs(model_dir)

        als_model.save(os.path.join(model_dir, "als_model"))
        user_indexer.save(os.path.join(model_dir, "user_indexer"))
        product_indexer.save(os.path.join(model_dir, "product_indexer"))

        print("Models saved successfully with columns:")
        print(f"- Original user column: user_id")
        print(f"- Indexed user column: user_id_index")

    except Exception as e:
        print(f"ERROR: {str(e)}")
        raise
    finally:
        spark.stop()

if __name__ == "__main__":
    train_and_save_models()
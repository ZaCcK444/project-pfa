from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.feature import StringIndexer
import os
import shutil

def train_and_save_models():
    # Initialize Spark with more memory
    spark = SparkSession.builder \
        .appName("ModelTraining") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .getOrCreate()
    
    try:
        # Load data
        print("Loading data...")
        reviews = spark.read.parquet("data/cleaned_reviews.parquet")
        
        # Verify columns
        required_columns = {'user_id', 'product_id', 'rating'}
        missing = required_columns - set(reviews.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        # Create indexers
        print("Creating indexers...")
        user_indexer = StringIndexer(
            inputCol="user_id",
            outputCol="userIdIndex"
        ).fit(reviews)
        
        product_indexer = StringIndexer(
            inputCol="product_id",
            outputCol="productIdIndex"
        ).fit(reviews)
        
        # Transform data
        print("Transforming data...")
        indexed_data = user_indexer.transform(reviews)
        indexed_data = product_indexer.transform(indexed_data)
        
        # Train ALS model
        print("Training ALS model...")
        als = ALS(
            maxIter=5,
            regParam=0.01,
            userCol="userIdIndex",
            itemCol="productIdIndex",
            ratingCol="rating",
            coldStartStrategy="drop",
            nonnegative=True
        )
        als_model = als.fit(indexed_data)
        
        # Prepare model paths
        model_paths = {
            'als': "models/als_model",
            'user_indexer': "models/user_indexer",
            'product_indexer': "models/product_indexer"
        }
        
        # Create models directory
        os.makedirs("models", exist_ok=True)
        
        # CORRECT WAY TO OVERWRITE MODELS
        print("\nSaving models with overwrite...")
        als_model.write().overwrite().save(model_paths['als'])
        user_indexer = StringIndexer(inputCol="user_id", outputCol="user_id_index")
         product_indexer = StringIndexer(inputCol="product_id", outputCol="product_id_index")
         
        print("\nSUCCESS! Models saved to:")
        for name, path in model_paths.items():
            print(f"- {name}: {os.path.abspath(path)}")
            
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        print("\nTROUBLESHOOTING:")
        print("1. Delete the models/ directory manually")
        print("2. Verify data/cleaned_reviews.parquet exists")
        print("3. Check columns: user_id, product_id, rating")
        raise
    finally:
        spark.stop()
        print("Spark session closed")

if __name__ == "__main__":
    print("Starting model training...")
    train_and_save_models()
    print("Training completed successfully!")
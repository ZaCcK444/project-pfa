from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import os

def load_data():
    spark = None
    try:
        # Initialize Spark with optimized configuration
        spark = SparkSession.builder \
            .appName("RecommendationSystem") \
            .config("spark.driver.memory", "4g") \
            .config("spark.executor.memory", "4g") \
            .config("spark.sql.shuffle.partitions", "100") \
            .getOrCreate()
        
        # Define and verify paths
        base_path = os.path.abspath("C:\Users\PC\Desktop\project\data")
        reviews_path = os.path.join(base_path, "cleaned_reviews.parquet")
        catalog_path = os.path.join(base_path, "product_catalog.parquet")
        
        if not os.path.exists(reviews_path):
            raise FileNotFoundError(f"Reviews data not found at: {reviews_path}")
        if not os.path.exists(catalog_path):
            raise FileNotFoundError(f"Catalog data not found at: {catalog_path}")
        
        # Load data
        reviews_df = spark.read.parquet(reviews_path)
        product_catalog = spark.read.parquet(catalog_path)
        
        # Cache and return
        reviews_df.cache()
        product_catalog.cache()
        
        print("Data loaded successfully:")
        print(f"- Reviews: {reviews_df.count()} records")
        print(f"- Products: {product_catalog.count()} records")
        
        return spark, reviews_df, product_catalog
        
    except Exception as e:
        if spark is not None:
            spark.stop()
        raise RuntimeError(f"Data loading failed: {str(e)}")

if __name__ == "__main__":
    spark = None
    try:
        spark, reviews_df, product_catalog = load_data()
        
        print("\nSample Reviews:")
        reviews_df.show(5, truncate=True)
        
        print("\nSample Products:")
        product_catalog.show(5, truncate=True)
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
    finally:
        if spark is not None:
            spark.stop()
            print("\nSpark session closed")
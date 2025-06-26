from pyspark.sql import SparkSession
import os
import logging
from pyspark.sql import SparkSession
from src.utils import get_spark_config
from src.spark_connector import create_spark_session

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

    def load_data():
     spark = None
       try:
        # Use the connector function to create a robust session
        spark = create_spark_session("RecommendationSystem")
        
        # Define paths
        data_dir = "data"
        reviews_path = os.path.join(data_dir, "cleaned_reviews.parquet")
        catalog_path = os.path.join(data_dir, "product_catalog.parquet")
        
        # Verify paths exist
        if not os.path.exists(reviews_path):
            raise FileNotFoundError(f"Reviews data not found at: {os.path.abspath(reviews_path)}")
        if not os.path.exists(catalog_path):
            raise FileNotFoundError(f"Catalog data not found at: {os.path.abspath(catalog_path)}")

        logger.info("Loading reviews data...")
        reviews_df = spark.read.parquet(reviews_path)
        
        logger.info("Loading product catalog...")
        product_catalog = spark.read.parquet(catalog_path)
        
        # Cache frequently used DataFrames
        reviews_df.cache()
        product_catalog.cache()
        
        logger.info(f"Data loaded successfully. Reviews: {reviews_df.count()}, Products: {product_catalog.count()}")
        
        return spark, reviews_df, product_catalog
        
    except Exception as e:
        logger.error(f"Data loading failed: {str(e)}")
        if spark is not None:
            spark.stop()
        raise RuntimeError(f"Data loading failed: {str(e)}")

if __name__ == "__main__":
    try:
        spark, reviews_df, product_catalog = load_data()
        
        print("\nReviews Schema:")
        reviews_df.printSchema()
        
        print("\nSample Reviews:")
        reviews_df.show(5, truncate=True)
        
        print("\nProducts Schema:")
        product_catalog.printSchema()
        
        print("\nSample Products:")
        product_catalog.show(5, truncate=True)
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
    finally:
        if 'spark' in locals():
            spark.stop()
            print("\nSpark session closed")
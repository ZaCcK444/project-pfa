from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, when, regexp_replace, expr
from pyspark.sql.types import DoubleType, FloatType, StringType
import logging
import os
from src.utils import get_spark_config
from src.spark_connector import create_spark_session

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def clean_data():
    spark = None
    try:
        spark = create_spark_session("ECommerceDataCleaning")
        
        logger.info("Loading raw data...")
        
        # Load the raw data
        df = spark.read.csv('data/amazon_reviews.csv', header=True, inferSchema=True)
        
        # Select and rename relevant columns
        df = df.select(
            col("review/userId").alias("user_id"),
            col("product/productId").alias("product_id"),
            col("product/title").alias("title"),
            col("product/price").alias("raw_price"),
            col("review/score").alias("raw_rating"),
            col("review/helpfulness").alias("helpfulness")
        )

        logger.info("Cleaning price data...")
        
        # Clean price data - CORRECTED VERSION
        df = df.withColumn("price", 
            when(
                (col("raw_price").isNull()) | 
                (col("raw_price") == "") | 
                (col("raw_price") == "unknown"), 
                None
            ).otherwise(
                regexp_replace(col("raw_price"), "[^0-9.]", "")
            )
        )
        
        # Safe conversion with validation
        df = df.withColumn("price", 
            when(
                (col("price") != "") & 
                (col("price").rlike("^[0-9.]+$")),
                col("price").cast(DoubleType())
            ).otherwise(None)
        )

        logger.info("Cleaning rating data...")
        
        # Clean rating data
        df = df.withColumn("raw_rating_str", col("raw_rating").cast(StringType()))
        
        df = df.withColumn("rating",
            when(col("raw_rating_str").contains("/"),
                expr("""
                    CASE WHEN split(raw_rating_str, '/')[1] = '0' THEN NULL
                    ELSE cast(split(raw_rating_str, '/')[0] as double) / cast(split(raw_rating_str, '/')[1] as double)
                    END
                """)
            ).otherwise(
                when(col("raw_rating_str").rlike("^[0-9.]+$"),
                    col("raw_rating_str").cast(FloatType())
                ).otherwise(None)
            )
        ).drop("raw_rating_str")

        logger.info("Processing helpfulness data...")
        
        # Extract helpful votes
        df = df.withColumn("helpful_votes",
            when(col("helpfulness").rlike("^[0-9]+/[0-9]+$"),
                expr("cast(split(helpfulness, '/')[0] as int)")
            ).otherwise(None)
        )

        # Remove records with missing essential data
        initial_count = df.count()
        df = df.dropna(subset=['user_id', 'product_id', 'rating'])
        filtered_count = df.count()
        logger.info(f"Removed {initial_count - filtered_count} records with missing essential data")

        # Calculate median price safely
        price_stats = df.filter(col("price").isNotNull()).agg(
            expr("percentile_approx(price, 0.5)").alias("median_price")
        ).first()
        median_price = price_stats.median_price if price_stats.median_price is not None else 0.0
        
        # Fill missing prices
        df = df.fillna(median_price, subset=['price'])

        logger.info("Filtering for active users and popular items...")
        
        # Filter for active users and popular items
        user_counts = df.groupBy("user_id").agg(count("*").alias("user_count"))
        product_counts = df.groupBy("product_id").agg(count("*").alias("product_count"))
        
        df = df.join(user_counts, "user_id") \
               .join(product_counts, "product_id") \
               .filter((col("user_count") >= 5) & (col("product_count") >= 10))
        
        # Clean up temporary columns
        df = df.drop("raw_price", "raw_rating", "helpfulness", "user_count", "product_count")

        # Save cleaned data
        logger.info("Saving cleaned data...")
        output_dir = "data"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        df.write.mode("overwrite").parquet(os.path.join(output_dir, "cleaned_reviews.parquet"))
        
        # Save product catalog
        product_catalog = df.select("product_id", "title", "price").distinct()
        product_catalog.write.mode("overwrite").parquet(os.path.join(output_dir, "product_catalog.parquet"))

        logger.info(f"Data cleaning completed successfully! Final records: {df.count()}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in data cleaning: {str(e)}")
        raise
    finally:
        if spark is not None:
            spark.stop()

if __name__ == "__main__":
    clean_data()
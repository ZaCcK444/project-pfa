from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, when, regexp_replace, expr
from pyspark.sql.types import DoubleType, FloatType

def clean_data():
    # Initialize Spark session
    spark = SparkSession.builder \
        .appName("ECommerceDataCleaning") \
        .getOrCreate()

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

    # Clean price data - PROPERLY CHAINED VERSION
    df = df.withColumn("price", 
        when(
            (col("raw_price").isNull() | 
            (col("raw_price") == "") | 
            (col("raw_price") == "unknown")), 
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

    # Clean rating data
    df = df.withColumn("rating",
        when(col("raw_rating").contains("/"),
            expr("""
                CASE WHEN split(raw_rating, '/')[1] = '0' THEN NULL
                ELSE cast(split(raw_rating, '/')[0] as double) / cast(split(raw_rating, '/')[1] as double)
                END
            """)
        ).otherwise(
            when(col("raw_rating").rlike("^[0-9.]+$"),
                col("raw_rating").cast(FloatType())
            ).otherwise(None)
        )
    )

    # Extract helpful votes
    df = df.withColumn("helpful_votes",
        when(col("helpfulness").rlike("^[0-9]+/[0-9]+$"),
            expr("cast(split(helpfulness, '/')[0] as int)")
        ).otherwise(None)
    )

    # Remove records with missing essential data
    df = df.dropna(subset=['user_id', 'product_id', 'rating'])

    # Calculate median price safely
    median_price = df.filter(col("price").isNotNull()).selectExpr("percentile_approx(price, 0.5)").first()[0] or 0.0
    
    # Fill missing prices
    df = df.fillna(median_price, subset=['price'])

    # Filter for active users and popular items
    user_counts = df.groupBy("user_id").agg(count("*").alias("user_count"))
    product_counts = df.groupBy("product_id").agg(count("*").alias("product_count"))
    
    df = df.join(user_counts, "user_id") \
           .join(product_counts, "product_id") \
           .filter((col("user_count") >= 5) & (col("product_count") >= 10))

    # Save cleaned data
    df.write.parquet("data/cleaned_reviews.parquet", mode="overwrite")
    
    # Save product catalog
    product_catalog = df.select("product_id", "title", "price").distinct()
    product_catalog.write.parquet("data/product_catalog.parquet", mode="overwrite")

    print("Data cleaning completed successfully!")
    print(f"Final records: {df.count()}")
    print("Sample data:")
    df.show(5, truncate=False)

    spark.stop()

if __name__ == "__main__":
    clean_data()
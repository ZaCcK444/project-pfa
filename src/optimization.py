from pyspark.sql import SparkSession
from pyspark.sql.functions import broadcast
from spark_loader import load_data
from hybrid_model import HybridRecommender
from src.spark_connector import create_spark_session

def optimize_spark_config(spark):
    # Optimize Spark configuration
    spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "10485760")  # 10MB
    spark.conf.set("spark.sql.shuffle.partitions", "200")
    spark.conf.set("spark.executor.memory", "4g")
    spark.conf.set("spark.driver.memory", "4g")
    spark.conf.set("spark.default.parallelism", "200")
    return spark

def optimize_queries(df):
    # Cache frequently used DataFrames
    df.cache()
    
    # Use broadcast join for small DataFrames
    def optimized_join(df1, df2, join_col):
        if df2.count() < 10000:  # Broadcast if small
            return df1.join(broadcast(df2), join_col)
        return df1.join(df2, join_col)
    
    return optimized_join

def optimize_recommendations(recommender, user_id):
    # Use optimized join function
    join_fn = optimize_queries(recommender.spark)
    
    # Get ALS recommendations with optimizations
    user_df = recommender.spark.createDataFrame([(user_id,)], ["userId"])
    user_index = recommender.user_indexer.transform(user_df).select("userIdIndex").first()[0]
    
    # Get ALS recommendations with optimizations
    als_recs = recommender.als_model.recommendForUserSubset(
        recommender.spark.createDataFrame([(user_index,)], ["userIdIndex"]), 
        10
    )
    
    # Rest of the optimized implementation...
    # (Similar to hybrid_model.py but with optimizations applied)

if __name__ == "__main__":

    
    spark = load_data()[0]  # Get just the Spark session
    spark = optimize_spark_config(spark)
    
    # Test with a sample user
    reviews_df = spark.read.parquet("data/cleaned_reviews.parquet")
    sample_user = reviews_df.select("userId").first()[0]
    
    recommender = HybridRecommender(spark)
    optimized_recs = optimize_recommendations(recommender, sample_user)
    
    print("Optimized recommendations:")
    optimized_recs.show()
    
    spark.stop()
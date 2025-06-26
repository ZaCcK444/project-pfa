from pyspark.sql import SparkSession
from pyspark.sql.functions import broadcast
from src.spark_loader import load_data
from src.hybrid_model import HybridRecommender
from src.spark_connector import create_spark_session

def optimize_spark_config(spark):
    """Optimize Spark configuration for better performance"""
    spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "10485760")  # 10MB
    spark.conf.set("spark.sql.shuffle.partitions", "200")
    spark.conf.set("spark.executor.memory", "4g")
    spark.conf.set("spark.driver.memory", "4g")
    spark.conf.set("spark.default.parallelism", "200")
    return spark

def optimize_queries(df):
    """Cache frequently used DataFrames and provide optimized join function"""
    # Cache frequently used DataFrames
    df.cache()
    
    # Use broadcast join for small DataFrames
    def optimized_join(df1, df2, join_col):
        if df2.count() < 10000:  # Broadcast if small
            return df1.join(broadcast(df2), join_col)
        return df1.join(df2, join_col)
    
    return optimized_join

def optimize_recommendations(recommender, user_id):
    """Get optimized recommendations for a user"""
    try:
        # Get ALS recommendations with optimizations
        user_df = recommender.spark.createDataFrame([(user_id,)], ["user_id"])
        indexed_user = recommender.user_indexer.transform(user_df)
        
        if indexed_user.isEmpty():
            return None
            
        user_index = indexed_user.select("user_id_index").first()[0]
        
        # Get ALS recommendations with optimizations
        user_subset = recommender.spark.createDataFrame([(user_index,)], ["user_id_index"])
        als_recs = recommender.als_model.recommendForUserSubset(user_subset, 10)
        
        # Process and return optimized recommendations
        if als_recs.isEmpty():
            return None
            
        # Extract recommendations and join with product catalog
        recommendations_row = als_recs.select("recommendations").first()
        if not recommendations_row or not recommendations_row[0]:
            return None
            
        product_recs = recommendations_row[0]
        product_indices = [r.product_id_index for r in product_recs]
        
        if not product_indices:
            return None
            
        product_indices_df = recommender.spark.createDataFrame(
            [(int(idx),) for idx in product_indices],
            ["product_id_index"]
        )
        
        # Transform back to product IDs
        product_ids_df = recommender.product_indexer.transform(product_indices_df)
        
        # Join with product catalog using broadcast if small
        if recommender.product_catalog.count() < 10000:
            result = product_ids_df.join(
                broadcast(recommender.product_catalog),
                "product_id",
                "inner"
            )
        else:
            result = product_ids_df.join(
                recommender.product_catalog,
                "product_id",
                "inner"
            )
        
        return result.select("product_id", "title", "price").limit(10)
        
    except Exception as e:
        print(f"Error in optimize_recommendations: {str(e)}")
        return None

if __name__ == "__main__":
    spark = None
    try:
        spark = create_spark_session("Optimization")
        spark = optimize_spark_config(spark)
        
        # Test with a sample user
        reviews_df = spark.read.parquet("data/cleaned_reviews.parquet")
        sample_user = reviews_df.select("user_id").first()[0]
        
        recommender = HybridRecommender(spark)
        optimized_recs = optimize_recommendations(recommender, sample_user)
        
        if optimized_recs is not None:
            print("Optimized recommendations:")
            optimized_recs.show()
        else:
            print("No recommendations found")
        
    except Exception as e:
        print(f"Error in optimization: {str(e)}")
    finally:
        if spark is not None:
            spark.stop()
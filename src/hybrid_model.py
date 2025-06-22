from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, udf
from pyspark.sql.types import FloatType
import numpy as np

class HybridRecommender:
    def __init__(self, spark):
        self.spark = spark
        
        # Load models and data
        self.als_model = ALSModel.load("../models/als_model")
        self.content_similarities = spark.read.parquet("../data/content_similarities.parquet")
        self.product_catalog = spark.read.parquet("../data/product_catalog.parquet")
        
        # Load StringIndexers to convert between IDs
        from pyspark.ml.feature import StringIndexerModel
        self.user_indexer = StringIndexerModel.load("../models/user_indexer")
        self.product_indexer = StringIndexerModel.load("../models/product_indexer")
    
    def get_als_recommendations(self, user_id, n=10):
        # Convert user ID to index
        user_df = self.spark.createDataFrame([(user_id,)], ["userId"])
        user_index = self.user_indexer.transform(user_df).select("userIdIndex").first()[0]
        
        # Get ALS recommendations
        als_recs = self.als_model.recommendForUserSubset(
            self.spark.createDataFrame([(user_index,)], ["userIdIndex"]), 
            n
        )
        
        # Convert back to original product IDs
        product_indices = [r.productIdIndex for r in als_recs.first().recommendations]
        product_indices_df = self.spark.createDataFrame(
            [(idx,) for idx in product_indices],
            ["productIdIndex"]
        )
        
        # Join with original product IDs
        original_ids = self.product_indexer \
            .transform(product_indices_df) \
            .select("productId")
        
        return original_ids
    
    def get_content_recommendations(self, user_id, n=10):
        # Get products the user has interacted with
        user_reviews = self.spark.read.parquet("../data/cleaned_reviews.parquet")
        
        # Get content-based recommendations
        content_recs = get_content_recommendations(
            user_id, user_reviews, self.content_similarities, n
        )
        
        return content_recs.select(col("similar_productId").alias("productId"))
    
    def hybrid_recommend(self, user_id, n=10, als_weight=0.7, content_weight=0.3):
        # Get recommendations from both models
        als_recs = self.get_als_recommendations(user_id, n)
        content_recs = self.get_content_recommendations(user_id, n)
        
        # Assign scores
        als_recs = als_recs.withColumn("als_score", lit(1.0))
        content_recs = content_recs.withColumn("content_score", lit(1.0))
        
        # Full outer join to combine recommendations
        hybrid_recs = als_recs.join(
            content_recs,
            "productId",
            "outer"
        ).fillna(0)
        
        # Calculate hybrid score
        hybrid_recs = hybrid_recs.withColumn(
            "hybrid_score",
            col("als_score") * als_weight + col("content_score") * content_weight
        )
        
        # Get top N recommendations
        top_recs = hybrid_recs.orderBy(col("hybrid_score").desc()).limit(n)
        
        # Join with product catalog for display
        final_recs = top_recs.join(
            self.product_catalog,
            "productId"
        ).select("productId", "title", "price", "hybrid_score")
        
        return final_recs

if __name__ == "__main__":
    from spark_loader import load_data
    
    spark, reviews_df, _ = load_data()
    
    # Initialize hybrid recommender
    recommender = HybridRecommender(spark)
    
    # Get a sample user
    sample_user = reviews_df.select("userId").first()[0]
    
    # Get hybrid recommendations
    print(f"\nHybrid recommendations for user {sample_user}:")
    hybrid_recs = recommender.hybrid_recommend(sample_user)
    hybrid_recs.show(truncate=False)
    
    spark.stop()
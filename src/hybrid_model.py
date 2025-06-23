# src/hybrid_model.py - Fixed version
from pyspark.ml.recommendation import ALSModel
from pyspark.ml.feature import StringIndexerModel
from pyspark.sql.functions import col, lit, coalesce
import os

class HybridRecommender:
    def __init__(self, spark):
        self.spark = spark
        self._load_models()
        self._load_data()
        
    def _load_models(self):
        """Load all required models with validation"""
        try:
            # Load ALS model
            self.als_model = ALSModel.load("models/als_model")
            
            # Load indexers
            self.user_indexer = StringIndexerModel.load("models/user_indexer")
            self.product_indexer = StringIndexerModel.load("models/product_indexer")
            
        except Exception as e:
            raise ValueError(f"Failed to load models: {str(e)}\n"
                          "Please ensure you've run train_models.py first")

    def _load_data(self):
        """Load required data files"""
        try:
            self.product_catalog = self.spark.read.parquet("data/product_catalog.parquet")
            self.content_similarities = self.spark.read.parquet("data/content_similarities.parquet")
            self.reviews_df = self.spark.read.parquet("data/cleaned_reviews.parquet")
        except Exception as e:
            raise ValueError(f"Failed to load data: {str(e)}")


    def get_als_recommendations(self, user_id, n=10):
    """Get ALS recommendations for a user with improved error handling"""
        try:
            # Convert user_id to index
            user_df = self.spark.createDataFrame([(user_id,)], ["user_id"])
            indexed_user = self.user_indexer.transform(user_df)
            
            # Check if user exists in the model
            if indexed_user.isEmpty():
                raise ValueError(f"User {user_id} not found in the model")
                
            user_index = indexed_user.select("user_id_index").first()[0]
            
            # Create a small DataFrame with the user index
            user_subset = self.spark.createDataFrame(
                [(user_index,)], 
                ["user_id_index"]
            ).cache()
            
            # Get recommendations
            recs = self.als_model.recommendForUserSubset(user_subset, n)
            
            # Check if recommendations exist
            if recs.isEmpty():
                return self.spark.createDataFrame(
                    [], 
                    self.product_catalog.schema
                )
            
            # Process recommendations safely
            product_recs = recs.select("recommendations").first()[0]
            if not product_recs:
                return self.spark.createDataFrame(
                    [], 
                    self.product_catalog.schema
                )
                
            product_indices = [r.product_id_index for r in product_recs]
            
            # Convert back to original product IDs
            product_indices_df = self.spark.createDataFrame(
                [(idx,) for idx in product_indices],
                ["product_id_index"]
            )
            
            product_ids_df = self.product_indexer.transform(product_indices_df)
            
            # Join with product catalog
            return product_ids_df.join(
                self.product_catalog,
                "product_id",
                "inner"
            ).select("product_id", "title", "price").limit(n)
            
        except Exception as e:
            print(f"Detailed ALS recommendation error: {str(e)}")
            # Return empty DataFrame with correct schema if something fails
            return self.spark.createDataFrame([], self.product_catalog.schema)
        
    def get_content_recommendations(self, user_id, n=10):
        """Get content-based recommendations"""
        try:
            from content_based import get_content_recommendations
            return get_content_recommendations(
                user_id,
                self.reviews_df,
                self.content_similarities,
                n
            ).join(
                self.product_catalog,
                col("similar_productId") == col("product_id")
            ).select("product_id", "title", "price")
            
        except Exception as e:
            raise ValueError(f"Content recommendation failed: {str(e)}")

    def hybrid_recommend(self, user_id, n=10, als_weight=0.7, content_weight=0.3):
    """Generate hybrid recommendations with fallback logic"""
        try:
            # Get ALS recommendations with fallback
            als_recs = self.get_als_recommendations(user_id, n)
            if als_recs.isEmpty():
                als_recs = self.spark.createDataFrame(
                    [], 
                    ["product_id", "title", "price", "als_score"]
                )
            else:
                als_recs = als_recs.withColumn("als_score", lit(1.0))
            
            # Get content recommendations with fallback
            content_recs = self.get_content_recommendations(user_id, n)
            if content_recs.isEmpty():
                content_recs = self.spark.createDataFrame(
                    [], 
                    ["product_id", "title", "price", "content_score"]
                )
            else:
                content_recs = content_recs.withColumn("content_score", lit(1.0))
            
            # Combine recommendations safely
            combined = als_recs.join(
                content_recs,
                ["product_id", "title", "price"],
                "outer"
            ).fillna(0)
            
            # Calculate hybrid score
            result = combined.withColumn(
                "hybrid_score",
                col("als_score") * als_weight + col("content_score") * content_weight
            ).orderBy(col("hybrid_score").desc()).limit(n)
            
            # Ensure we return at least some recommendations
            if result.isEmpty():
                # Fallback to just ALS if hybrid fails
                return als_recs.limit(n)
            return result
            
        except Exception as e:
            print(f"Hybrid recommendation error: {str(e)}")
            # Final fallback - return empty DataFrame with correct schema
            return self.spark.createDataFrame(
                [], 
                ["product_id", "title", "price", "hybrid_score"]
            )
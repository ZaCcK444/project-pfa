from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALSModel
from pyspark.ml.feature import StringIndexerModel
from pyspark.sql.functions import col, lit, explode, sum as _sum
from pyspark.sql.types import StructType, StructField, StringType, DoubleType
import pyspark.sql.functions as F
import os
import time

class HybridRecommender:
    def __init__(self, spark):
        self.spark = spark
        self._setup_checkpointing()
        self._load_models()  # This will now call the properly implemented method
        self._load_data()
        
    def _setup_checkpointing(self):
        """Configure checkpoint directory"""
        checkpoint_dir = "/tmp/spark_checkpoints"
        self.spark.sparkContext.setCheckpointDir(checkpoint_dir)
        
    def _load_models(self):
        """Proper implementation of model loading"""
        try:
            # Verify models directory exists
            if not os.path.exists("models"):
                raise FileNotFoundError("Models directory not found at 'models/'")
                
            # Load ALS model
            als_model_path = os.path.join("models", "als_model")
            if not os.path.exists(als_model_path):
                raise FileNotFoundError(f"ALS model not found at {als_model_path}")
            self.als_model = ALSModel.load(als_model_path)
            
            # Load user indexer
            user_indexer_path = os.path.join("models", "user_indexer")
            if not os.path.exists(user_indexer_path):
                raise FileNotFoundError(f"User indexer not found at {user_indexer_path}")
            self.user_indexer = StringIndexerModel.load(user_indexer_path)
            
            # Load product indexer
            product_indexer_path = os.path.join("models", "product_indexer")
            if not os.path.exists(product_indexer_path):
                raise FileNotFoundError(f"Product indexer not found at {product_indexer_path}")
            self.product_indexer = StringIndexerModel.load(product_indexer_path)
            
            print("All models loaded successfully")
        except Exception as e:
            raise ValueError(f"Failed to load models: {str(e)}\n"
                          "Please ensure you've run train_models.py first to generate the required models")

    def _load_data(self):
        """Load required data files"""
        try:
            # Verify data directory exists
            if not os.path.exists("data"):
                raise FileNotFoundError("Data directory not found at 'data/'")
                
            # Load product catalog
            product_catalog_path = os.path.join("data", "product_catalog.parquet")
            if not os.path.exists(product_catalog_path):
                raise FileNotFoundError(f"Product catalog not found at {product_catalog_path}")
            self.product_catalog = self.spark.read.parquet(product_catalog_path)
            
            # Load content similarities
            content_similarities_path = os.path.join("data", "content_similarities.parquet")
            if not os.path.exists(content_similarities_path):
                raise FileNotFoundError(f"Content similarities not found at {content_similarities_path}")
            self.content_similarities = self.spark.read.parquet(content_similarities_path)
            
            # Load reviews data
            reviews_path = os.path.join("data", "cleaned_reviews.parquet")
            if not os.path.exists(reviews_path):
                raise FileNotFoundError(f"Reviews data not found at {reviews_path}")
            self.reviews_df = self.spark.read.parquet(reviews_path)
            
            print("All data loaded successfully")
        except Exception as e:
            raise ValueError(f"Failed to load data: {str(e)}")

    def get_als_recommendations(self, user_id, n=10):
        """Get ALS recommendations for a user"""
        try:
            # Convert user_id to index
            user_df = self.spark.createDataFrame([(user_id,)], ["user_id"])
            indexed_user = self.user_indexer.transform(user_df)
            
            if indexed_user.isEmpty():
                return self._empty_als_df()
                
            user_index = indexed_user.select("user_id_index").first()[0]
            
            # Get recommendations
            user_subset = self.spark.createDataFrame([(user_index,)], ["user_id_index"])
            recs = self.als_model.recommendForUserSubset(user_subset, n)
            
            if recs.isEmpty():
                return self._empty_als_df()
            
            # Process recommendations
            product_recs = recs.select("recommendations").first()[0]
            if not product_recs:
                return self._empty_als_df()
                
            product_indices = [r.product_id_index for r in product_recs]
            product_indices_df = self.spark.createDataFrame(
                [(idx,) for idx in product_indices],
                ["product_id_index"]
            )
            
            product_ids_df = self.product_indexer.transform(product_indices_df)
            
            return product_ids_df.join(
                self.product_catalog,
                "product_id",
                "inner"
            ).select("product_id", "title", "price").limit(n)
            
        except Exception as e:
            print(f"ALS recommendation error: {str(e)}")
            return self._empty_als_df()

    def _empty_als_df(self):
        """Create empty DataFrame with correct schema for ALS recommendations"""
        schema = StructType([
            StructField("product_id", StringType()),
            StructField("title", StringType()),
            StructField("price", DoubleType())
        ])
        return self.spark.createDataFrame([], schema)

    def get_content_recommendations(self, user_id, n=10):
        """Get content-based recommendations"""
        try:
            # Define schema for empty fallback
            content_schema = StructType([
                StructField("product_id", StringType()),
                StructField("title", StringType()),
                StructField("price", DoubleType()),
                StructField("total_score", DoubleType())
            ])
            
            # Get user's liked products
            liked_products = self.reviews_df.filter(col("user_id") == user_id) \
                .select("product_id", "rating") \
                .withColumn("weight", (col("rating") - 3) / 2)
            
            # Join with similarities
            recommendations = liked_products.join(
                self.content_similarities,
                liked_products["product_id"] == self.content_similarities["product1"],
                "left"
            ).select(
                "product_id",
                "weight",
                explode(col("sorted_similarities")).alias("similarity")
            ).select(
                "product_id",
                "weight",
                col("similarity.similar_productId").alias("recommended_product"),
                (col("similarity.score") * col("weight")).alias("weighted_score")
            )
            
            # Filter out already rated products
            user_rated = self.reviews_df.filter(col("user_id") == user_id) \
                .select("product_id").distinct()
                
            final_recs = recommendations.join(
                user_rated,
                col("recommended_product") == col("product_id"),
                "left_anti"
            ).groupBy("recommended_product") \
             .agg(_sum("weighted_score").alias("total_score")) \
             .orderBy(col("total_score").desc()) \
             .limit(n) \
             .join(
                 self.product_catalog,
                 col("recommended_product") == col("product_id")
             ).select("product_id", "title", "price", "total_score")
            
            return final_recs
            
        except Exception as e:
            print(f"Content recommendation error: {str(e)}")
            return self.spark.createDataFrame([], content_schema)

    def hybrid_recommend(self, user_id, n=10, als_weight=0.7, content_weight=0.3):
        """Generate hybrid recommendations"""
        try:
            # Define schema for empty results
            hybrid_schema = StructType([
                StructField("product_id", StringType()),
                StructField("title", StringType()),
                StructField("price", DoubleType()),
                StructField("hybrid_score", DoubleType())
            ])
            
            # Get ALS recommendations
            als_recs = self.get_als_recommendations(user_id, n)
            als_recs = als_recs.withColumn("als_score", lit(1.0)) if not als_recs.isEmpty() \
                else self.spark.createDataFrame([], als_recs.schema.add("als_score", DoubleType()))
            
            # Get content recommendations
            content_recs = self.get_content_recommendations(user_id, n)
            content_recs = content_recs.withColumnRenamed("total_score", "content_score") \
                if not content_recs.isEmpty() \
                else self.spark.createDataFrame([], content_recs.schema.add("content_score", DoubleType()))
            
            # Combine recommendations
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
            
            return result if not result.isEmpty() else self.spark.createDataFrame([], hybrid_schema)
            
        except Exception as e:
            print(f"Hybrid recommendation error: {str(e)}")
            return self.spark.createDataFrame([], hybrid_schema)
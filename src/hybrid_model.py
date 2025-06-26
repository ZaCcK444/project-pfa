from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALSModel
from pyspark.ml.feature import StringIndexerModel
from pyspark.sql.functions import col, lit, explode, sum as _sum
from pyspark.sql.types import StructType, StructField, StringType, DoubleType
import pyspark.sql.functions as F
import os
import logging
from src.spark_connector import create_spark_session

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HybridRecommender:
    def __init__(self, spark):
        self.spark = spark
        self._setup_checkpointing()
        self._load_models()
        self._load_data()
        
    def _setup_checkpointing(self):
        """Configure checkpoint directory"""
        checkpoint_dir = os.path.join("checkpoints")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.spark.sparkContext.setCheckpointDir(checkpoint_dir)
        
    def _load_models(self):
        """Load all required models with validation"""
        try:
            # Verify models directory exists
            models_dir = "models"
            if not os.path.exists(models_dir):
                raise FileNotFoundError(f"Models directory not found at {os.path.abspath(models_dir)}")
                
            # Load ALS model
            als_model_path = os.path.join(models_dir, "als_model")
            if not os.path.exists(als_model_path):
                raise FileNotFoundError(f"ALS model not found at {als_model_path}")
            self.als_model = ALSModel.load(als_model_path)
            
            # Load indexers
            user_indexer_path = os.path.join(models_dir, "user_indexer")
            product_indexer_path = os.path.join(models_dir, "product_indexer")
            
            if not os.path.exists(user_indexer_path):
                raise FileNotFoundError(f"User indexer not found at {user_indexer_path}")
            if not os.path.exists(product_indexer_path):
                raise FileNotFoundError(f"Product indexer not found at {product_indexer_path}")
                
            self.user_indexer = StringIndexerModel.load(user_indexer_path)
            self.product_indexer = StringIndexerModel.load(product_indexer_path)
            
            logger.info("All models loaded successfully")
            
        except Exception as e:
            logger.error(f"Model loading failed: {str(e)}")
            raise

    def _load_data(self):
        """Load required data files with validation"""
        try:
            # Verify data directory exists
            data_dir = "data"
            if not os.path.exists(data_dir):
                raise FileNotFoundError(f"Data directory not found at {os.path.abspath(data_dir)}")
                
            # Check if all required files exist
            product_catalog_path = os.path.join(data_dir, "product_catalog.parquet")
            content_similarities_path = os.path.join(data_dir, "content_similarities.parquet")
            reviews_path = os.path.join(data_dir, "cleaned_reviews.parquet")
            
            if not os.path.exists(product_catalog_path):
                raise FileNotFoundError(f"Product catalog not found at {product_catalog_path}")
            if not os.path.exists(content_similarities_path):
                raise FileNotFoundError(f"Content similarities not found at {content_similarities_path}")
            if not os.path.exists(reviews_path):
                raise FileNotFoundError(f"Reviews data not found at {reviews_path}")
                
            # Load datasets
            self.product_catalog = self.spark.read.parquet(product_catalog_path)
            self.content_similarities = self.spark.read.parquet(content_similarities_path)
            self.reviews_df = self.spark.read.parquet(reviews_path)
            
            # Cache frequently used data
            self.product_catalog.cache()
            self.reviews_df.cache()
            
            logger.info("All data loaded successfully")
            
        except Exception as e:
            logger.error(f"Data loading failed: {str(e)}")
            raise

    def get_als_recommendations(self, user_id, n=10):
        """Get ALS recommendations for a user with proper error handling"""
        try:
            # Convert user_id to index
            user_df = self.spark.createDataFrame([(user_id,)], ["user_id"])
            indexed_user = self.user_indexer.transform(user_df)
            
            if indexed_user.isEmpty():
                logger.warning(f"User {user_id} not found in indexer")
                return self._empty_als_df()
                
            user_index = indexed_user.select("user_id_index").first()[0]
            
            # Get recommendations
            user_subset = self.spark.createDataFrame([(user_index,)], ["user_id_index"])
            recs = self.als_model.recommendForUserSubset(user_subset, n)
            
            if recs.isEmpty():
                logger.warning(f"No ALS recommendations found for user {user_id}")
                return self._empty_als_df()
            
            # Process recommendations
            recommendations_row = recs.select("recommendations").first()
            if not recommendations_row or not recommendations_row[0]:
                logger.warning(f"Empty recommendations for user {user_id}")
                return self._empty_als_df()
                
            product_recs = recommendations_row[0]
            product_indices = [r.product_id_index for r in product_recs]
            
            if not product_indices:
                logger.warning(f"No product indices found for user {user_id}")
                return self._empty_als_df()
                
            product_indices_df = self.spark.createDataFrame(
                [(int(idx),) for idx in product_indices],
                ["product_id_index"]
            )
            
            # Transform back to product IDs
            product_ids_df = self.product_indexer.transform(product_indices_df)
            
            # Join with product catalog
            result = product_ids_df.join(
                self.product_catalog,
                "product_id",
                "inner"
            ).select("product_id", "title", "price").limit(n)
            
            return result
            
        except Exception as e:
            logger.error(f"ALS recommendation error for user {user_id}: {str(e)}")
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
        """Get content-based recommendations with proper error handling"""
        try:
            # Check if user exists
            user_exists = self.reviews_df.filter(col("user_id") == user_id).count() > 0
            if not user_exists:
                logger.warning(f"User {user_id} not found in reviews")
                return self._empty_content_df()
            
            # Get user's liked products with weights
            liked_products = self.reviews_df.filter(col("user_id") == user_id) \
                .select("product_id", "rating") \
                .withColumn("weight", (col("rating") - 3) / 2) \
                .filter(col("weight").isNotNull())

            if liked_products.isEmpty():
                logger.warning(f"No liked products found for user {user_id}")
                return self._empty_content_df()
            
            # Join with similarities
            recommendations = liked_products.join(
                self.content_similarities,
                liked_products["product_id"] == self.content_similarities["product1"],
                "inner"
            )
            
            if recommendations.isEmpty():
                logger.warning(f"No content similarities found for user {user_id}")
                return self._empty_content_df()
            
            # Process similar items
            recommendations = recommendations.select(
                "product_id",
                "weight",
                explode(col("similar_items")).alias("rec")
            ).select(
                "product_id",
                "weight",
                col("rec.product2").alias("recommended_product"),
                (col("rec.similarity") * col("weight")).alias("weighted_score")
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
             .limit(n)

            # Join with product catalog
            if final_recs.count() > 0:
                final_recs = final_recs.join(
                    self.product_catalog,
                    col("recommended_product") == col("product_id"),
                    "inner"
                ).select("product_id", "title", "price", "total_score")
            else:
                logger.warning(f"No final recommendations for user {user_id}")
                return self._empty_content_df()

            return final_recs
            
        except Exception as e:
            logger.error(f"Content recommendation error for user {user_id}: {str(e)}")
            return self._empty_content_df()

    def _empty_content_df(self):
        """Create empty DataFrame with correct schema for content recommendations"""
        schema = StructType([
            StructField("product_id", StringType()),
            StructField("title", StringType()),
            StructField("price", DoubleType()),
            StructField("total_score", DoubleType())
        ])
        return self.spark.createDataFrame([], schema)

    def hybrid_recommend(self, user_id, n=10, als_weight=0.7, content_weight=0.3):
        """Generate hybrid recommendations with proper error handling"""
        try:
            # Get recommendations from both models
            als_recs = self.get_als_recommendations(user_id, n * 2)  # Get more to combine
            content_recs = self.get_content_recommendations(user_id, n * 2)
            
            # Add scores for combination
            if not als_recs.isEmpty():
                als_recs = als_recs.withColumn("als_score", lit(1.0))
            else:
                als_recs = self.spark.createDataFrame([], 
                    StructType([
                        StructField("product_id", StringType()),
                        StructField("title", StringType()),
                        StructField("price", DoubleType()),
                        StructField("als_score", DoubleType())
                    ])
                )

            if not content_recs.isEmpty():
                # Normalize content scores to 0-1 range
                max_score_row = content_recs.agg(F.max("total_score").alias("max_score")).first()
                max_score = max_score_row.max_score if max_score_row.max_score else 1.0
                
                content_recs = content_recs.withColumn(
                    "content_score", 
                    col("total_score") / max_score
                ).drop("total_score")
            else:
                content_recs = self.spark.createDataFrame([], 
                    StructType([
                        StructField("product_id", StringType()),
                        StructField("title", StringType()),
                        StructField("price", DoubleType()),
                        StructField("content_score", DoubleType())
                    ])
                )

            # Combine recommendations
            if als_recs.isEmpty() and content_recs.isEmpty():
                return self._empty_hybrid_df()
            
            # Full outer join to combine all recommendations
            combined = als_recs.join(
                content_recs,
                ["product_id", "title", "price"],
                "full_outer"
            ).fillna(0, subset=["als_score", "content_score"])

            # Calculate hybrid score
            result = combined.withColumn(
                "hybrid_score",
                col("als_score") * als_weight + col("content_score") * content_weight
            ).select("product_id", "title", "price", "hybrid_score") \
             .orderBy(col("hybrid_score").desc()) \
             .limit(n)

            return result
            
        except Exception as e:
            logger.error(f"Hybrid recommendation error for user {user_id}: {str(e)}")
            return self._empty_hybrid_df()

    def _empty_hybrid_df(self):
        """Create empty DataFrame with correct schema for hybrid recommendations"""
        schema = StructType([
            StructField("product_id", StringType()),
            StructField("title", StringType()),
            StructField("price", DoubleType()),
            StructField("hybrid_score", DoubleType())
        ])
        return self.spark.createDataFrame([], schema)
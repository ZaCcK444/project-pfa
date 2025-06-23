from pyspark.ml.recommendation import ALSModel
from pyspark.ml.feature import StringIndexerModel
from pyspark.sql.functions import col, lit
import os

class HybridRecommender:
    def __init__(self, spark):
        self.spark = spark
        
        # MUST match EXACTLY what's in your data and training code
        self.USER_ID_COL = "user_id"  # Original column name in your data
        self.USER_INDEX_COL = "user_id_index"  # From StringIndexer
        self.PRODUCT_ID_COL = "product_id"  # Original column name
        self.PRODUCT_INDEX_COL = "product_id_index"  # From StringIndexer
        
        # Load models with validation
        self._load_models()
        
        # Load supporting data with Windows paths
        self.product_catalog = spark.read.parquet(os.path.join("data", "product_catalog.parquet"))
        self.content_similarities = spark.read.parquet(os.path.join("data", "content_similarities.parquet"))

    def _load_models(self):
        """Load and validate all models with Windows paths"""
        try:
            # Load ALS model
            self.als_model = ALSModel.load(os.path.join("models", "als_model"))
            
            # Load and validate user indexer
            self.user_indexer = StringIndexerModel.load(os.path.join("models", "user_indexer"))
            if self.user_indexer.getInputCol() != "user_id":
                raise ValueError(f"User indexer input column mismatch. Expected 'user_id', got '{self.user_indexer.getInputCol()}'")
            
            # Load product indexer
            self.product_indexer = StringIndexerModel.load(os.path.join("models", "product_indexer"))
            if self.product_indexer.getInputCol() != "product_id":
                raise ValueError(f"Product indexer input column mismatch. Expected 'product_id', got '{self.product_indexer.getInputCol()}'")

        except Exception as e:
            raise ValueError(f"Failed to load models: {str(e)}\n"
                          "1. Delete the 'models' folder\n"
                          "2. Retrain models using train_models.py\n"
                          "3. Verify your data files exist in the 'data' folder")

    def get_als_recommendations(self, user_id, n=10):
        """Get recommendations using the correct column names"""
        try:
            # Create DataFrame with correct column name
            user_df = self.spark.createDataFrame(
                [(user_id,)],
                [self.USER_ID_COL]  # Using "user_id" as this is the original column
            )
            
            # Transform to index
            indexed_user = self.user_indexer.transform(user_df)
            user_index = indexed_user.select(self.USER_INDEX_COL).first()[0]
            
            # Get recommendations
            recs = self.als_model.recommendForUserSubset(
                self.spark.createDataFrame([(user_index,)], [self.USER_INDEX_COL]),
                n
            )
            
            # Process recommendations
            product_recs = recs.first().recommendations
            product_indices = [r.__getattr__(self.PRODUCT_INDEX_COL) for r in product_recs]
            product_indices_df = self.spark.createDataFrame(
                [(idx,) for idx in product_indices],
                [self.PRODUCT_INDEX_COL]
            )
            
            # Convert back to original product IDs
            return self.product_indexer.transform(product_indices_df)\
                .select(self.PRODUCT_ID_COL)
                
        except Exception as e:
            raise ValueError(f"Failed to get recommendations: {str(e)}")
    def get_content_recommendations(self, user_id, n=10):
        """Get content-based recs with consistent column names"""
        try:
            from content_based import get_content_recommendations
            
            user_reviews = self.spark.read.parquet("data/cleaned_reviews.parquet")
            return get_content_recommendations(
                user_id, 
                user_reviews, 
                self.content_similarities, 
                n
            ).select(col("similar_productId").alias(self.PRODUCT_ID_COL))
            
        except Exception as e:
            raise ValueError(f"Content recommendation failed: {str(e)}")
    
    def hybrid_recommend(self, user_id, n=10, als_weight=0.7, content_weight=0.3):
        """Generate hybrid recommendations"""
        try:
            als_recs = self.get_als_recommendations(user_id, n)
            content_recs = self.get_content_recommendations(user_id, n)
            
            # Combine with scores
            combined = als_recs.withColumn("als_score", lit(1.0))\
                .join(
                    content_recs.withColumn("content_score", lit(1.0)),
                    self.PRODUCT_ID_COL,
                    "outer"
                ).fillna(0)
            
            # Calculate hybrid score
            return combined.withColumn(
                    "hybrid_score",
                    col("als_score")*als_weight + col("content_score")*content_weight
                )\
                .orderBy(col("hybrid_score").desc())\
                .limit(n)\
                .join(
                    self.product_catalog,
                    self.PRODUCT_ID_COL
                )\
                .select(self.PRODUCT_ID_COL, "title", "price", "hybrid_score")
                
        except Exception as e:
            raise ValueError(f"Hybrid recommendation failed: {str(e)}")
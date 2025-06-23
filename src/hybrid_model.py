from pyspark.ml.recommendation import ALSModel
from pyspark.ml.feature import StringIndexerModel
from pyspark.sql import SparkSession
import os

class HybridRecommender:
    def __init__(self, spark):
        self.spark = spark
        
        # These MUST match your training code and data
        self.expected_mappings = {
            'user': {
                'input_col': "userId",
                'output_col': "userIdIndex"
            },
            'product': {
                'input_col': "productId",
                'output_col': "productIdIndex"
            }
        }
        
        # Load models with validation
        self._load_and_validate_models()
        
        print("Recommender initialized successfully with verified column mappings")

    def _load_and_validate_models(self):
        """Load and validate all models"""
        # 1. Load ALS model
        self.als_model = ALSModel.load("models/als_model")
        
        # 2. Load and validate user indexer
        self.user_indexer = StringIndexerModel.load("models/user_indexer")
        self._validate_indexer(
            self.user_indexer,
            expected_input=self.expected_mappings['user']['input_col'],
            expected_output=self.expected_mappings['user']['output_col'],
            indexer_type="user"
        )
        
        # 3. Load and validate product indexer
        self.product_indexer = StringIndexerModel.load("models/product_indexer")
        self._validate_indexer(
            self.product_indexer,
            expected_input=self.expected_mappings['product']['input_col'],
            expected_output=self.expected_mappings['product']['output_col'],
            indexer_type="product"
        )

    def _validate_indexer(self, indexer, expected_input, expected_output, indexer_type):
        """Validate an indexer's configuration"""
        actual_input = indexer.getInputCol()
        actual_output = indexer.getOutputCol()
        
        if actual_input != expected_input or actual_output != expected_output:
            error_msg = f"""
            {indexer_type.capitalize()} indexer mismatch!
            Expected: {expected_input} -> {expected_output}
            Actual:   {actual_input} -> {actual_output}
            
            Solutions:
            1. Delete your models directory: rm -rf models/
            2. Retrain your models with matching column names
            3. Verify your data uses these exact column names:
               - User ID column: {expected_input}
               - Product ID column: {expected_output}
            """
            raise ValueError(error_msg)
            
    def get_als_recommendations(self, user_id, n=10):
        """Get ALS recommendations with proper column validation"""
        try:
            # Create DF with correct user ID column name
            user_df = self.spark.createDataFrame(
                [(user_id,)],
                [self.USER_ID_COL]
            )
            
            # Transform to numeric index
            indexed_user = self.user_indexer.transform(user_df)
            user_index = indexed_user.select(self.USER_INDEX_COL).first()[0]
            
            # Get recommendations
            als_recs = self.als_model.recommendForUserSubset(
                self.spark.createDataFrame([(user_index,)], [self.USER_INDEX_COL]),
                n
            )
            
            # Convert back to original product IDs
            recs = als_recs.first().recommendations
            product_indices = [r.__getattr__(self.PRODUCT_INDEX_COL) for r in recs]
            product_indices_df = self.spark.createDataFrame(
                [(idx,) for idx in product_indices],
                [self.PRODUCT_INDEX_COL]
            )
            
            return self.product_indexer.transform(product_indices_df)\
                .select(self.PRODUCT_ID_COL)
                
        except Exception as e:
            raise ValueError(
                f"ALS recommendation failed for user {user_id}\n"
                f"Error: {str(e)}\n"
                f"Verify your models expect: {self.USER_ID_COL}->{self.USER_INDEX_COL}"
            )

  
    def get_content_recommendations(self, user_id, n=10):
        """Get content-based recommendations"""
        try:
            from content_based import get_content_recommendations
            
            user_reviews = self.spark.read.parquet("data/cleaned_reviews.parquet")
            content_recs = get_content_recommendations(
                user_id, 
                user_reviews, 
                self.content_similarities, 
                n
            )
            
            return content_recs.select(col("similar_product_id").alias("product_id"))
            
        except Exception as e:
            raise ValueError(f"Error generating content recommendations: {str(e)}")
    
    def hybrid_recommend(self, user_id, n=10, als_weight=0.7, content_weight=0.3):
        """Get hybrid recommendations combining both approaches"""
        try:
            als_recs = self.get_als_recommendations(user_id, n)
            content_recs = self.get_content_recommendations(user_id, n)
            
            # Assign scores
            als_recs = als_recs.withColumn("als_score", lit(1.0))
            content_recs = content_recs.withColumn("content_score", lit(1.0))
            
            # Combine recommendations
            hybrid_recs = als_recs.join(
                content_recs,
                "product_id",
                "outer"
            ).fillna(0)
            
            # Calculate hybrid score
            hybrid_recs = hybrid_recs.withColumn(
                "hybrid_score",
                col("als_score") * als_weight + col("content_score") * content_weight
            )
            
            # Get top N recommendations with product details
            return hybrid_recs.orderBy(col("hybrid_score").desc()).limit(n).join(
                self.product_catalog,
                "product_id"
            ).select("product_id", "title", "price", "hybrid_score")
            
        except Exception as e:
            raise ValueError(f"Error generating hybrid recommendations: {str(e)}")
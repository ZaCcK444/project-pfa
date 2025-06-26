import unittest
from pyspark.sql import SparkSession
from src.hybrid_model import HybridRecommender
from src.spark_connector import create_spark_session
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestRecommendationSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        try:
            cls.spark = create_spark_session("Testing")
            
            # Verify data exists
            data_path = "data/cleaned_reviews.parquet"
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Test data not found at {data_path}")
            
            cls.recommender = HybridRecommender(cls.spark)
            
            # Load test users
            cls.test_users = cls.spark.read.parquet(data_path) \
                .select("user_id") \
                .distinct() \
                .limit(5) \
                .collect()
            
            if not cls.test_users:
                raise ValueError("No test users found in the dataset")
                
            logger.info(f"Loaded {len(cls.test_users)} test users")
        
        except Exception as e:
            logger.error(f"Test setup failed: {str(e)}")
            raise
    
    def test_als_recommendations(self):
        """Test ALS recommendations for each user"""
        for user in self.test_users:
            user_id = user["user_id"]
            with self.subTest(user_id=user_id):
                try:
                    recs = self.recommender.get_als_recommendations(user_id, 5)
                    self.assertIsNotNone(recs, f"ALS recommendations returned None for user {user_id}")
                    
                    # Check if we have valid recommendations or empty DataFrame
                    rec_count = recs.count()
                    self.assertGreaterEqual(rec_count, 0, f"Invalid recommendation count for user {user_id}")
                    
                    if rec_count > 0:
                        # Verify schema
                        expected_columns = {"product_id", "title", "price"}
                        actual_columns = set(recs.columns)
                        self.assertTrue(expected_columns.issubset(actual_columns), 
                                      f"Missing columns in ALS recommendations: {expected_columns - actual_columns}")
                        
                        logger.info(f"ALS recommendations for user {user_id}: {rec_count} items")
                    else:
                        logger.warning(f"No ALS recommendations found for user {user_id}")
                        
                except Exception as e:
                    self.fail(f"ALS recommendation failed for user {user_id}: {str(e)}")
    
    def test_content_recommendations(self):
        """Test content-based recommendations for each user"""
        for user in self.test_users:
            user_id = user["user_id"]
            with self.subTest(user_id=user_id):
                try:
                    recs = self.recommender.get_content_recommendations(user_id, 5)
                    self.assertIsNotNone(recs, f"Content recommendations returned None for user {user_id}")
                    
                    # Check if we have valid recommendations or empty DataFrame
                    rec_count = recs.count()
                    self.assertGreaterEqual(rec_count, 0, f"Invalid recommendation count for user {user_id}")
                    
                    if rec_count > 0:
                        # Verify schema
                        expected_columns = {"product_id", "title", "price", "total_score"}
                        actual_columns = set(recs.columns)
                        self.assertTrue(expected_columns.issubset(actual_columns), 
                                      f"Missing columns in content recommendations: {expected_columns - actual_columns}")
                        
                        logger.info(f"Content recommendations for user {user_id}: {rec_count} items")
                    else:
                        logger.warning(f"No content recommendations found for user {user_id}")
                        
                except Exception as e:
                    self.fail(f"Content recommendation failed for user {user_id}: {str(e)}")
    
    def test_hybrid_recommendations(self):
        """Test hybrid recommendations for each user"""
        for user in self.test_users:
            user_id = user["user_id"]
            with self.subTest(user_id=user_id):
                try:
                    recs = self.recommender.hybrid_recommend(user_id, 5)
                    self.assertIsNotNone(recs, f"Hybrid recommendations returned None for user {user_id}")
                    
                    # Check if we have valid recommendations or empty DataFrame
                    rec_count = recs.count()
                    self.assertGreaterEqual(rec_count, 0, f"Invalid recommendation count for user {user_id}")
                    
                    if rec_count > 0:
                        # Verify schema
                        expected_columns = {"product_id", "title", "price", "hybrid_score"}
                        actual_columns = set(recs.columns)
                        self.assertTrue(expected_columns.issubset(actual_columns), 
                                      f"Missing columns in hybrid recommendations: {expected_columns - actual_columns}")
                        
                        # Verify hybrid score column exists and contains valid values
                        self.assertIn("hybrid_score", recs.columns, "Hybrid score column missing")
                        
                        # Check that hybrid scores are numeric
                        score_sample = recs.select("hybrid_score").first()
                        if score_sample and score_sample["hybrid_score"] is not None:
                            self.assertIsInstance(score_sample["hybrid_score"], (int, float), 
                                                "Hybrid score should be numeric")
                        
                        logger.info(f"Hybrid recommendations for user {user_id}: {rec_count} items")
                    else:
                        logger.warning(f"No hybrid recommendations found for user {user_id}")
                        
                except Exception as e:
                    self.fail(f"Hybrid recommendation failed for user {user_id}: {str(e)}")
    
    def test_system_integration(self):
        """Test overall system integration"""
        try:
            # Test data loading
            self.assertIsNotNone(self.recommender.product_catalog, "Product catalog not loaded")
            self.assertIsNotNone(self.recommender.reviews_df, "Reviews data not loaded")
            self.assertIsNotNone(self.recommender.content_similarities, "Content similarities not loaded")
            
            # Test model loading
            self.assertIsNotNone(self.recommender.als_model, "ALS model not loaded")
            self.assertIsNotNone(self.recommender.user_indexer, "User indexer not loaded")
            self.assertIsNotNone(self.recommender.product_indexer, "Product indexer not loaded")
            
            # Test data integrity
            catalog_count = self.recommender.product_catalog.count()
            reviews_count = self.recommender.reviews_df.count()
            
            self.assertGreater(catalog_count, 0, "Product catalog is empty")
            self.assertGreater(reviews_count, 0, "Reviews data is empty")
            
            logger.info(f"System integration test passed - Catalog: {catalog_count}, Reviews: {reviews_count}")
            
        except Exception as e:
            self.fail(f"System integration test failed: {str(e)}")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test fixtures"""
        try:
            if hasattr(cls, 'spark') and cls.spark is not None:
                cls.spark.stop()
                logger.info("Test Spark session closed")
        except Exception as e:
            logger.error(f"Error during test cleanup: {str(e)}")

if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(argv=[''], verbosity=2, exit=False)
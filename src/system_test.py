import unittest
from pyspark.sql import SparkSession
from hybrid_model import HybridRecommender

class TestRecommendationSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spark = SparkSession.builder \
            .appName("Testing") \
            .getOrCreate()
        
        cls.recommender = HybridRecommender(cls.spark)
        
        # Load test data
        cls.test_users = cls.spark.read.parquet("data/cleaned_reviews.parquet") \
            .select("userId") \
            .distinct() \
            .limit(10) \
            .collect()
    
    def test_als_recommendations(self):
        for user in self.test_users:
            user_id = user["userId"]
            recs = self.recommender.get_als_recommendations(user_id)
            self.assertGreater(recs.count(), 0, f"No ALS recommendations for user {user_id}")
    
    def test_content_recommendations(self):
        for user in self.test_users:
            user_id = user["userId"]
            recs = self.recommender.get_content_recommendations(user_id)
            self.assertGreater(recs.count(), 0, f"No content recommendations for user {user_id}")
    
    def test_hybrid_recommendations(self):
        for user in self.test_users:
            user_id = user["userId"]
            recs = self.recommender.hybrid_recommend(user_id)
            self.assertGreater(recs.count(), 0, f"No hybrid recommendations for user {user_id}")
            self.assertTrue("hybrid_score" in recs.columns, "Hybrid score column missing")
    
    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

if __name__ == "__main__":
    unittest.main(argv=[''], verbosity=2, exit=False)
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import StringIndexer
from pyspark.sql import SparkSession
import os

def train_als_model(reviews_df):
    # Verify required columns exist
    required_columns = {'user_id', 'product_id', 'rating'}
    if not required_columns.issubset(set(reviews_df.columns)):
        missing = required_columns - set(reviews_df.columns)
        raise ValueError(f"Missing required columns: {missing}")

    # Split data into train and test sets
    (train, test) = reviews_df.randomSplit([0.8, 0.2], seed=42)
    
    # Build ALS model
    als = ALS(
        maxIter=5,
        regParam=0.01,
        userCol="user_id_index",  # Using indexed column
        itemCol="product_id_index",  # Using indexed column
        ratingCol="rating",
        coldStartStrategy="drop",
        nonnegative=True
    )
    
    # Train model
    model = als.fit(train)
    
    # Evaluate on test data
    predictions = model.transform(test)
    evaluator = RegressionEvaluator(
        metricName="rmse",
        labelCol="rating",
        predictionCol="prediction"
    )
    rmse = evaluator.evaluate(predictions)
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    
    # Generate recommendations
    user_recs = model.recommendForAllUsers(10)
    product_recs = model.recommendForAllItems(10)
    
    return model, user_recs, product_recs

if __name__ == "__main__":
    from spark_loader import load_data
    
    spark = None
    try:
        # Load data
        spark, reviews_df, _ = load_data()
        
        # Verify and show schema
        print("\nDataFrame Schema:")
        reviews_df.printSchema()
        
        # Create numeric indices for ALS
        user_indexer = StringIndexer(inputCol="user_id", outputCol="user_id_index")
        product_indexer = StringIndexer(inputCol="product_id", outputCol="product_id_index")
        
        # Fit and transform
        indexed_reviews = user_indexer.fit(reviews_df).transform(reviews_df)
        indexed_reviews = product_indexer.fit(indexed_reviews).transform(indexed_reviews)
        
        # Train model
        model, user_recs, product_recs = train_als_model(indexed_reviews)
        
        # Show sample recommendations
        print("\nTop 5 User Recommendations:")
        user_recs.show(5, truncate=False)
        
        print("\nTop 5 Product Recommendations:")
        product_recs.show(5, truncate=False)
        
        # Save model
        model_path = os.path.abspath("models/als_model")
        model.save(model_path)
        print(f"\nModel saved to: {model_path}")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
    finally:
        if spark:
            spark.stop()
            print("\nSpark session closed")
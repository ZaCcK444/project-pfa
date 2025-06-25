from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import StringIndexer
import os
import sys  # Make sure this is imported
from spark_connector import create_spark_session

def prepare_data(spark, data_path="data/cleaned_reviews.parquet"):
    """Load and prepare data with validation"""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}")
    
    df = spark.read.parquet(data_path)
    
    # Validate required columns
    required_cols = {"user_id", "product_id", "rating"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Create indices
    user_indexer = StringIndexer(inputCol="user_id", outputCol="user_id_index")
    product_indexer = StringIndexer(inputCol="product_id", outputCol="product_id_index")
    
    indexed_df = user_indexer.fit(df).transform(df)
    indexed_df = product_indexer.fit(indexed_df).transform(indexed_df)
    
    return indexed_df

def train_and_evaluate(df):
    """Train ALS model with evaluation"""
    train, test = df.randomSplit([0.8, 0.2], seed=42)
    
    als = ALS(
        maxIter=5,
        regParam=0.01,
        userCol="user_id_index",
        itemCol="product_id_index",
        ratingCol="rating",
        coldStartStrategy="drop",
        nonnegative=True,
        implicitPrefs=False
    )
    
    model = als.fit(train)
    predictions = model.transform(test)
    
    evaluator = RegressionEvaluator(
        metricName="rmse",
        labelCol="rating",
        predictionCol="prediction"
    )
    rmse = evaluator.evaluate(predictions)
    
    return model, rmse

def main():
    spark = None
    try:
        spark = create_spark_session("ALS_Recommender")
        
        print("Loading and preparing data...")
        df = prepare_data(spark)
        
        print("Training ALS model...")
        model, rmse = train_and_evaluate(df)
        print(f"Model trained successfully. RMSE: {rmse:.4f}")
        
        # Save model
        model_path = "models/als_model"
        model.save(model_path)
        print(f"Model saved to {model_path}")
        
        # Generate recommendations
        print("Generating recommendations...")
        user_recs = model.recommendForAllUsers(10)
        product_recs = model.recommendForAllItems(10)
        
        print("\nSample user recommendations:")
        user_recs.show(5, truncate=False)
        
    except Exception as e:
        print(f"\nERROR: {str(e)}", file=sys.stderr)
    finally:
        if spark:
            spark.stop()
            print("Spark session closed")

if __name__ == "__main__":
    main()
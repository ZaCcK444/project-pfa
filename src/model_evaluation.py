from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALSModel, ALS
import matplotlib.pyplot as plt
import numpy as np
from pyspark.sql import SparkSession
from src.utils import get_spark_config
from src.spark_connector import create_spark_session



def evaluate_model(model, test_data):
    # Make predictions
    predictions = model.transform(test_data)
    
    # Calculate RMSE
    evaluator = RegressionEvaluator(
        metricName="rmse",
        labelCol="rating",
        predictionCol="prediction"
    )
    rmse = evaluator.evaluate(predictions)
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    
    # Calculate MAE
    evaluator.setMetricName("mae")
    mae = evaluator.evaluate(predictions)
    print(f"Mean Absolute Error (MAE): {mae}")
    
    # Convert to Pandas for visualization
    pred_pandas = predictions.select(['rating', 'prediction']).toPandas()
    
    # Scatter plot of actual vs predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(pred_pandas['rating'], pred_pandas['prediction'], alpha=0.3)
    plt.plot([1, 5], [1, 5], 'r--')  # Perfect prediction line
    plt.xlabel('Actual Rating')
    plt.ylabel('Predicted Rating')
    plt.title('Actual vs Predicted Ratings')
    plt.grid()
    plt.show()
    
    # Error distribution
    pred_pandas['error'] = pred_pandas['rating'] - pred_pandas['prediction']
    plt.figure(figsize=(10, 6))
    plt.hist(pred_pandas['error'], bins=50)
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Errors')
    plt.grid()
    plt.show()

def hyperparameter_tuning(train_data):
    # Define parameter grid
    ranks = [10, 50, 100]
    regParams = [0.01, 0.1, 1.0]
    maxIters = [5, 10]
    
    best_model = None
    best_rmse = float('inf')
    best_params = {}
    
    for rank in ranks:
        for regParam in regParams:
            for maxIter in maxIters:
                als = ALS(
                    rank=rank,
                    maxIter=maxIter,
                    regParam=regParam,
                    userCol="userIdIndex",
                    itemCol="productIdIndex",
                    ratingCol="rating",
                    coldStartStrategy="drop",
                    nonnegative=True
                )
                
                model = als.fit(train_data)
                predictions = model.transform(train_data)
                evaluator = RegressionEvaluator(
                    metricName="rmse",
                    labelCol="rating",
                    predictionCol="prediction"
                )
                rmse = evaluator.evaluate(predictions)
                
                print(f"Rank: {rank}, RegParam: {regParam}, MaxIter: {maxIter} => RMSE: {rmse}")
                
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_model = model
                    best_params = {
                        'rank': rank,
                        'regParam': regParam,
                        'maxIter': maxIter
                    }
    
    print("\nBest parameters:", best_params)
    print("Best RMSE:", best_rmse)
    
    return best_model
if __name__ == "__main__":
    spark = create_spark_session("ModelEvaluation")


if __name__ == "__main__":
    spark = create_spark_session("ModelEvaluation")
    try:
        from src.spark_loader import load_data
        _, reviews_df, _ = load_data()
        
    if __name__ == "__main__":
    conf = get_spark_config("ModelEvaluation")
    spark = SparkSession.builder \
        .config(conf=conf) \
        .getOrCreate()

    try:
        from src.spark_loader import load_data
        _, reviews_df, _ = load_data()
        
        # Load indexed data (from Week 5)
        from pyspark.ml.feature import StringIndexer
        
        user_indexer = StringIndexer(inputCol="user_id", outputCol="userIdIndex").fit(reviews_df)
        product_indexer = StringIndexer(inputCol="product_id", outputCol="productIdIndex").fit(reviews_df)
        
        indexed_reviews = user_indexer.transform(reviews_df)
        indexed_reviews = product_indexer.transform(indexed_reviews)
        
        # Split data
        (train, test) = indexed_reviews.randomSplit([0.8, 0.2], seed=42)
        
        # Option 1: Evaluate existing model
        print("\nEvaluating existing model...")
        model = ALSModel.load("models/als_model")
        evaluate_model(model, test)
        
        # Option 2: Hyperparameter tuning (takes longer)
        print("\nRunning hyperparameter tuning...")
        best_model = hyperparameter_tuning(train)
        evaluate_model(best_model, test)
        
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
    finally:
        spark.stop()
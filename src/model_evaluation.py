from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALSModel, ALS
from pyspark.ml.feature import StringIndexer
import matplotlib.pyplot as plt
import numpy as np
import os
from src.utils import get_spark_config
from src.spark_connector import create_spark_session

def evaluate_model(model, test_data):
    """Evaluate ALS model with comprehensive metrics"""
    try:
        # Make predictions
        predictions = model.transform(test_data)
        
        # Remove null predictions for evaluation
        predictions_clean = predictions.filter(predictions.prediction.isNotNull())
        
        if predictions_clean.count() == 0:
            print("No valid predictions to evaluate")
            return None, None
        
        # Calculate RMSE
        evaluator = RegressionEvaluator(
            metricName="rmse",
            labelCol="rating",
            predictionCol="prediction"
        )
        rmse = evaluator.evaluate(predictions_clean)
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        
        # Calculate MAE
        evaluator.setMetricName("mae")
        mae = evaluator.evaluate(predictions_clean)
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        
        # Convert to Pandas for visualization (sample to avoid memory issues)
        sample_size = min(10000, predictions_clean.count())
        pred_pandas = predictions_clean.sample(False, sample_size / predictions_clean.count()) \
                                      .select(['rating', 'prediction']).toPandas()
        
        # Scatter plot of actual vs predicted
        plt.figure(figsize=(10, 6))
        plt.scatter(pred_pandas['rating'], pred_pandas['prediction'], alpha=0.3)
        plt.plot([1, 5], [1, 5], 'r--', label='Perfect Prediction')
        plt.xlabel('Actual Rating')
        plt.ylabel('Predicted Rating')
        plt.title('Actual vs Predicted Ratings')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        # Error distribution
        pred_pandas['error'] = pred_pandas['rating'] - pred_pandas['prediction']
        plt.figure(figsize=(10, 6))
        plt.hist(pred_pandas['error'], bins=50, alpha=0.7)
        plt.xlabel('Prediction Error')
        plt.ylabel('Frequency')
        plt.title('Distribution of Prediction Errors')
        plt.axvline(x=0, color='r', linestyle='--', label='Perfect Prediction')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        return rmse, mae
        
    except Exception as e:
        print(f"Error in model evaluation: {str(e)}")
        return None, None

def hyperparameter_tuning(train_data, validation_data):
    """Perform hyperparameter tuning with cross-validation"""
    # Define parameter grid
    ranks = [10, 50, 100]
    regParams = [0.01, 0.1, 1.0]
    maxIters = [5, 10]
    
    best_model = None
    best_rmse = float('inf')
    best_params = {}
    
    print("Starting hyperparameter tuning...")
    print("=" * 50)
    
    for rank in ranks:
        for regParam in regParams:
            for maxIter in maxIters:
                try:
                    print(f"Testing: rank={rank}, regParam={regParam}, maxIter={maxIter}")
                    
                    als = ALS(
                        rank=rank,
                        maxIter=maxIter,
                        regParam=regParam,
                        userCol="user_id_index",
                        itemCol="product_id_index",
                        ratingCol="rating",
                        coldStartStrategy="drop",
                        nonnegative=True
                    )
                    
                    model = als.fit(train_data)
                    predictions = model.transform(validation_data)
                    
                    # Clean predictions
                    predictions_clean = predictions.filter(predictions.prediction.isNotNull())
                    
                    if predictions_clean.count() == 0:
                        print(f"  No valid predictions - skipping")
                        continue
                        
                    evaluator = RegressionEvaluator(
                        metricName="rmse",
                        labelCol="rating",
                        predictionCol="prediction"
                    )
                    rmse = evaluator.evaluate(predictions_clean)
                    
                    print(f"  RMSE: {rmse:.4f}")
                    
                    if rmse < best_rmse:
                        best_rmse = rmse
                        best_model = model
                        best_params = {
                            'rank': rank,
                            'regParam': regParam,
                            'maxIter': maxIter
                        }
                        print(f"  ✓ New best model!")
                    
                except Exception as e:
                    print(f"  Error: {str(e)}")
                    continue
    
    print("=" * 50)
    print(f"Best parameters: {best_params}")
    print(f"Best RMSE: {best_rmse:.4f}")
    
    return best_model, best_params, best_rmse

def main():
    """Main evaluation function"""
    spark = None
    try:
        spark = create_spark_session("ModelEvaluation")
        
        # Load and prepare data
        reviews_path = "data/cleaned_reviews.parquet"
        if not os.path.exists(reviews_path):
            raise FileNotFoundError(f"Reviews data not found at {reviews_path}")
            
        reviews_df = spark.read.parquet(reviews_path)
        
        # Validate required columns
        required_cols = {"user_id", "product_id", "rating"}
        missing = required_cols - set(reviews_df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        print("Creating user and product indices...")
        
        # Create and fit indexers
        user_indexer = StringIndexer(inputCol="user_id", outputCol="user_id_index")
        product_indexer = StringIndexer(inputCol="product_id", outputCol="product_id_index")
        
        user_indexer_model = user_indexer.fit(reviews_df)
        product_indexer_model = product_indexer.fit(reviews_df)
        
        # Transform data
        indexed_reviews = user_indexer_model.transform(reviews_df)
        indexed_reviews = product_indexer_model.transform(indexed_reviews)
        
        # Split data
        print("Splitting data...")
        (train, validation, test) = indexed_reviews.randomSplit([0.6, 0.2, 0.2], seed=42)
        
        print(f"Train: {train.count()}, Validation: {validation.count()}, Test: {test.count()}")
        
        # Option 1: Evaluate existing model if available
        model_path = "models/als_model"
        if os.path.exists(model_path):
            print("\nEvaluating existing model...")
            try:
                model = ALSModel.load(model_path)
                rmse, mae = evaluate_model(model, test)
                if rmse is not None:
                    print(f"Existing model RMSE: {rmse:.4f}, MAE: {mae:.4f}")
            except Exception as e:
                print(f"Could not load existing model: {str(e)}")
        
        # Option 2: Hyperparameter tuning
        print("\nRunning hyperparameter tuning...")
        best_model, best_params, best_rmse = hyperparameter_tuning(train, validation)
        
        if best_model is not None:
            print("\nEvaluating best model on test set...")
            test_rmse, test_mae = evaluate_model(best_model, test)
            print(f"Final test RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}")
            
            # Save the best model
            models_dir = "models"
            if not os.path.exists(models_dir):
                os.makedirs(models_dir)
                
            best_model.write().overwrite().save(os.path.join(models_dir, "best_als_model"))
            print("Best model saved!")
        else:
            print("No valid model found during tuning")
        
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        raise
    finally:
        if spark is not None:
            spark.stop()
            print("Spark session closed")

if __name__ == "__main__":
    main()
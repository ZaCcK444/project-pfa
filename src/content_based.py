from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer, IDF
from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, udf, explode
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.feature import MinMaxScaler, VectorAssembler
import pyspark.sql.functions as F
from pyspark.sql.types import DoubleType, ArrayType, FloatType
import numpy as np

def preprocess_product_data(product_catalog):
    """Enhanced product data preprocessing with additional features"""
    # Clean title text
    product_catalog = product_catalog.withColumn(
        "cleaned_title",
        F.regexp_replace(F.lower(col("title")), "[^a-zA-Z0-9\\s]", "")
    )
    
    # Text processing pipeline
    tokenizer = Tokenizer(inputCol="cleaned_title", outputCol="words")
    remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
    countVectorizer = CountVectorizer(inputCol="filtered_words", outputCol="raw_features", 
                                    vocabSize=1000, minDF=2)
    idf = IDF(inputCol="raw_features", outputCol="tfidf_features")
    
    # Price processing
    assembler = VectorAssembler(inputCols=["price"], outputCol="price_vec")
    scaler = MinMaxScaler(inputCol="price_vec", outputCol="scaled_price")
    
    # Build and execute pipeline
    pipeline = Pipeline(stages=[
        tokenizer,
        remover,
        countVectorizer,
        idf,
        assembler,
        scaler
    ])
    
    model = pipeline.fit(product_catalog)
    return model.transform(product_catalog)

def calculate_similarity(processed_data):
    """Optimized similarity calculation with LSH approximation"""
    try:
        # Combine features first
        assembler = VectorAssembler(
            inputCols=["tfidf_features", "scaled_price"],
            outputCol="combined_features"
        )
        processed_data = assembler.transform(processed_data)
        
        # Use LSH for approximate similarity joins
        from pyspark.ml.feature import BucketedRandomProjectionLSH
        brp = BucketedRandomProjectionLSH(
            inputCol="combined_features",
            outputCol="hashes",
            bucketLength=2.0,
            numHashTables=3
        )
        
        model = brp.fit(processed_data)
        similarities = model.approxSimilarityJoin(
            processed_data, 
            processed_data, 
            2.0, 
            distCol="distance"
        )
        
        # Convert distance to similarity (1-distance)
        similarities = similarities.filter(
            col("datasetA.product_id") < col("datasetB.product_id")
        ).select(
            col("datasetA.product_id").alias("product1"),
            col("datasetB.product_id").alias("product2"),
            (1 - col("distance")).alias("similarity")
        ).filter(col("similarity") > 0.1)
        
        # Get top 10 similar products for each item
        top_similarities = similarities.groupBy("product1").agg(
            F.collect_list(
                F.struct(
                    col("product2").alias("similar_productId"),
                    col("similarity").alias("score")
                )
            ).alias("similarities")
        ).withColumn(
            "sorted_similarities",
            F.expr("slice(array_sort(similarities, (a, b) -> case when a.score > b.score then -1 else 1 end), 1, 10)")
        ).select("product1", "sorted_similarities")
        
        return top_similarities
        
    except Exception as e:
        print(f"Error in similarity calculation: {str(e)}")
        raise

def get_content_recommendations(user_id, user_reviews, content_similarities, n=5):
    """Improved recommendation logic without driver collection"""
    # Get user's liked products with weights based on rating
    liked_products = user_reviews.filter(col("user_id") == user_id) \
        .select("product_id", "rating") \
        .withColumn("weight", (col("rating") - 3) / 2)  # Normalize 1-5 to 0-1
    
    # Join with similarities and apply rating weights
    recommendations = liked_products.join(
        content_similarities,
        liked_products["product_id"] == content_similarities["product1"],
        "left"
    ).select(
        "product_id",
        "weight",
        "sorted_similarities"
    )
    
    # Explode and calculate weighted scores
    recommendations = recommendations.withColumn(
        "rec",
        F.explode(col("sorted_similarities"))
    ).select(
        "product_id",
        "weight",
        col("rec.similar_productId").alias("recommended_product"),
        (col("rec.score") * col("weight")).alias("weighted_score")
    )
    
    # Create DF of user's rated products instead of collecting to driver
    user_rated_products = user_reviews.filter(col("user_id") == user_id) \
        .select("product_id").distinct().withColumnRenamed("product_id", "rated_product")
    
    # Filter using left anti join to avoid collecting to driver
    return recommendations.join(
        user_rated_products,
        col("recommended_product") == col("rated_product"),
        "left_anti"
    ).groupBy("recommended_product") \
        .agg(F.sum("weighted_score").alias("total_score")) \
        .orderBy(col("total_score").desc()) \
        .limit(n)

if __name__ == "__main__":
    from spark_loader import load_data
    
    spark = SparkSession.builder \
        .appName("ContentBasedRecSys") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .config("spark.driver.memory", "8g") \
        .config("spark.executor.memory", "8g") \
        .config("spark.sql.shuffle.partitions", "200") \
        .config("spark.sql.execution.pyspark.udf.faulthandler.enabled", "true") \
        .getOrCreate()
    
    try:
        # Load data
        _, reviews_df, product_catalog = load_data()
        
        # Preprocess product data
        print("Preprocessing product data...")
        processed_data = preprocess_product_data(product_catalog.limit(1000))
        
        # Calculate similarities
        print("Calculating product similarities...")
        content_similarities = calculate_similarity(processed_data)
        
        # Save similarities
        content_similarities.write.parquet(
            "data/content_similarities.parquet", 
            mode="overwrite"
        )
        
        # Get recommendations for a sample user
        sample_user = reviews_df.select("user_id").first()[0]
        user_recs = get_content_recommendations(
            sample_user, 
            reviews_df, 
            content_similarities
        )
        
        print(f"Content-based recommendations for user {sample_user}:")
        user_recs.show()
        
    except Exception as e:
        print(f"Error: {str(e)}")
        raise
    finally:
        spark.stop()
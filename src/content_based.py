from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, udf, explode
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.feature import MinMaxScaler, VectorAssembler
import pyspark.sql.functions as F
from pyspark.sql.types import DoubleType, ArrayType, FloatType
import numpy as np

def preprocess_product_data(product_catalog):
    # Clean title text
    product_catalog = product_catalog.withColumn(
        "cleaned_title",
        F.regexp_replace(F.lower(col("title")), "[^a-zA-Z0-9\\s]", "")
    )
    
    # Text processing pipeline
    tokenizer = Tokenizer(inputCol="cleaned_title", outputCol="words")
    remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
    hashingTF = HashingTF(inputCol="filtered_words", outputCol="raw_features", numFeatures=1000)
    idf = IDF(inputCol="raw_features", outputCol="tfidf_features")
    
    # Price processing
    assembler = VectorAssembler(inputCols=["price"], outputCol="price_vec")
    scaler = MinMaxScaler(inputCol="price_vec", outputCol="scaled_price")
    
    # Build pipeline
    pipeline = Pipeline(stages=[
        tokenizer,
        remover,
        hashingTF,
        idf,
        assembler,
        scaler
    ])
    
    model = pipeline.fit(product_catalog)
    processed_data = model.transform(product_catalog)
    
    return processed_data

def calculate_similarity(processed_data):
    # First, collect the product features to driver
    product_features = processed_data.select("productId", "tfidf_features", "scaled_price").collect()
    
    # Create a broadcast variable with the features
    features_map = {row.productId: (row.tfidf_features, row.scaled_price[0]) for row in product_features}
    broadcast_features = SparkSession.getActiveSession().sparkContext.broadcast(features_map)
    
    # Define a more efficient cosine similarity UDF
    def cosine_sim_udf(product1, product2):
        features = broadcast_features.value
        try:
            vec1, price1 = features[product1]
            vec2, price2 = features[product2]
            
            # Combine TF-IDF and price into a single vector
            combined1 = np.append(vec1.toArray(), price1)
            combined2 = np.append(vec2.toArray(), price2)
            
            # Calculate cosine similarity
            dot_product = np.dot(combined1, combined2)
            norm1 = np.linalg.norm(combined1)
            norm2 = np.linalg.norm(combined2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return float(dot_product / (norm1 * norm2))
        except:
            return 0.0
    
    # Register the UDF
    cosine_sim = udf(cosine_sim_udf, FloatType())
    
    # Create a DataFrame with all product pairs (without cross join)
    products = processed_data.select("productId").distinct()
    products_list = products.collect()
    
    # Create pairs using DataFrame operations instead of cross join
    from itertools import combinations
    pairs = []
    for p1, p2 in combinations(products_list, 2):
        pairs.append((p1.productId, p2.productId))
    
    pairs_df = SparkSession.getActiveSession().createDataFrame(pairs, ["product1", "product2"])
    
    # Calculate similarities
    similarities = pairs_df.withColumn(
        "similarity",
        cosine_sim(col("product1"), col("product2"))
    ).filter(col("similarity") > 0.1)  # Filter out very low similarities
    
    # Get top similar products for each product
    top_similarities = similarities.union(
        similarities.select(
            col("product2").alias("product1"),
            col("product1").alias("product2"),
            col("similarity")
        )
    ).groupBy("product1").agg(
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

def get_content_recommendations(user_id, user_reviews, content_similarities, n=5):
    # Get products the user has rated highly (>=4 stars)
    liked_products = user_reviews.filter(col("userId") == user_id) \
        .filter(col("rating") >= 4) \
        .select("productId") \
        .distinct()
    
    # Get similar products to the liked ones
    recommendations = liked_products.join(
        content_similarities,
        liked_products["productId"] == content_similarities["product1"],
        "left"
    ).select("sorted_similarities")
    
    # Explode and aggregate recommendations
    recommendations = recommendations.withColumn(
        "rec",
        F.explode(col("sorted_similarities"))
    ).select("rec.*") \
     .groupBy("similar_productId") \
     .agg(F.avg("score").alias("avg_score")) \
     .orderBy(col("avg_score").desc()) \
     .limit(n)
    
    return recommendations

if __name__ == "__main__":
    from spark_loader import load_data
    
    spark = SparkSession.builder \
        .appName("ContentBasedRecSys") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .config("spark.sql.shuffle.partitions", "100") \
        .getOrCreate()
    
    try:
        # Load data with smaller subset for testing
        _, reviews_df, product_catalog = load_data()
        product_catalog = product_catalog.limit(1000)  # Limit for testing
        
        # Preprocess product data
        print("Preprocessing product data...")
        processed_data = preprocess_product_data(product_catalog)
        
        # Calculate similarities with smaller batch
        print("Calculating product similarities...")
        content_similarities = calculate_similarity(processed_data)
        
        # Save similarities
        content_similarities.write.parquet("data/content_similarities.parquet", mode="overwrite")
        
        # Example: Get recommendations for a user
        sample_user = reviews_df.select("userId").first()[0]
        user_recs = get_content_recommendations(sample_user, reviews_df, content_similarities)
        
        print(f"Content-based recommendations for user {sample_user}:")
        user_recs.show()
        
    except Exception as e:
        print(f"Error: {str(e)}")
        raise
    finally:
        spark.stop()
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, udf
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.feature import MinMaxScaler, VectorAssembler
import pyspark.sql.functions as F
from pyspark.sql.types import DoubleType, ArrayType, FloatType

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
    
    # Price processing - convert to vector directly
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
    # Extract price value from scaled_price vector
    def extract_price(v):
        return float(v[0]) if v else None
    
    extract_price_udf = udf(extract_price, DoubleType())
    
    # Combine features
    def combine_features(tfidf, price):
        if tfidf is None or price is None:
            return None
        tfidf_array = tfidf.toArray().tolist()
        return Vectors.dense(tfidf_array + [price])
    
    combine_features_udf = udf(combine_features, VectorUDT())
    
    processed_data = processed_data.withColumn(
        "price_value",
        extract_price_udf(col("scaled_price"))
    ).withColumn(
        "features",
        combine_features_udf(col("tfidf_features"), col("price_value"))
    ).drop("price_value")
    
    # Cross join to get all product pairs
    products = processed_data.select("productId", "features").alias("p1")
    products_cross = processed_data.select("productId", "features").alias("p2")
    
    pairs = products.crossJoin(products_cross) \
        .filter(col("p1.productId") != col("p2.productId"))
    
    # Calculate cosine similarity
    def cosine_sim(v1, v2):
        if v1 is None or v2 is None:
            return None
        return float(v1.dot(v2) / (v1.norm(2) * v2.norm(2)))
    
    cosine_sim_udf = udf(cosine_sim, DoubleType())
    
    similarities = pairs.withColumn(
        "similarity",
        cosine_sim_udf(col("p1.features"), col("p2.features"))
    ).filter(col("similarity").isNotNull())
    
    # Get top similar products for each product
    top_similarities = similarities.groupBy("p1.productId") \
        .agg(F.collect_list(
            F.struct(
                col("p2.productId").alias("similar_productId"),
                col("similarity").alias("score")
            )
        ).alias("similarities")) \
        .withColumn(
            "sorted_similarities",
            F.expr("slice(array_sort(similarities, (a, b) -> case when a.score > b.score then -1 else 1 end), 1, 10)")
        ) \
        .select("productId", "sorted_similarities")
    
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
        liked_products["productId"] == content_similarities["productId"]
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
    
    spark, reviews_df, product_catalog = load_data()
    
    # Preprocess product data
    processed_data = preprocess_product_data(product_catalog)
    
    # Calculate similarities
    content_similarities = calculate_similarity(processed_data)
    
    # Save similarities
    content_similarities.write.parquet("data/content_similarities.parquet", mode="overwrite")
    
    # Example: Get recommendations for a user
    sample_user = reviews_df.select("userId").first()[0]
    user_recs = get_content_recommendations(sample_user, reviews_df, content_similarities)
    
    print(f"Content-based recommendations for user {sample_user}:")
    user_recs.show()
    
    spark.stop()
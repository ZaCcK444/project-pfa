from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, ArrayType
from pyspark.sql.window import Window
from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer, IDF, MinMaxScaler, VectorAssembler
from pyspark.ml import Pipeline
import logging
from src.spark_connector import create_spark_session


# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preprocess_product_data(product_catalog):
    """Preprocess product data with robust error handling"""
    try:
        # Validate input schema
        required_columns = {'product_id', 'title', 'price'}
        if not required_columns.issubset(set(product_catalog.columns)):
            missing = required_columns - set(product_catalog.columns)
            raise ValueError(f"Missing required columns: {missing}")

        # Clean title text (safe for nulls)
        product_catalog = product_catalog.withColumn(
            "cleaned_title",
            F.regexp_replace(F.lower(F.coalesce(F.col("title"), F.lit("")), "[^a-zA-Z0-9\\s]", "")
        ).filter(F.length(F.col("cleaned_title")) > 0)
       )
        # Text processing pipeline
        tokenizer = Tokenizer(inputCol="cleaned_title", outputCol="words")
        remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
        count_vectorizer = CountVectorizer(
            inputCol="filtered_words",
            outputCol="raw_features",
            vocabSize=1000,
            minDF=2
        )
        idf = IDF(inputCol="raw_features", outputCol="tfidf_features")
        
        # Numeric feature processing
        assembler = VectorAssembler(inputCols=["price"], outputCol="price_vec")
        scaler = MinMaxScaler(inputCol="price_vec", outputCol="scaled_price")

        pipeline = Pipeline(stages=[tokenizer, remover, count_vectorizer, idf, assembler, scaler])
        return pipeline.fit(product_catalog).transform(product_catalog)

    except Exception as e:
        logger.error(f"Product preprocessing failed: {str(e)}", exc_info=True)
        raise

def calculate_similarity(processed_data):
    """Calculate product similarities with LSH and proper error handling"""
    try:
        # Feature combination
        assembler = VectorAssembler(
            inputCols=["tfidf_features", "scaled_price"],
            outputCol="combined_features"
        )
        processed_data = assembler.transform(processed_data)

        # Approximate similarity join (LSH)
        from pyspark.ml.feature import BucketedRandomProjectionLSH
        brp = BucketedRandomProjectionLSH(
            inputCol="combined_features",
            outputCol="hashes",
            bucketLength=2.0,
            numHashTables=3
        )
        model = brp.fit(processed_data)
        
        # Calculate similarities between products
        similarities = model.approxSimilarityJoin(
            processed_data, processed_data, 2.0, distCol="distance"
        ).filter(
            F.col("datasetA.product_id") < F.col("datasetB.product_id")  # Avoid duplicates
        ).select(
            F.col("datasetA.product_id").alias("product1"),
            F.col("datasetB.product_id").alias("product2"),
            (1 - F.col("distance")).alias("similarity")
        ).filter(F.col("similarity") > 0.1)  # Minimum similarity threshold

        # Get top 10 similar items per product
        window_spec = Window.partitionBy("product1").orderBy(F.col("similarity").desc())
        top_similarities = similarities.withColumn("rank", F.rank().over(window_spec)) \
            .filter(F.col("rank") <= 10) \
            .groupBy("product1") \
            .agg(F.collect_list(F.struct("product2", "similarity")).alias("similar_items"))

        return top_similarities

    except Exception as e:
        logger.error(f"Similarity calculation failed: {str(e)}", exc_info=True)
        raise

def get_content_recommendations(user_id, user_reviews, product_catalog, content_similarities, n=5):
    """Generate content-based recommendations with proper fallbacks"""
    try:
        # Schema for empty results
        result_schema = StructType([
            StructField("product_id", StringType()),
            StructField("title", StringType()),
            StructField("price", DoubleType()),
            StructField("score", DoubleType())
        ])

        # Validate inputs
        if user_reviews is None or product_catalog is None or content_similarities is None:
            raise ValueError("Input DataFrames cannot be None")
            
        if not isinstance(user_id, str):
            raise TypeError("user_id must be a string")

        # Check if user exists
        user_exists = user_reviews.filter(F.col("user_id") == user_id).count() > 0
        if not user_exists:
            logger.warning(f"No reviews found for user {user_id}")
            return user_reviews.sparkSession.createDataFrame([], result_schema)

        # Get user's liked products with weights (normalized rating)
        liked_products = user_reviews.filter(F.col("user_id") == user_id) \
            .select("product_id", "rating") \
            .withColumn("weight", (F.col("rating") - 3) / 2)  # Convert 1-5 scale to -1 to +1

        # Join with similarities and calculate recommendation scores
        recommendations = liked_products.join(
            content_similarities,
            liked_products["product_id"] == content_similarities["product1"],
            "inner"
        ).select(
            "product_id",
            "weight",
            F.explode("similar_items").alias("similar_item")
        ).select(
            "product_id",
            "weight",
            F.col("similar_item.product2").alias("recommended_product"),
            (F.col("similar_item.similarity") * F.col("weight")).alias("score")
        )

        # Filter out already rated products
        rated_products = user_reviews.filter(F.col("user_id") == user_id) \
            .select("product_id").distinct()
            
        final_recs = recommendations.join(
            rated_products,
            recommendations["recommended_product"] == rated_products["product_id"],
            "left_anti"
        ).groupBy("recommended_product") \
         .agg(F.sum("score").alias("total_score")) \
         .orderBy(F.col("total_score").desc()) \
         .limit(n)

        # Join with product catalog to get product details
        if final_recs.count() > 0:
            final_recs = final_recs.join(
                product_catalog.select("product_id", "title", "price"),
                final_recs["recommended_product"] == product_catalog["product_id"],
                "inner"
            ).select(
                "product_id",
                "title",
                "price",
                "total_score"
            )
        else:
            logger.info(f"No recommendations generated for user {user_id}")
            return user_reviews.sparkSession.createDataFrame([], result_schema)

        return final_recs

    except Exception as e:
        logger.error(f"Recommendation failed for user {user_id}: {str(e)}", exc_info=True)
        return user_reviews.sparkSession.createDataFrame([], result_schema)
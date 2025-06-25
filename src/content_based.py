from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer, IDF, MinMaxScaler, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, ArrayType
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preprocess_product_data(product_catalog):
    """Safely preprocess product data with validation and error handling"""
    try:
        # Validate input schema
        required_columns = {'product_id', 'title', 'price'}
        if not required_columns.issubset(set(product_catalog.columns)):
            missing = required_columns - set(product_catalog.columns)
            raise ValueError(f"Missing required columns: {missing}")

        # Clean title text (safe for nulls)
        product_catalog = product_catalog.withColumn(
            "cleaned_title",
            F.regexp_replace(F.lower(F.col("title")), "[^a-zA-Z0-9\\s]", "")
        ).filter(F.col("cleaned_title").isNotNull())

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
        logger.error(f"Product preprocessing failed: {str(e)}")
        raise

def calculate_similarity(processed_data):
    """Crash-resistant similarity calculation with LSH"""
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
        
        similarities = model.approxSimilarityJoin(
            processed_data, processed_data, 2.0, distCol="distance"
        ).filter(
            F.col("datasetA.product_id") < F.col("datasetB.product_id")
        ).select(
            F.col("datasetA.product_id").alias("product1"),
            F.col("datasetB.product_id").alias("product2"),
            (1 - F.col("distance")).alias("similarity")
        ).filter(F.col("similarity") > 0.1)

        # Get top 10 similar items per product
        top_similarities = similarities.groupBy("product1").agg(
            F.collect_list(F.struct("product2", "similarity")).alias("similarities")
        ).withColumn(
            "sorted_similarities",
            F.expr("slice(array_sort(similarities, (a, b) -> CASE WHEN a.similarity > b.similarity THEN -1 ELSE 1 END), 1, 10)")
        ).select("product1", "sorted_similarities")

        return top_similarities

    except Exception as e:
        logger.error(f"Similarity calculation failed: {str(e)}")
        raise

def get_content_recommendations(user_id, user_reviews, content_similarities, n=5):
    """Robust recommendation generation with fallbacks"""
    try:
        # Schema for empty results
        result_schema = StructType([
            StructField("product_id", StringType()),
            StructField("title", StringType()),
            StructField("price", DoubleType()),
            StructField("total_score", DoubleType())
        ])

        # Validate inputs
        if user_reviews.filter(F.col("user_id") == user_id).count() == 0:
            logger.warning(f"No reviews found for user {user_id}")
            return user_reviews.sparkSession.createDataFrame([], result_schema)

        # Get user's liked products with weights
        liked_products = user_reviews.filter(F.col("user_id") == user_id) \
            .select("product_id", "rating") \
            .withColumn("weight", (F.col("rating") - 3) / 2) \
            .filter(F.col("weight").isNotNull())

        # Explode similarities and calculate scores
        recommendations = liked_products.join(
            content_similarities,
            liked_products["product_id"] == content_similarities["product1"],
            "left"
        ).select(
            "product_id",
            "weight",
            F.explode(F.col("sorted_similarities")).alias("rec")
        ).select(
            "product_id",
            "weight",
            F.col("rec.product2").alias("recommended_product"),
            (F.col("rec.similarity") * F.col("weight")).alias("weighted_score")
        )

        # Filter out already rated products
        user_rated = user_reviews.filter(F.col("user_id") == user_id) \
            .select("product_id").distinct()

        final_recs = recommendations.join(
            user_rated,
            F.col("recommended_product") == F.col("product_id"),
            "left_anti"
        ).groupBy("recommended_product") \
         .agg(F.sum("weighted_score").alias("total_score")) \
         .orderBy(F.col("total_score").desc()) \
         .limit(n)

        return final_recs

    except Exception as e:
        logger.error(f"Recommendation failed for user {user_id}: {str(e)}")
        return user_reviews.sparkSession.createDataFrame([], result_schema)
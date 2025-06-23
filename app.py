import streamlit as st
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import pandas as pd
import sys
from pathlib import Path

# Add src directory to Python path
sys.path.append(str(Path(__file__).parent / "src"))
from hybrid_model import HybridRecommender

# Initialize Spark session
@st.cache_resource
def init_spark():
    spark = SparkSession.builder \
        .appName("StreamlitApp") \
        .config("spark.driver.memory", "2g") \
        .getOrCreate()
    return spark

spark = init_spark()

# Initialize recommender
@st.cache_resource
def init_recommender(spark):
    return HybridRecommender(spark)

recommender = init_recommender(spark)

# Load user list
@st.cache_data
def load_users():
    try:
        reviews = spark.read.parquet("data/cleaned_reviews.parquet")
        users = reviews.select("userId").distinct().toPandas()["userId"].tolist()
        return users
    except Exception as e:
        st.error(f"Error loading user data: {str(e)}")
        return []

# Streamlit app
st.title("E-commerce Product Recommender")

# User selection
user_list = load_users()
if not user_list:
    st.error("No users found. Please check your data files.")
    st.stop()

selected_user = st.selectbox("Select a user:", user_list)

# Number of recommendations
num_recs = st.slider("Number of recommendations:", 5, 20, 10)

# Recommendation type
rec_type = st.radio("Recommendation type:", 
                   ["Hybrid", "Collaborative Filtering", "Content-Based"])

# Get recommendations button
if st.button("Get Recommendations"):
    with st.spinner("Generating recommendations..."):
        try:
            if rec_type == "Hybrid":
                recommendations = recommender.hybrid_recommend(selected_user, num_recs)
            elif rec_type == "Collaborative Filtering":
                recommendations = recommender.get_als_recommendations(selected_user, num_recs)
                recommendations = recommendations.join(
                    spark.read.parquet("data/product_catalog.parquet"),
                    "productId"
                ).select("productId", "title", "price")
            else:
                reviews = spark.read.parquet("data/cleaned_reviews.parquet")
                similarities = spark.read.parquet("data/content_similarities.parquet")
                recommendations = recommender.get_content_recommendations(selected_user, num_recs)
                recommendations = recommendations.join(
                    spark.read.parquet("data/product_catalog.parquet"),
                    recommendations["productId"] == col("productId")
                ).select("productId", "title", "price")
            
            # Convert to Pandas for display
            recs_pd = recommendations.toPandas()
            
            # Display recommendations
            st.subheader(f"Top {num_recs} Recommendations for User {selected_user}")
            
            for idx, row in recs_pd.iterrows():
                with st.expander(f"{row['title']} - ${row['price']:.2f}"):
                    st.write(f"**Product ID:** {row['productId']}")
                    st.write(f"**Price:** ${row['price']:.2f}")
                    if rec_type == "Hybrid" and 'hybrid_score' in row:
                        st.write(f"**Recommendation Score:** {row['hybrid_score']:.2f}")
        
        except Exception as e:
            st.error(f"Error generating recommendations: {str(e)}")
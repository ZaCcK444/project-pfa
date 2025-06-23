# app.py - Fixed version
import streamlit as st
from pyspark.sql import SparkSession
import os
from pathlib import Path

from hybrid_model import HybridRecommender
# Add src directory to Python path
import sys
sys.path.append(str(Path(__file__).parent / "src"))


@st.cache_resource
def init_spark():
    """Initialize and cache Spark session"""
    return SparkSession.builder \
        .appName("ECommerceRecommender") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .config("spark.sql.execution.pyspark.udf.faulthandler.enabled", "true") \
        .config("spark.python.worker.faulthandler.enabled", "true") \
        .config("spark.sql.shuffle.partitions", "100") \
        .config("spark.default.parallelism", "100") \
        .getOrCreate()

@st.cache_resource
def init_recommender(_spark):
    """Initialize recommender with validation"""
    try:
        return HybridRecommender(_spark)
    except Exception as e:
        st.error(f"Failed to initialize recommender: {str(e)}")
        st.stop()

@st.cache_data
def load_users(_spark):
    """Load distinct user IDs"""
    try:
        reviews = _spark.read.parquet("data/cleaned_reviews.parquet")
        return reviews.select("user_id").distinct().toPandas()["user_id"].tolist()
    except Exception as e:
        st.error(f"Error loading users: {str(e)}")
        return []

def display_recommendations(recommendations):
    """Display recommendations in Streamlit with empty state handling"""
    if recommendations.isEmpty():
        st.warning("No recommendations could be generated for this user.")
        return
    
    recs_pd = recommendations.toPandas()
    
    st.subheader(f"Top {len(recs_pd)} Recommendations")
    cols = st.columns(3)
    
    for idx, row in recs_pd.iterrows():
        with cols[idx % 3]:
            st.markdown(f"**{row['title']}**")
            st.caption(f"${row['price']:.2f}")
            if 'hybrid_score' in row:
                st.progress(min(row['hybrid_score'], 1.0))
                st.caption(f"Score: {row['hybrid_score']:.2f}")
            elif 'als_score' in row:
                st.caption("ALS Recommendation")
            elif 'content_score' in row:
                st.caption("Content-Based Recommendation")

def main():
    st.set_page_config(page_title="Product Recommender", layout="wide")
    st.title("E-commerce Product Recommender")
    
    # Initialize services
    spark = init_spark()
    recommender = init_recommender(spark)
    
    # User selection
    user_list = load_users(spark)
    if not user_list:
        st.error("No users found. Please check your data files.")
        st.stop()
    
    selected_user = st.selectbox("Select a user:", user_list)
    num_recs = st.slider("Number of recommendations:", 5, 20, 10)
    rec_type = st.radio("Recommendation type:", 
                       ["Hybrid", "Collaborative Filtering", "Content-Based"])
    
    # Generate recommendations
    if st.button("Get Recommendations"):
        with st.spinner("Generating recommendations..."):
            try:
                if rec_type == "Hybrid":
                    recs = recommender.hybrid_recommend(selected_user, num_recs)
                elif rec_type == "Collaborative Filtering":
                    recs = recommender.get_als_recommendations(selected_user, num_recs)
                else:
                    recs = recommender.get_content_recommendations(selected_user, num_recs)
                
                display_recommendations(recs)
                
            except Exception as e:
                st.error(f"Recommendation error: {str(e)}")

if __name__ == "__main__":
    main()
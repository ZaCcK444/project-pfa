import streamlit as st
from pyspark.sql import SparkSession
import os
from pathlib import Path
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add src directory to Python path
sys.path.append(str(Path(__file__).parent / "src"))

@st.cache_resource
def init_spark():
    """Initialize Spark with optimized settings"""
    try:
        spark = SparkSession.builder \
            .appName("ECommerceRecommender") \
            .config("spark.driver.memory", "4g") \
            .config("spark.executor.memory", "4g") \
            .config("spark.sql.shuffle.partitions", "200") \
            .config("spark.default.parallelism", "200") \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .getOrCreate()
        logger.info("Spark session initialized successfully")
        return spark
    except Exception as e:
        logger.error(f"Failed to initialize Spark: {str(e)}")
        st.error("Failed to initialize Spark session. Please check the logs.")
        st.stop()

@st.cache_resource
def init_recommender(_spark):
    """Initialize recommender with validation"""
    try:
        from hybrid_model import HybridRecommender
        recommender = HybridRecommender(_spark)
        logger.info("Recommender initialized successfully")
        return recommender
    except Exception as e:
        logger.error(f"Failed to initialize recommender: {str(e)}")
        st.error(f"Failed to initialize recommender: {str(e)}")
        st.stop()

@st.cache_data
def load_users(_spark):
    """Load distinct user IDs with error handling"""
    try:
        reviews = _spark.read.parquet("data/cleaned_reviews.parquet")
        users = reviews.select("user_id").distinct().toPandas()["user_id"].tolist()
        logger.info(f"Loaded {len(users)} users")
        return users
    except Exception as e:
        logger.error(f"Error loading users: {str(e)}")
        st.error(f"Error loading users: {str(e)}")
        return []

def display_recommendations(recommendations):
    """Display recommendations in Streamlit with proper formatting"""
    if recommendations.isEmpty():
        st.warning("No recommendations could be generated.")
        return
    
    recs_pd = recommendations.toPandas()
    
    st.subheader(f"Top {len(recs_pd)} Recommendations")
    cols = st.columns(3)
    
    for idx, row in recs_pd.iterrows():
        with cols[idx % 3]:
            with st.container():
                st.markdown(f"**{row['title']}**")
                st.caption(f"Price: ${row['price']:.2f}")
                if 'hybrid_score' in row:
                    st.caption(f"Score: {row['hybrid_score']:.2f}")
                    st.progress(min(row['hybrid_score'], 1.0))
                elif 'total_score' in row:
                    st.caption(f"Content Score: {row['total_score']:.2f}")
                elif 'als_score' in row:
                    st.caption("Collaborative Filtering Recommendation")

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
                logger.error(f"Recommendation error for user {selected_user}: {str(e)}")

if __name__ == "__main__":
    main()
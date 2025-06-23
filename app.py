import streamlit as st
from pyspark.sql import SparkSession
import os
import sys
from pathlib import Path

# Add src directory to Python path
sys.path.append(str(Path(__file__).parent / "src"))
from hybrid_model import HybridRecommender

@st.cache_resource
def init_spark():
    """Initialize and cache Spark session"""
    return SparkSession.builder \
        .appName("ECommerceRecommender") \
        .config("spark.driver.memory", "2g") \
        .getOrCreate()

@st.cache_resource
def init_recommender(_spark):
    """Initialize with enhanced validation"""
    try:
        recommender = HybridRecommender(_spark)
        
        # Test with a sample user
        test_user = _spark.read.parquet("data/cleaned_reviews.parquet") \
            .select("userId").first()[0]
        
        # Verify recommendations can be generated
        test_recs = recommender.get_als_recommendations(test_user, 1)
        print(f"Test recommendation generated for user {test_user}")
        
        return recommender
        
    except Exception as e:
        st.error(f"""
        Recommender initialization failed!
        Error: {str(e)}
        
        Please verify:
        1. Your models were trained with correct column names
        2. Your data files contain the expected columns
        3. You have proper file permissions
        """)
        st.stop()
        
@st.cache_data
def load_users(_spark):
    """Load and cache distinct user IDs"""
    try:
        reviews = _spark.read.parquet("data/cleaned_reviews.parquet")
        return reviews.select("user_id").distinct().toPandas()["user_id"].tolist()
    except Exception as e:
        st.error(f"Error loading users: {str(e)}")
        return []

def display_recommendations(recommendations):
    """Display recommendations in Streamlit"""
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
                    recs = recs.join(
                        spark.read.parquet("data/product_catalog.parquet"),
                        "product_id"
                    ).select("product_id", "title", "price")
                else:
                    recs = recommender.get_content_recommendations(selected_user, num_recs)
                    recs = recs.join(
                        spark.read.parquet("data/product_catalog.parquet"),
                        "product_id"
                    ).select("product_id", "title", "price")
                
                display_recommendations(recs)
                
            except Exception as e:
                st.error(f"Recommendation error: {str(e)}")

if __name__ == "__main__":
    main()
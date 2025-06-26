import streamlit as st
from src.spark_connector import create_spark_session
from pyspark.sql.functions import col
import os
from pathlib import Path
import sys
import logging
from typing import Optional, List, Tuple
import pandas as pd
from src.utils import ensure_project_structure
from pyspark.sql import SparkSession

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add src directory to Python path
sys.path.append(str(Path(__file__).parent / "src"))
PROJECT_ROOT = ensure_project_structure()

@st.cache_resource
def init_spark() -> SparkSession:
    """Initialize Spark with optimized settings"""
    try:
        # Set environment variables for Spark
        os.environ['SPARK_LOCAL_IP'] = '127.0.0.1'
        os.environ['SPARK_LOCAL_DIRS'] = '/tmp/spark-temp'
        
        # Create Spark session with enhanced configuration
        spark = create_spark_session(
            app_name="ECommerceRecommender",
            max_retries=5,
            retry_delay=15
        )
        
        # Additional configuration specific to this app
        spark.conf.set("spark.driver.memory", "8g")
        spark.conf.set("spark.executor.memory", "8g")
        spark.conf.set("spark.sql.shuffle.partitions", "200")
        spark.conf.set("spark.default.parallelism", "200")
        
        logger.info("Spark session initialized successfully")
        return spark
    except Exception as e:
        logger.error(f"Failed to initialize Spark: {str(e)}", exc_info=True)
        st.error(f"Failed to initialize Spark session: {str(e)}")
        st.stop()

@st.cache_resource
def init_recommender(_spark: SparkSession):
    """Initialize recommender with validation"""
    try:
        from src.hybrid_model import HybridRecommender
        recommender = HybridRecommender(_spark)
        logger.info("Recommender initialized successfully")
        return recommender
    except ImportError as e:
        logger.error(f"Failed to import recommender module: {str(e)}", exc_info=True)
        st.error("Recommender module not found. Please ensure the hybrid_model.py exists in src/")
        st.stop()
    except Exception as e:
        logger.error(f"Failed to initialize recommender: {str(e)}", exc_info=True)
        st.error(f"Failed to initialize recommender: {str(e)}")
        st.stop()

@st.cache_data
def load_users(_spark: SparkSession) -> List[str]:
    """Load distinct user IDs with error handling"""
    try:
        data_path = PROJECT_ROOT / "data/cleaned_reviews.parquet"
        if not data_path.exists():
            raise FileNotFoundError(f"Cleaned reviews data not found at {data_path}")
            
        reviews = _spark.read.parquet(str(data_path))
        
        if "user_id" not in reviews.columns:
            raise ValueError("user_id column not found in reviews data")
            
        users_df = reviews.select("user_id").distinct()
        if users_df.isEmpty():
            raise ValueError("No users found in the dataset")
            
        users = users_df.toPandas()["user_id"].tolist()
        logger.info(f"Loaded {len(users)} users")
        return users
    except Exception as e:
        logger.error(f"Error loading users: {str(e)}", exc_info=True)
        st.error(f"Error loading users: {str(e)}")
        st.stop()

def display_recommendations(recommendations) -> None:
    """Display recommendations in Streamlit with proper formatting"""
    if recommendations is None or (hasattr(recommendations, 'isEmpty') and recommendations.isEmpty()):
        st.warning("No recommendations could be generated.")
        return
    
    try:
        recs_pd = recommendations.limit(20).toPandas()  # Safeguard against large results
        
        if recs_pd.empty:
            st.warning("No recommendations available after processing.")
            return
            
        st.subheader(f"Top {len(recs_pd)} Recommendations")
        
        # Create responsive columns
        cols_per_row = min(3, len(recs_pd))
        cols = st.columns(cols_per_row)
        
        for idx, row in recs_pd.iterrows():
            with cols[idx % cols_per_row]:
                with st.container():
                    st.markdown(f"**{row.get('title', 'Untitled Product')}**")
                    
                    if 'price' in row and pd.notna(row['price']):
                        st.caption(f"Price: ${float(row['price']):.2f}")
                    
                    # Display appropriate score based on recommendation type
                    if 'hybrid_score' in row and pd.notna(row['hybrid_score']):
                        score = float(row['hybrid_score'])
                        st.caption(f"Score: {score:.2f}")
                        st.progress(min(score, 1.0))
                    elif 'total_score' in row and pd.notna(row['total_score']):
                        st.caption(f"Content Score: {float(row['total_score']):.2f}")
                    elif 'als_score' in row and pd.notna(row['als_score']):
                        st.caption("Collaborative Filtering Recommendation")
                    else:
                        st.caption("Recommended for you")
                        
                    st.markdown("---")
    except Exception as e:
        logger.error(f"Error displaying recommendations: {str(e)}", exc_info=True)
        st.error("Failed to display recommendations. Please check the logs.")

def show_welcome_message() -> None:
    """Display welcome message and app information"""
    st.title("📚 E-commerce Product Recommender")
    st.markdown("""
        This recommender system provides personalized product recommendations using:
        - **Collaborative Filtering**: Based on user behavior patterns
        - **Content-Based**: Based on product features and descriptions
        - **Hybrid**: Combining both approaches for better recommendations
    """)
    st.markdown("---")

def get_user_selection(spark: SparkSession) -> Tuple[str, int, str]:
    """Handle user selection UI and validation"""
    user_list = load_users(spark)
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_user = st.selectbox(
            "👤 Select a user:", 
            user_list,
            help="Choose a user ID to generate recommendations for"
        )
    
    with col2:
        num_recs = st.slider(
            "🔢 Number of recommendations:", 
            5, 20, 10,
            help="Adjust how many recommendations to display"
        )
    
    rec_type = st.radio(
        "🎛️ Recommendation type:", 
        ["Hybrid", "Collaborative Filtering", "Content-Based"],
        index=0,
        horizontal=True,
        help="Choose the recommendation algorithm to use"
    )
    
    return selected_user, num_recs, rec_type

def main() -> None:
    # Configure page
    st.set_page_config(
        page_title="Product Recommender", 
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    show_welcome_message()
    
    # Initialize services
    spark = init_spark()
    recommender = init_recommender(spark)
    
    # Get user selection
    selected_user, num_recs, rec_type = get_user_selection(spark)
    
    # Generate recommendations
    if st.button("🚀 Get Recommendations", help="Click to generate recommendations"):
        with st.spinner("✨ Generating recommendations..."):
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
                logger.error(
                    f"Recommendation error for user {selected_user}: {str(e)}", 
                    exc_info=True
                )

if __name__ == "__main__":
    main()
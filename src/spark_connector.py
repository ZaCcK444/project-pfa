import os
import sys
import time
from pyspark.sql import SparkSession
from .utils import get_spark_config  # Relative import

def create_spark_session(app_name="PySparkApp", max_retries=3, retry_delay=10):
    """Create and configure a stable Spark session with robust error handling"""
    for attempt in range(max_retries):
        try:
            # Set environment variables
            os.environ['PYSPARK_PYTHON'] = sys.executable
            os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
            os.environ['SPARK_LOCAL_IP'] = '127.0.0.1'
            os.environ['SPARK_LOCAL_DIRS'] = '/tmp/spark-temp'
            
            # Get optimized configuration
            conf = get_spark_config(app_name)
            
            # Create session
            spark = SparkSession.builder \
                .appName(app_name) \
                .config(conf=conf) \
                .getOrCreate()
            
            # Validate connection with robust test
            test_rdd = spark.sparkContext.parallelize(range(1, 101))
            if test_rdd.sum() == 5050:
                return spark
            else:
                spark.stop()
                raise RuntimeError("Spark connection test failed: Result mismatch")
                
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Spark initialization failed (attempt {attempt+1}/{max_retries}): {str(e)}")
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                continue
            else:
                raise RuntimeError(f"Failed to create Spark session after {max_retries} attempts: {str(e)}")
    
    # Should never reach here
    raise RuntimeError("Unexpected error in Spark session creation")
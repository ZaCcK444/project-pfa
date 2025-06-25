import os
import sys  # Add this import at the top
from pyspark.sql import SparkSession
from pyspark import SparkConf

def create_spark_session(app_name="PySparkApp"):
    """Create and configure a stable Spark session with connection fixes"""
    
    # Set environment variables first
    os.environ['PYSPARK_PYTHON'] = sys.executable
    os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
    
    conf = SparkConf()
    
    # Core configuration
    conf.set("spark.driver.host", "localhost")
    conf.set("spark.driver.bindAddress", "0.0.0.0")
    conf.set("spark.driver.memory", "4g")
    conf.set("spark.executor.memory", "4g")
    
    # Network and connection settings
    conf.set("spark.network.timeout", "600s")
    conf.set("spark.executor.heartbeatInterval", "60s")
    conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
    
    # Python worker settings
    conf.set("spark.python.worker.reuse", "false")
    conf.set("spark.sql.execution.pyspark.udf.faulthandler.enabled", "true")
    
    # Serialization
    conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    
    # Create session
    spark = SparkSession.builder \
        .appName(app_name) \
        .config(conf=conf) \
        .getOrCreate()
    
    # Validate connection
    try:
        spark.sparkContext.parallelize([1, 2, 3]).count()  # Simple test
        return spark
    except Exception as e:
        spark.stop()
        raise RuntimeError(f"Spark connection test failed: {str(e)}")
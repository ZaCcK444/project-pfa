from pyspark.sql import SparkSession
import os

def test_spark():
    spark = None
    try:
        # Create minimal Spark session
        spark = SparkSession.builder \
            .appName("TestSpark") \
            .config("spark.driver.memory", "1g") \
            .config("spark.executor.memory", "1g") \
            .config("spark.sql.execution.arrow.pyspark.enabled", "false") \
            .config("spark.sql.adaptive.enabled", "false") \
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
            .getOrCreate()
        
        # Test basic functionality
        data = [(1, "test"), (2, "data")]
        df = spark.createDataFrame(data, ["id", "value"])
        print("Spark test successful:")
        df.show()
        
        return True
        
    except Exception as e:
        print(f"Spark test failed: {e}")
        return False
    finally:
        if spark:
            spark.stop()

if __name__ == "__main__":
    test_spark()
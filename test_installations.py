import pyspark
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession

# Initialize Spark
spark = SparkSession.builder \
    .appName("InstallationTest") \
    .config("spark.driver.memory", "2g") \
    .getOrCreate()

# Test Spark
df = spark.createDataFrame([(1, "Alice"), (2, "Bob")], ["id", "name"])
df.show()

# Test other libraries
print("NumPy version:", np.__version__)
print("Pandas version:", pd.__version__)

# Clean up
spark.stop()
print("All installations verified successfully!")
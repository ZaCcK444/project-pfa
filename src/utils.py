import os
from pathlib import Path
from pyspark import SparkConf

def ensure_project_structure():
    """Ensure required directories exist"""
    base_dir = Path(__file__).parent.parent
    required_dirs = [
        "data",
        "models",
        "checkpoints"
    ]
    
    for dir_name in required_dirs:
        dir_path = base_dir / dir_name
        dir_path.mkdir(exist_ok=True)
    
    return base_dir

def get_spark_config(app_name="PySparkApp", driver_memory="4g", executor_memory="4g"):
    """Return optimized Spark configuration for socket stability"""
    conf = SparkConf()
    # Network stability configuration
    conf.set("spark.driver.host", "localhost")
    conf.set("spark.driver.bindAddress", "127.0.0.1")
    conf.set("spark.network.timeout", "1000s")
    conf.set("spark.executor.heartbeatInterval", "300s")
    conf.set("spark.worker.timeout", "600")
    
    # Resource allocation
    conf.set("spark.driver.memory", driver_memory)
    conf.set("spark.executor.memory", executor_memory)
    conf.set("spark.sql.shuffle.partitions", "100")
    conf.set("spark.default.parallelism", "100")
    
    # Performance optimization
    conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    conf.set("spark.sql.adaptive.enabled", "true")
    conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
    conf.set("spark.ui.showConsoleProgress", "false")
    
    # Connection stability
    conf.set("spark.port.maxRetries", "100")
    conf.set("spark.rpc.numRetries", "20")
    conf.set("spark.rpc.retry.wait", "60s")
    
    # Python-specific settings
    conf.set("spark.executorEnv.PYTHONHASHSEED", "0")
    conf.set("spark.python.worker.reuse", "false")
    
    return conf
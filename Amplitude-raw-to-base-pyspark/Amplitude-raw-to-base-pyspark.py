import sys
import logging
from datetime import datetime
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql.functions import col, dayofmonth, hour, month, year

def get_logger(job_name):
    logger = logging.getLogger(job_name)
    logger.setLevel(logging.INFO)

    # Direct logs to standard out so CloudWatch picks them up
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)

    # Format: 2026-06-24 11:00:00 - INFO - Your message
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


## @params: [JOB_NAME]
args = getResolvedOptions(sys.argv, ['JOB_NAME', 'source_bucket', 'target_bucket'])

job_name = args["JOB_NAME"]
logger = get_logger(job_name)
logger.info(f"AWS Glue Job {job_name} Initialized successfully.")

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)


today = datetime.now()
year_str = today.strftime("%Y")
month_str = today.strftime("%m")
day_str = today.strftime("%d")

try:
    s3_source_basepath = args['source_bucket']
    s3_source_fullpath = f"{s3_source_basepath}/{year_str}/{month_str}/{day_str}"
    logger.info(f'retrieving files from {s3_source_fullpath}')
    
    s3_target_basepath = args['target_bucket']
    s3_target_fullpath = f"{s3_target_basepath}/"
    
    logger.info('reading json data')
    df_raw = spark.read.format("json").load(s3_source_fullpath)
    
    record_count = df_raw.count()
    logger.info(f'{record_count} events read successfully')
    
    if record_count == 0:
        logger.error('no records found in source path, terminating glue job')
    
    df_base = df_raw.select(
        'event_id',
        'session_id',
        'user_id',
        'device_id',
        'amplitude_id',
        'ip_address',
        'country',
        'region',
        'city',
        'device_family',
        'device_type',
        'event_time',
        'server_received_time'
        )
    
    df_partition = (
        df_base.withColumn("year", year("server_received_time")).
        withColumn("month", month("server_received_time")).
        withColumn("day", dayofmonth("server_received_time")).
        withColumn("hour", hour("server_received_time"))
        )
    
    logger.info(f'writing partitioned files to {s3_target_fullpath}')
    
    (
    df_partition.coalesce(1).write.mode("append")
    .partitionBy("year", "month", "day")
    .option("path", s3_target_fullpath)
    .format("parquet")
    .saveAsTable("amplitude_analytics.tkz_amplitude_base_parquet")
    )
    
    logger.info(f'files written successfully')

except Exception as e:
    # Crucial for debugging: Capture the exact error trace before crashing
    logger.error(f"job failed due to error: {str(e)}", exc_info=True)
    raise e      # Re-raise the error so the Glue service registers the job as FAILED

finally:
    # Always commit to save state/bookmarks
    job.commit()
    logger.info("Glue Job script finished execution.")
    
job.commit()
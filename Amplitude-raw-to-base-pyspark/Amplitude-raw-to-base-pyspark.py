import sys
import logging
from datetime import datetime, timedelta
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql.functions import col, explode, dayofmonth, hour, month, year, lit, input_file_name, current_timestamp, to_json, struct


def flatten_dataframe(df):
    """Dynamically flattens nested structs and explodes arrays in a Spark DataFrame."""
    while True:
        # Identify complex types (Structs and Arrays)
        struct_fields = [
            f.name for f in df.schema.fields if f.dataType.typeName() == "struct"
        ]
        array_fields = [
            f.name for f in df.schema.fields if f.dataType.typeName() == "array"
        ]

        if not struct_fields and not array_fields:
            break  # DataFrame is completely flat

        # 1. Flatten Structs
        if struct_fields:
            for field_name in struct_fields:
                # Expand struct columns into 'structName_fieldName'
                struct_schema = df.schema[field_name].dataType
                expanded_cols = [
                    col(f"{field_name}.{sub_f.name}").alias(
                        f"{field_name}_{sub_f.name}"
                    )
                    for sub_f in struct_schema.fields
                ]
                # Drop original struct and append expanded columns
                df = df.select("*", *expanded_cols).drop(field_name)

        # 2. Explode Arrays (Optional: Handle with care as it multiplies rows)
        if array_fields:
            for field_name in array_fields:
                df = df.withColumn(field_name, explode(col(field_name)))

    return df



def get_logger(job_name):
    logger = logging.getLogger(job_name)
    logger.setLevel(logging.INFO)

    # Direct logs to standard out so CloudWatch picks them up
    handler = logging.StreamHandler(sys.stdout) #sending the logs to standard
    handler.setLevel(logging.INFO)

    # Format: 2026-06-24 11:00:00 - INFO - Your message
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def add_metadata(df, job_run_id):
    "adds columns for date partitioning, and standard metadata to help with auditing"
    return df \
    .withColumn("_ingested_at", current_timestamp()) \
    .withColumn("_source_file", input_file_name()) \
    .withColumn("_pipeline_id", lit(job_run_id)) \
    .withColumn("year", lit(year_str)) \
    .withColumn("month", lit(month_str)) \
    .withColumn("day", lit(day_str))

#gets arguments from system variables (think environment variables)
args = getResolvedOptions(sys.argv, ['JOB_NAME', 'source_bucket', 'target_bucket'])

job_name = args['JOB_NAME']
logger = get_logger(job_name)
logger.info(f"AWS Glue Job {job_name} Initialized successfully.")

#standard launch script for spark in glue
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
spark.conf.set("spark.sql.sources.partitionOverwriteMode", "dynamic") #this is not standard, but used to enable the "overwrite" mode below in the "write" block
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

##-- initialize date variables --##
today = datetime.utcnow()
yesterday = (today - timedelta(days=1))
year_str = yesterday.strftime("%Y")
month_str = yesterday.strftime("%m")
day_str = yesterday.strftime("%d")

##-- grab job run ID from args (we don't need to call it out in the getResolvedOptions function because it exists by default in sys.argv) --##
job_run_id = args['JOB_RUN_ID'] 

amp_prefix = "processed-amplitude-raw"
ipinfo_prefix = "processed-ipinfo-raw"

##-- DROP the glue tables first to address schema changes that might otherwise break them...this could pose a problem down the line --##
spark.sql("DROP TABLE IF EXISTS amplitude_analytics.amplitude_ip_info")
spark.sql("DROP TABLE IF EXISTS amplitude_analytics.amplitude_amplitude_events")

try:
    s3_source_basepath = args['source_bucket']
    s3_source_fullpath_amp = f"{s3_source_basepath}/{amp_prefix}/{year_str}/{month_str}/{day_str}"
    s3_source_fullpath_ip = f"{s3_source_basepath}/{ipinfo_prefix}/{year_str}/{month_str}/{day_str}"
    logger.info(f'retrieving files from {s3_source_fullpath_amp}, {s3_source_fullpath_ip}')
    
    s3_target_basepath = args['target_bucket']
    s3_target_fullpath_amp = f"{s3_target_basepath}/amp"
    s3_target_fullpath_ip = f"{s3_target_basepath}/ip"
    
    logger.info('reading json data')
    df_raw_amp = spark.read.format("json").load(s3_source_fullpath_amp)
    df_raw_ip = spark.read.format("json").option("multiline", "true").load(s3_source_fullpath_ip)
    
    # record_count = df_raw.count()
    # logger.info(f'{record_count} events read successfully')
    
    # if record_count == 0:
    #     logger.error('no records found in source path, terminating glue job')
    
    df_amp = flatten_dataframe(df_raw_amp)
    df_amp = add_metadata(df_amp, job_run_id)
    df_amp_final = df_amp.select("*")
    
    df_ip = add_metadata(df_raw_ip, job_run_id)
    df_ip_final = df_ip.select(
        # Grabs literally every column read from the JSON and turns it into one string
        to_json(struct("*")).alias("network_json"),
        col("_pipeline_id").cast("string"),     # Force the schema type
        col("_ingested_at").cast("timestamp"),
        col("_source_file").cast("string"),
        col("year"),
        col("month"),
        col("day")
    )
    
    
    logger.info(f'writing partitioned files to {s3_target_fullpath_amp}, {s3_target_fullpath_ip}')
    
    (
    df_amp_final.coalesce(1).write.mode("overwrite")
    .partitionBy("year", "month", "day")
    .option("path", s3_target_fullpath_amp)
    .format("parquet")
    .saveAsTable("amplitude_analytics.amplitude_events")
    )
    
    (
    df_ip_final.coalesce(1).write.mode("overwrite")
    .partitionBy("year", "month", "day")
    .option("path", s3_target_fullpath_ip)
    .format("parquet")
    .saveAsTable("amplitude_analytics.amplitude_ip_info")
    )
    
    spark.sql("MSCK REPAIR TABLE amplitude_analytics.amplitude_events")
    spark.sql("MSCK REPAIR TABLE amplitude_analytics.amplitude_ip_info")
    
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
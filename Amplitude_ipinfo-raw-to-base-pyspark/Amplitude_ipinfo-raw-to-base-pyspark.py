import sys
import logging
from datetime import datetime, timedelta
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql.functions import col, explode, dayofmonth, hour, month, year


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
spark.conf.set(
    "spark.sql.sources.partitionOverwriteMode", "dynamic"
)
job = Job(glueContext)
job.init(args['JOB_NAME'], args)


today = datetime.utcnow()
yesterday = (today - timedelta(days=1))
year_str = yesterday.strftime("%Y")
month_str = yesterday.strftime("%m")
day_str = yesterday.strftime("%d")

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
    
    df_base = flatten_dataframe(df_raw)
    
    df_partition = (
        df_base.withColumn("year", year(yesterday)).
        withColumn("month", month(yesterday)).
        withColumn("day", dayofmonth(yesterday))
        )
    
    logger.info(f'writing partitioned files to {s3_target_fullpath}')
    
    (
    df_partition.coalesce(1).write.mode("overwrite")
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
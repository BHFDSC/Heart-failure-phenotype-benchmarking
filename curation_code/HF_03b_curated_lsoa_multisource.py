# Databricks notebook source
# MAGIC %md
# MAGIC # HF_03b_curated_lsoa_multisource
# MAGIC
# MAGIC This notebook reformats the curated multisource LSOA table, which is to be used in the lsoa notebook.

# COMMAND ----------

# MAGIC %md # Setup

# COMMAND ----------

spark.sql('CLEAR CACHE')
spark.conf.set('spark.sql.legacy.allowCreatingManagedTableUsingNonemptyLocation', 'true')

# COMMAND ----------

import pyspark.sql.functions as f
import pyspark.sql.types as t
from pyspark.sql import Window

from functools import reduce

# import databricks.koalas as ks
import pandas as pd
import pyspark.pandas as ps
import numpy as np

import re
import io
import datetime

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import dates as mdates
import seaborn as sns

print("Matplotlib version: ", matplotlib.__version__)
print("Seaborn version: ", sns.__version__)
_datetimenow = datetime.datetime.now() # .strftime("%Y%m%d")
print(f"_datetimenow:  {_datetimenow}")

# COMMAND ----------

# MAGIC %run "Workspace/Shared/SHDS/common/functions"

# COMMAND ----------

# MAGIC %md # Parameters

# COMMAND ----------

# MAGIC %run "./HF_02_main_parameters"

# COMMAND ----------

# MAGIC %md # Data

# COMMAND ----------

lsoa_multisource = spark.table(hds_curated_assets_lsoa_multisource)

# COMMAND ----------

# MAGIC %md # Prepare

# COMMAND ----------

# Preparing lsoa_multisource
# Reformatting record_date column in the lsoa_multisource table to ensure that all dates are converted from string to date format (consistent)
cur_lsoa_multisource = (
    lsoa_multisource
    .filter(f.col('lsoa').isNotNull()) # drop null lsoa
    .filter(f.col('record_date').isNotNull()) # drop null record date 
    .withColumn('record_date', f.when(f.col('record_date').rlike("^\d{4}-\d{2}-\d{2}$"),  # Matches 'yyyy-MM-dd'
        f.to_date(f.col('record_date'), "yyyy-MM-dd"))
                .when(f.col('record_date').rlike("^\d{8}$"),  # Matches 'yyyyMMdd'
        f.to_date(f.col('record_date'), "yyyyMMdd")))
    # indicator
    .withColumn('ind_record_date', 
                f.when(f.col('record_date').rlike("^\d{4}-\d{2}-\d{2}$"), f.lit(1))
                .otherwise(f.lit(0)))
    .withColumnRenamed('person_id', 'PERSON_ID')
    .withColumnRenamed('lsoa', 'lsoa_multisource')
    .withColumnRenamed('record_date', 'lsoa_multisource_record_date')          
    .withColumnRenamed('data_source', 'lsoa_multisource_data_source')
    .withColumn('in_england', f.when(f.substring(f.col('lsoa_multisource'), 0, 1) == 'E', '1')
                .otherwise('0'))
    )

# COMMAND ----------

# Check 
tmpt = tab(cur_lsoa_multisource, 'ind_record_date')

ind_record_date_min = cur_lsoa_multisource.groupBy().agg(f.min(f.col('ind_record_date'))).collect()[0][0] 
assert ind_record_date_min != 0, 'The minimum value of ind_record_date is 0'

# COMMAND ----------

# Drop columns
cur_lsoa_multisource = (
    cur_lsoa_multisource
    .drop('ind_record_date', 'archived_on')
)

# COMMAND ----------

# MAGIC %md
# MAGIC # Save

# COMMAND ----------

save_table(cur_lsoa_multisource, f'{proj}_cur_lsoa_multisource', save_previous=False)
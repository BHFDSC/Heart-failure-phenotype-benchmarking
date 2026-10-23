# Databricks notebook source
# MAGIC %md # HF_05_demographics
# MAGIC  
# MAGIC **Description** This notebook creates the demographics table, which includes key patient characteristics of the cohort, such as date of birth, sex, ethnicity, date of death.
# MAGIC  
# MAGIC **Authors** Jadene Lewis, Tom Bolton (Health Data Science Team, BHF Data Science Centre)
# MAGIC
# MAGIC **Reviewers**  
# MAGIC
# MAGIC **Acknowledgements** 
# MAGIC
# MAGIC **Data Input**
# MAGIC - **`path_tmp_hf_cohort`, `hds_curated_assets_demographic`**
# MAGIC
# MAGIC **Data Output**
# MAGIC - **`hf_benchmarking_jl_tmp_demographics`** : Curated demographics table (before inclusion/exclusion criteria have been applied)

# COMMAND ----------

# MAGIC %md
# MAGIC # Setup

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



# COMMAND ----------

# MAGIC %md
# MAGIC #0. Parameters

# COMMAND ----------

# MAGIC %run "./HF_02_main_parameters"

# COMMAND ----------

# MAGIC %md
# MAGIC # 1. Data

# COMMAND ----------

# Load data
demographics = spark.table(hds_curated_assets_demographic)  


# COMMAND ----------

display(demographics)

# COMMAND ----------

# MAGIC %md 
# MAGIC # 2. Prepare


# COMMAND ----------

# Preparing demographics table: renaming columns and recoding sex_code column
demographics = (
    demographics
    .withColumn('sex_code', f.when(f.col('sex_code') == 'F', '1')
                .when(f.col('sex_code') == 'M', '2')
                .when(f.col('sex_code') == '9', f.lit(None))
                .otherwise(f.col('sex_code')))
    .select(f.col('person_id').alias('PERSON_ID'), 
            f.col('date_of_birth').alias('DOB'),
            f.col('sex_code').alias('sex'),
            f.col('ethnicity_5_group').alias('eth5'),
            f.col('date_of_death').alias('DOD'),
            'in_gdppr')
    .where(f.col('PERSON_ID').isNotNull()) # filtering out null person_id
)

# COMMAND ----------

# Check
tmpt = tab(demographics, 'sex')

# COMMAND ----------

# MAGIC %md 
# MAGIC # 3. Check

# COMMAND ----------

# Demographics table check
count_var(demographics, 'PERSON_ID') 
tab(demographics, 'in_gdppr')

# COMMAND ----------

# MAGIC %md
# MAGIC # 4. Demographics


# COMMAND ----------

# MAGIC %md # 5. Check


# COMMAND ----------

# MAGIC %md
# MAGIC # 6. Save

# COMMAND ----------

save_table(df=demographics, out_name=f'{proj}_tmp_demographics', save_previous=False)

# COMMAND ----------

display(demographics) 

# COMMAND ----------


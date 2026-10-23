# Databricks notebook source
# MAGIC %md # HF_D06a-lsoa
# MAGIC  
# MAGIC **Description** This notebook creates the LSOA table, which includes LSOA, region and deprivation information
# MAGIC  
# MAGIC **Authors** Jadene Lewis, Tom Bolton (Health Data Science Team, BHF Data Science Centre)
# MAGIC
# MAGIC **Reviewers**  
# MAGIC
# MAGIC **Acknowledgements** 
# MAGIC
# MAGIC **Data Input**
# MAGIC
# MAGIC **Data Output**
# MAGIC - **`hf_benchmarking_jl_tmp_lsoa`** : Curated LSOA table (before inclusion/exclusion criteria have been applied)

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

# MAGIC %md
# MAGIC #0. Parameters

# COMMAND ----------

# MAGIC %run "./HF_02_main_parameters"

# COMMAND ----------

# MAGIC %run "./HF_02a_custom_functions"

# COMMAND ----------

# MAGIC %md
# MAGIC # 1. Data

# COMMAND ----------

# Load data 
lsoa_multisource = spark.table(hds_curated_assets_lsoa_multisource)

# Load lookups
lsoa_2001 = spark.table(f'{dsa}.hds_cur_lsoa_2001_imd_lookup') # do not need these lookups as the July version of lsoa_multisource has no 2001 only LSOAs 
lsoa_2011 = spark.table(f'{dsa}.hds_cur_lsoa_2011_imd_lookup')

# COMMAND ----------

# MAGIC %md # 2. Check

# COMMAND ----------

# Exploring null values of record_date before preparing
display(lsoa_multisource.filter(f.col('record_date').isNull()).groupBy().agg(f.count(f.lit(1))))
display(lsoa_multisource.filter(f.col('lsoa').isNull()).groupBy().agg(f.count(f.lit(1)))) 

# COMMAND ----------

# MAGIC %md 
# MAGIC # 3. Prepare

# COMMAND ----------

# # Preparing different views of tmp_hf_cohort
# tmp_hf_cohort_person_id = (
#     tmp_hf_cohort
#     .select('person_id') 
#     .dropDuplicates()
# )

# tmp_hf_cohort_person_id_date = (
#     tmp_hf_cohort
#     .select('person_id', 'study_period_start')
#     .dropDuplicates()
# )


# COMMAND ----------

# Preparing lsoa_multisource
# Reformatting record_date column in the lsoa_multisource table to ensure that all dates are converted from string to date format (consistent)
lsoa_multisource = (
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
                # .when(f.col('record_date').rlike("^\d{8}$"), f.lit(2))
                .otherwise(f.lit(0)))
    #.withColumnRenamed('person_id', 'PERSON_ID')
    #.withColumnRenamed('lsoa', 'lsoa_multisource')
    .withColumnRenamed('record_date', 'lsoa_multisource_record_date')          
    .withColumnRenamed('data_source', 'lsoa_multisource_data_source')
    .drop('archived_on')
    )

# COMMAND ----------

display(lsoa_multisource)

# COMMAND ----------

# Check 
tmpt = tab(lsoa_multisource, 'ind_record_date')

ind_record_date_min = lsoa_multisource.groupBy().agg(f.min(f.col('ind_record_date'))).collect()[0][0] 
assert ind_record_date_min != 0, 'The minimum value of ind_record_date is 0'

# drop('ind_record_date')

# COMMAND ----------

lsoa_multisource = (
    lsoa_multisource
    .drop('ind_record_date')
)

# COMMAND ----------

# Select and rename columns of LSOA lookups
lsoa_2001 = (
    lsoa_2001
    .select(f.col('LSOA_2001').alias('lsoa'))
    .withColumn('ind_lsoa_2001', f.lit(1))
)

lsoa_2011 = (
    lsoa_2011
    .select(f.col('LSOA_2011').alias('lsoa'))
    .withColumn('ind_lsoa_2011', f.lit(1))
)

# COMMAND ----------

# MAGIC %md 
# MAGIC # 4. Check

# COMMAND ----------

# display(tmp_hf_cohort.orderBy(f.col("person_id").desc()))

# COMMAND ----------

# # Check
# count_varlist(tmp_hf_cohort, ['person_id', 'study_period_start'])


# COMMAND ----------

# check for distinct LSOA in 2001 and 2011 lookups
count_var(lsoa_2001, 'lsoa')
count_var(lsoa_2011, 'lsoa')

# COMMAND ----------

# MAGIC %md
# MAGIC # 5. LSOA

# COMMAND ----------

# Using multisource LSOA as main LSOA
lsoa_multisource_2 = (
    lsoa_multisource
    .withColumn('study_start_date', f.lit(study_start_date))
    .withColumn('date_diff', f.abs(f.datediff(f.col('lsoa_multisource_record_date'), f.col('study_start_date'))))
)

# COMMAND ----------

# Adding a 'lsoa_tie_flag' column- flags where there is a tie by date_diff (two distinct LSOA values for same date_diff). Arbritraily chooses one of the flags
tmp_ties = (
    lsoa_multisource_2
    .groupBy('person_id', 'study_start_date', 'date_diff')
    .agg(f.countDistinct(f.col('lsoa')).alias('n_distinct_lsoa_multisource'))
)

lsoa_multisource_3 = (
    lsoa_multisource_2
    .join(tmp_ties, ['person_id', 'study_start_date', 'date_diff'], how = 'left')
    .withColumn('lsoa_tie_flag', f.when(f.col('n_distinct_lsoa_multisource') > 1, '1')
                .when(f.col('n_distinct_lsoa_multisource') == 1, '0')
                .otherwise(None)
))

# COMMAND ----------

# check
tmpt = tab(lsoa_multisource_3, 'n_distinct_lsoa_multisource', 'lsoa_tie_flag')

# COMMAND ----------

# Select the LSOA closest to study start date
window_spec = Window.partitionBy('person_id', 'study_start_date').orderBy('date_diff', 'lsoa') # random selection where there are multiple LSOA with same date_diff. order by both columns to keep stable

lsoa_multisource_4 = (
    lsoa_multisource_3
    .withColumn('row_num', f.row_number().over(window_spec))
    .filter(f.col('row_num') == 1)
    .drop('row_num')
)

# COMMAND ----------

display(lsoa_multisource_4) 


# COMMAND ----------

# Add indicator columns
lsoa_multisource_5 = (
    lsoa_multisource_4
    .withColumn('ind_lsoa_null', f.when(f.col('lsoa').isNull(), f.lit(1)).otherwise(0))
    .withColumn('lsoa_1', f.substring('lsoa', 1, 1))
    .withColumn('ind_lsoa_england', f.when(f.col('lsoa_1') == f.lit('E'), f.lit(1)).otherwise(0)) 
) 

# COMMAND ----------

# DBTITLE 1,Join to LSOA01 and LSOA11 lookups
# Join to LSOA01 and LSOA11 lookups
lsoa_multisource_6 = merge(lsoa_multisource_5, lsoa_2001, ['lsoa'], validate='m:1', keep_results=['both', 'left_only'], indicator=0); print()
lsoa_multisource_7 = merge(lsoa_multisource_6, lsoa_2011, ['lsoa'], validate='m:1', keep_results=['both', 'left_only'], indicator=0); print()

# COMMAND ----------

# DBTITLE 1,Flagging Non-England, LSOA11, LSOA01
lsoa_multisource_8 = (
    lsoa_multisource_7
    .withColumn('lsoa_desc', 
                f.when((f.col('ind_lsoa_2001') == 1) & (f.col('ind_lsoa_2011') == 1), f.lit('2001 and 2011'))
                .when((f.col('ind_lsoa_2001') == 1) & (f.col('ind_lsoa_2011') == 0), f.lit('2001 only'))
                .when((f.col('ind_lsoa_2011') == 1) & (f.col('ind_lsoa_2001') == 0), f.lit('2011 only'))
                .when(f.substring(f.col('lsoa'), 0, 1) != 'E', 'Non-England')
                .otherwise(f.lit('99999999999')) # where lsoa is null
    )
    .withColumn('in_england', f.when(f.substring(f.col('lsoa'), 0, 1) == 'E', '1')
                .otherwise('0'))
)

# COMMAND ----------

display(lsoa_multisource_8)

# COMMAND ----------

# Check
tmpt = tab(lsoa_multisource_8, 'lsoa_desc') 

# COMMAND ----------

# Rename columns
lsoa_multisource_8 = (
    lsoa_multisource_8
    .select('person_id', 'study_start_date', 
'lsoa',
'lsoa_multisource_record_date',
'lsoa_multisource_data_source',
'date_diff',
'lsoa_tie_flag',
'ind_lsoa_null',
'lsoa_1',
'ind_lsoa_england',
'ind_lsoa_2001',
'ind_lsoa_2011',
'in_england')
)

# COMMAND ----------

# DBTITLE 1,Check
# Exploring lsoa column in more detail
tmpt = tab(lsoa_multisource_8, 'lsoa')

# COMMAND ----------

# create a temp table 
save_table(df=lsoa_multisource_8, out_name=f'{proj}_tmp_lsoa', save_previous=False)

# COMMAND ----------

# Reload table
lsoa_multisource_8 = spark.table(f'.{proj}_tmp_lsoa')

# COMMAND ----------

# Check
count_varlist(lsoa_multisource_8, ['person_id', 'study_start_date', 'lsoa'])

# COMMAND ----------

# Drop unnecessary columns
lsoa_multisource_9 = (
    lsoa_multisource_8
    .drop('ind_lsoa_null', 'lsoa_1', 'ind_lsoa_england', 'ind_lsoa_2001', 'ind_lsoa_2011')
)

# COMMAND ----------

# Check: distinct values of lsoa_multisource_data_source
tmpt = tab(lsoa_multisource_9, 'lsoa_multisource_data_source')

# COMMAND ----------

# MAGIC %md
# MAGIC # 6. Check

# COMMAND ----------

_checks = (
    lsoa_multisource_9
    .withColumn('lsoa_is_null', f.when(f.col('lsoa').isNull(), 1)
                .otherwise(0))
    .withColumn('date_diff', f.abs(f.datediff(f.col('lsoa_multisource_record_date'), f.col('study_start_date'))))
    .withColumn('date_diff_grouped', f.when(f.col('date_diff') == 0, '0')
                .when(f.col('date_diff') <= 7, '<= 1week')
                .when((f.col('date_diff') > 7) & (f.col('date_diff') <= 30.4167), '1week-1month')
                .when((f.col('date_diff') > 31) & (f.col('date_diff') <= 182.5), '1month-6months')
                .when((f.col('date_diff') > 182.5) & (f.col('date_diff') <= 365), '6months-1year')
                .when(f.col('date_diff') > 365, '>1year')
                .otherwise(None))
)    

# COMMAND ----------

chk2 = tab(_checks, 'lsoa_is_null')
chk6 = tabstat(_checks, 'date_diff')
chk7 = tab(_checks, 'date_diff_grouped')

# COMMAND ----------

count_var(lsoa_multisource_9, 'person_id')

# COMMAND ----------

# MAGIC %md
# MAGIC # 7. Save

# COMMAND ----------

save_table(df=lsoa_multisource_9, out_name=f'{proj}_tmp_lsoa', save_previous=False)


# COMMAND ----------

display(spark.table(f'{path_tmp_lsoa}'))

# COMMAND ----------

count_var(lsoa_multisource_9, 'person_id') 
count_varlist(lsoa_multisource_9, ['person_id', 'study_start_date']) 
count_varlist(lsoa_multisource_9, ['person_id', 'study_start_date', 'lsoa'])

# COMMAND ----------


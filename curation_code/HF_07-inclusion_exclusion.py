# Databricks notebook source
# MAGIC %md # HF_07-inclusion_exclusion
# MAGIC  
# MAGIC **Description** This notebook applies the inclusion/exclusion criteria.
# MAGIC  
# MAGIC **Authors** Jadene Lewis, Tom Bolton (Health Data Science Team, BHF Data Science Centre)
# MAGIC  
# MAGIC **Reviewers**
# MAGIC
# MAGIC **Acknowledgements** CCU004_03 (Tom Bolton, Fionna Chalmers, Anna Stevenson)
# MAGIC
# MAGIC **Data Output**
# MAGIC - **`hf_benchmarking_jl_base_cohort`**: Base cohort table (after inclusion/exclusion criteria have been applied)
# MAGIC - **`hf_benchmarking_jl_out_lsoa`**: Final LSOA table - to be used for extracting IMD details later (after inclusion/exclusion criteria have been applied)
# MAGIC - **`hf_benchmarking_jl_tmp_inc_exc_flow`** : flowchart displaying total n of cohort after each inclusion/exclusion rule is applied


# COMMAND ----------

# MAGIC %md
# MAGIC # 0. Setup

# COMMAND ----------

spark.sql('CLEAR CACHE')
spark.conf.set('spark.sql.legacy.allowCreatingManagedTableUsingNonemptyLocation', 'true')

# COMMAND ----------

# DBTITLE 1,Libraries
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

# DBTITLE 1,Common functions
# MAGIC %run "/Workspace/Shared/SHDS/common/functions"

# COMMAND ----------

# MAGIC %md
# MAGIC # 1. Parameters

# COMMAND ----------

# MAGIC %run "./HF_02_main_parameters"

# COMMAND ----------

# MAGIC %run "./HF_02a_custom_functions" 

# COMMAND ----------

# MAGIC %md
# MAGIC # 2. Data

# COMMAND ----------

patient_id_table = spark.table(patient_id_table) # table with list of patient IDs
tmp_demographics = spark.table(path_tmp_demographics)   # .drop('_merge') # drop in previous notebook
tmp_lsoa = spark.table(path_tmp_lsoa)
qa_table = spark.table(path_tmp_quality_assurance)

# COMMAND ----------

# MAGIC %md
# MAGIC # 3. Prepare

# COMMAND ----------

# # reduce
tmp_demographics = (
  tmp_demographics
  .withColumn('first_char', f.substring(f.col('PERSON_ID'), 1, 1))
  .withColumn('in_demographics', f.lit(1))
  .withColumn('flag_deaths', f.when(f.col('DOD').isNotNull(), 1)
              .otherwise(0)) # in_deaths -> flag_deaths
)

# COMMAND ----------

# Join tmp_demographics with tmp_lsoa to get 'DATE' and 'in_england' columns (needed for inc/exc criteria)
tmp_demographics = (
    tmp_demographics
    .join(tmp_lsoa.select('PERSON_ID', 'in_england'), ['PERSON_ID'], how='left')
    .withColumn('study_start_date', f.lit(study_start_date))
    .withColumn('study_end_date', f.lit(study_end_date))
)

# COMMAND ----------

# Selecting relevant columns
patient_id_table = (
    patient_id_table
    .select('pseudo_id', 'valid_nhs_number', 'first_char')
    .withColumn('in_patient_id', f.lit(1)) # in_patient_id_valid  -> in_patient_id
    .withColumnRenamed('pseudo_id', 'PERSON_ID') # PERSON_ID
)

# check
count_var(patient_id_table, 'PERSON_ID')

# COMMAND ----------

display(patient_id_table)

# COMMAND ----------

# Merging tables
# join on person_id and first character columns (in patient_id_table) (to speed up the join)
_merged = (
    tmp_demographics    
    .join(patient_id_table, on=['first_char', 'PERSON_ID'], how='left')
)


# COMMAND ----------

# Save as temp table
save_table(df=_merged, out_name=f'{proj}_tmp_inc_exc_merged', save_previous=False)

# COMMAND ----------

# Load temp table
_merged = spark.table(f'.{proj}_tmp_inc_exc_merged')

# COMMAND ----------

# check
display(_merged)

# COMMAND ----------

# check
tmpt = tab(_merged, 'in_demographics'); print()
tmpt = tab(_merged, 'in_patient_id'); print()
tmpt = tab(_merged, 'flag_deaths'); print()

# check only true false in valid nhs number flag

in_patient_id_min = _merged.select(f.min(f.col('in_patient_id'))).collect()[0][0]
assert in_patient_id_min == 1, "The minimum value of in_patient_id is not 1"

# COMMAND ----------

# MAGIC %md 
# MAGIC # 4. Inclusion / exclusion

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.1. Exclude patients that died before study start date

# COMMAND ----------

tmp0 = _merged
tmpc = count_var(tmp0, 'PERSON_ID', ret=1, df_desc='original', indx=0); print()

# COMMAND ----------

display(tmp0)

# COMMAND ----------

# filter out patients who died before study start date
tmp0 = (
  tmp0
  .dropDuplicates()
  .withColumn('DOD_flag', 
              f.when(f.col('DOD').isNull(), 1)
              .when(f.col('DOD') < f.col('study_start_date'), 2) # <= to <
              .when(f.col('DOD') >= f.col('study_start_date'), 3) # > to >= 
              .otherwise(999999)
             )
)

# check 
tmpt = tab(tmp0, 'DOD_flag'); print()
# assert tmp0.where(~(f.col('DOD_flag').isin([1,2,3]))).count() == 0
tmpt = tabstat(tmp0, 'DOD', byvar='DOD_flag', date=1); print()

window_spec = Window.partitionBy('PERSON_ID')
tmp0 = (
    tmp0
    .withColumn('flag_DOD_before_DATE', f.when(f.col('DOD_flag') == 2, f.lit(1)).otherwise(0))
    .withColumn('flag_DOD_before_DATE_max', f.max(f.col('flag_DOD_before_DATE')).over(window_spec))
)

tmpt = tab(tmp0, 'DOD_flag', 'flag_DOD_before_DATE_max')

# filter out and tidy
tmp1 = (
    tmp0
    .filter(f.col('flag_DOD_before_DATE_max') == 0)
)

# check
tmpt = count_var(tmp1, 'PERSON_ID', ret=1, df_desc='post exclusion of patients who died before the study start date', indx=1); print() # rerun
tmpc = tmpc.unionByName(tmpt)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.2. Exclude patients without a valid NHS Number

# COMMAND ----------

tmp2 = (
    tmp1
    .withColumn('valid_nhs_number_flag', 
                f.when(f.col('valid_nhs_number') == 'True', 1)
                .when(f.col('valid_nhs_number') == 'False', 0)
                .otherwise(999999))
)

# check 
tmpt = tab(tmp2, 'valid_nhs_number_flag'); print()
# assert tmp2.where(~(f.col('valid_nhs_number_flag').isin([0, 1]))).count() == 0
# tmpt = tabstat(tmp2, 'DOD', byvar='valid_nhs_number_flag', date=1); print()

# filter out and tidy
tmp3 = (
  tmp2
  .where((f.col('valid_nhs_number_flag').isin([1])))
  .drop('valid_nhs_number_flag', 'valid_nhs_number', '_merge', 'in_demographics', 'flag_deaths', 'first_char', 'in_patient_id', 'DOD_flag', 'flag_DOD_before_DATE', 'flag_DOD_before_DATE_max')
)

# check
tmpt = count_var(tmp3, 'PERSON_ID', ret=1, df_desc='post exclusion of patients who do not have a valid NHS number', indx=2); print()
tmpc = tmpc.unionByName(tmpt)

# COMMAND ----------

display(tmp3)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.3. Exclude patients with a non-England LSOA

# COMMAND ----------

# check 
tmpt = tab(tmp3, 'in_england'); print()

# filter out and tidy
tmp4 = (
  tmp3
  .where((f.col('in_england').isin([1])))
  .drop('valid_nhs_number_flag', 'valid_nhs_number', '_merge', 'in_demographics', 'flag_deaths', 'first_char', 'in_patient_id', 'DOD_flag', 'flag_DOD_before_DATE', 'flag_DOD_before_DATE_max', 'in_england')
)

# check
tmpt = count_var(tmp4, 'PERSON_ID', ret=1, df_desc='post exclusion of patients who do not have an England LSOA', indx=3); print()
tmpc = tmpc.unionByName(tmpt)

# COMMAND ----------

display(tmp4)

# COMMAND ----------

# MAGIC %md ## 4.4. Exclude patients with HF records before the study start date. 

# COMMAND ----------

# Create flag to identify records within study period
tmp4a = (
    tmp4
    .withColumn('after_study_start', f.when(f.col('DATE') >= f.lit(study_start_date), 1)
                .when(f.col('DATE') < f.lit(study_start_date), 0)
                .otherwise(999999))
)

# check 
tmpt1 = tab(tmp4a, 'after_study_start'); print()

# filter out and tidy
tmp4b = (
  tmp4a
  .where((f.col('after_study_start').isin([1])))
  .drop('after_study_start')
)

# check
tmpt = count_var(tmp4b, 'PERSON_ID', ret=1, df_desc='post exclusion of patients who do not have a HF record after the study start date', indx=4); print()
tmpc = tmpc.unionByName(tmpt)

# COMMAND ----------

# MAGIC %md ## 4.5. Exclude patients with HF records after study end date

# COMMAND ----------

# Create flag to identify HF records within study period
tmp4c = (
    tmp4b
    .withColumn('before_study_end', f.when(f.col('DATE') <= f.lit(study_end_date), 1)
                .when(f.col('DATE') > f.lit(study_end_date), 0)
                .otherwise(999999))
)

# check 
tmpt1 = tab(tmp4c, 'before_study_end'); print()

# filter out and tidy
tmp5 = (
  tmp4c
  .where((f.col('before_study_end').isin([1])))
  .drop('before_study_end')
)

# check
tmpt = count_var(tmp5, 'PERSON_ID', ret=1, df_desc='post exclusion of patients who do not have a HF record before the study end date', indx=5); print()
tmpc = tmpc.unionByName(tmpt)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.6. Exclude patients failing QA rules

# COMMAND ----------

# Join with qa table
qa_merged = (
  merge(tmp4, qa_table, ['PERSON_ID'], keep_results=['both', 'right_only'], indicator=1) # double check if there are any 'right_only'
)

# COMMAND ----------

# tabulate the numbers of each rule that is 'failed'
tab(qa_merged, '_rule_concat') 

# COMMAND ----------

# check 
tmpt = tab(qa_merged, '_rule_concat'); print()

# Filter out patients that have not met quality assurance criteria
tmp5 = (
  qa_merged
  .where(~(f.col('_rule_concat').like("%1%")))
  .drop('_rule_1', '_rule_2', '_rule_3', "_rule_4", '_rule_5', '_rule_6', '_rule_7', '_rule_8', '_rule_9', '_rule_10', '_rule_total', '_merge')
)

# check
tmpt = count_var(tmp5, 'PERSON_ID', ret=1, df_desc='post exclusion of patients who failed QA rules', indx=4); print()
tmpc = tmpc.unionByName(tmpt)

# COMMAND ----------

count_var(qa_merged, 'PERSON_ID') 
count_var(tmp5, 'PERSON_ID')

# COMMAND ----------

# # save qa_merged
# save_table(df=qa_merged, out_name=f'{proj}_tmp_qa_merged', save_previous=False)

# COMMAND ----------

# MAGIC %md ## 4.7. Exclude patients not in GDPPR ('in_gdppr' flag)
# MAGIC

# COMMAND ----------

# Check for number of patients not in GDPPR
tab(tmp5, 'in_gdppr') 
# NB: Include the 'in_gdppr' column in the out_demographics. 

# COMMAND ----------

# Check for number of patients not in GDPPR
tmpt = tab(tmp5, 'in_gdppr'); print()

# Filter out patients that have not met quality assurance criteria
tmp6 = (
  tmp5
  .where(f.col('in_gdppr') == 1)
)

# check
tmpt = count_var(tmp6, 'PERSON_ID', ret=1, df_desc='post exclusion of patients who do not have a record in GDPPR', indx=5); print()
tmpc = tmpc.unionByName(tmpt)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.8. Exclude patients aged < 18 (at study start date)

# COMMAND ----------

# Create age column
tmp6 = (
    tmp6
    .withColumn('study_start_age', f.floor(f.datediff(f.col("study_start_date"), f.col("DOB"))/365.25))
    .withColumn('under_18', f.when(f.col('study_start_age') < 18, f.lit(1))
                .otherwise(0))  
)

# Check for number of patients < 18 at study start date
tmpt = tab(tmp6, 'under_18'); print()

# Filter out patients that have not met quality assurance criteria
tmp7 = (
  tmp6
  .where(f.col('study_start_age') >= 18)
)

# check
tmpt = count_var(tmp7, 'PERSON_ID', ret=1, df_desc='post exclusion of patients who are aged < 18 at the time of the study start date', indx=6); print()
tmpc = tmpc.unionByName(tmpt)

# COMMAND ----------

# MAGIC %md
# MAGIC # 5. Flow diagram

# COMMAND ----------

# check flow table
tmpp = (
  tmpc
  .orderBy('indx')
  .select('indx', 'df_desc', 'n', 'n_id', 'n_id_distinct')
  .withColumnRenamed('df_desc', 'stage')
  .toPandas())
tmpp['n'] = tmpp['n'].astype(int)
tmpp['n_id'] = tmpp['n_id'].astype(int)
tmpp['n_id_distinct'] = tmpp['n_id_distinct'].astype(int)
tmpp['n_diff'] = (tmpp['n'] - tmpp['n'].shift(1)).fillna(0).astype(int)
tmpp['n_id_diff'] = (tmpp['n_id'] - tmpp['n_id'].shift(1)).fillna(0).astype(int)
tmpp['n_id_distinct_diff'] = (tmpp['n_id_distinct'] - tmpp['n_id_distinct'].shift(1)).fillna(0).astype(int)
for v in [col for col in tmpp.columns if re.search("^n.*", col)]:
  tmpp.loc[:, v] = tmpp[v].map('{:,.0f}'.format)
for v in [col for col in tmpp.columns if re.search(".*_diff$", col)]:  
  tmpp.loc[tmpp['stage'] == 'original', v] = ''
# tmpp = tmpp.drop('indx', axis=1)
print(tmpp.to_string()); print()

# COMMAND ----------

# MAGIC %md
# MAGIC # Save

# COMMAND ----------

# DBTITLE 1,Base cohort
# Saving final base cohort table
save_table(df=tmp7, out_name=f'{proj}_base_cohort', save_previous=False)

# COMMAND ----------

# DBTITLE 1,LSOA table
# Save LSOA table (to be used later on to get IMD details for benchmarking)
tmp7_reload = spark.table(f'{dsa}.{proj}_base_cohort')

# COMMAND ----------

# Join to tmp7_reload to restrict tmp_lsoa to base cohort (after inc/exc criteria have been applied)
tmp7_lsoa = (
    tmp7_reload
    .join(tmp_lsoa, 'person_id', how='inner')
    .select('PERSON_ID', 'lsoa')
)

# Check
count_var(tmp7_reload, 'PERSON_ID')
count_varlist(tmp7_lsoa, ['PERSON_ID', 'lsoa'])

# COMMAND ----------

save_table(tmp7_lsoa, f'{proj}_out_lsoa', save_previous=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Flow table

# COMMAND ----------

save_table(df=tmpc, out_name=f'{proj}_tmp_inc_exc_flow', save_previous=False)

# COMMAND ----------

display(spark.table(f'{dsa}.{proj}_tmp_inc_exc_flow').orderBy('indx'))

# COMMAND ----------


# Databricks notebook source
# MAGIC %md
# MAGIC # Benchmarking analysis

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
# MAGIC # 0. Parameters

# COMMAND ----------

# MAGIC %run "../HF_02_main_parameters"

# COMMAND ----------

# MAGIC %md
# MAGIC # 1. Data

# COMMAND ----------

hf_cohort_ever_without_logic = spark.table(f'{dsa}.{proj}_out_hf_ever_cases_phenos_without_logic')
hf_cohort_study_period_without_logic = spark.table(f'{dsa}.{proj}_out_hf_study_period_demographics_phenos_without_logic')
hf_cohort_ever_w_logic = spark.table(f'{dsa}.{proj}_out_hf_ever_cases_phenos_with_logic')
hf_cohort_study_period_w_logic = spark.table(f'{dsa}.{proj}_out_hf_study_period_demographics_phenos_with_logic')
# hf_cohort_study_period_demo_without_logic = spark.table(f'{dsa}.{proj}_out_hf_study_period_demographics_without_logic')

out_lsoa = spark.table(f'{dsa}.{proj}_out_lsoa')
imd_lookup = spark.table(path_ref_imd)
region_lookup = spark.table(path_ref_region)

# COMMAND ----------

display(imd_lookup.limit(10))

# COMMAND ----------

display(imd_lookup.groupBy('IMD_2019_QUINTILES').count())
display(region_lookup.groupBy('region_name').count())

# COMMAND ----------

# DBTITLE 1,Join phenos w and w/out logic
hf_cohort_ever = (
    hf_cohort_ever_w_logic
    .unionByName(hf_cohort_ever_without_logic)
)

hf_cohort_study_period = (
    hf_cohort_study_period_w_logic
    .join(hf_cohort_study_period_without_logic, ['person_id', 'DOB', 'study_start_age', 'sex', 'eth5', 'DOD'], how='full')
)

# COMMAND ----------

hf_cohort_ever.columns

# COMMAND ----------

hf_cohort_study_period.columns

# COMMAND ----------

# MAGIC %md
# MAGIC # 2. Prepare

# COMMAND ----------

# Create an age group column
hf_cohort_ever = (
    hf_cohort_ever
    .withColumn('study_start_age_grp', f.when(f.col('study_start_age').between(18, 59), '18-59')
                .when(f.col('study_start_age').between(60,69), '60-69')
                .when(f.col("study_start_age").between(70,79), '70-79')
                .when(f.col('study_start_age').between(80,89), '80-89')
                .when(f.col('study_start_age') >= 90, '90+'))
    .withColumn('eth5', f.when(f.col('eth5').isin(['White']), 'White')
                .when(f.col('eth5').isin(['Asian or Asian British']), 'Asian')
                .when(f.col('eth5').isin(['Black, Black British, Caribbean or African']), 'Black')
                .when(f.col('eth5').isin(['Mixed or multiple ethnic groups']), 'Mixed')
                .when(f.col('eth5').isin(['Other ethnic group']), 'Other')
                .when(f.col('eth5').isNull(), 'Unknown')
                .otherwise(None))  
)

display(hf_cohort_ever.limit(10))

# COMMAND ----------

# Join to lsoa table
hf_cohort_ever_2 = (
    hf_cohort_ever
    .join(out_lsoa, 'person_id', how='left')
)

# COMMAND ----------

hf_cohort_ever_3 = (
    hf_cohort_ever_2
    .filter(f.col("event_date_curated").isNotNull())
)

# COMMAND ----------

save_table(hf_cohort_ever_3, f'{proj}_hf_cohort_ever_3', save_previous=False)

# COMMAND ----------

# reload
hf_cohort_ever_3 = spark.table(f'{dsa}.{proj}_hf_cohort_ever_3')

# COMMAND ----------

# check
tab(hf_cohort_ever_3, 'name')

# COMMAND ----------

# MAGIC %md
# MAGIC # 3. Explore

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ever cases/events (no date restrictions applied)

# COMMAND ----------

# DBTITLE 1,Total number of ever event dates
hf_cohort_ever_count = (
    hf_cohort_ever_3
    .groupBy('name')
    .agg(f.count("event_date_curated"))
)

display(hf_cohort_ever_count)

# COMMAND ----------

# DBTITLE 1,Total number of patients
total_patient_count = (
    hf_cohort_ever_3
    .groupBy("name")
    .agg(f.countDistinct('person_id'))
)

display(total_patient_count)

# COMMAND ----------

# DBTITLE 1,Median age
# Median and IQR age
median_age = (
    hf_cohort_ever_3
    .groupBy('name')
    .agg(f.median(f.col('study_start_age')))
)

display(median_age)

# COMMAND ----------

# DBTITLE 1,IQR age
iqr_age = (
    hf_cohort_ever_3
    .groupby("name")
    .agg(
        f.expr("percentile(study_start_age, array(0.25))")[0].alias("Q1"),
        f.expr("percentile(study_start_age, array(0.75))")[0].alias("Q3")
        )
    .withColumn("IQR", f.col("Q3") - f.col("Q1"))
)

display(iqr_age)

# COMMAND ----------

# DBTITLE 1,Numbers (& percentage) of each age group
# Numbers (& percentage) of each age group
w = Window.partitionBy('name')
percent_by_age_grp_ever = (
    hf_cohort_ever_3
    .withColumn('study_start_age_grp', f.when(f.col('study_start_age').between(18, 59), '18-59')
                .when(f.col('study_start_age').between(60,69), '60-69')
                .when(f.col("study_start_age").between(70,79), '70-79')
                .when(f.col('study_start_age').between(80,89), '80-89')
                .when(f.col('study_start_age') >= 90, '90+')
                .otherwise(None))
    .groupBy('name', 'study_start_age_grp')
    .agg(f.count("*").alias('count'))
    .withColumn('group_total', f.sum(f.col('count')).over(w))
    .withColumn('percentage', f.col('count')*100/f.col('group_total'))
    .withColumn('category_type', f.lit('age'))
    .withColumnRenamed('study_start_age_grp', 'category')
)
display(percent_by_age_grp_ever)

# COMMAND ----------

# DBTITLE 1,Sex
# Sex
w = Window.partitionBy('name')

percent_by_sex_ever = (
    hf_cohort_ever_3
    .groupBy('name', 'sex')
    .agg(f.count("*").alias('count'))
    .withColumn("group_total", f.sum(f.col('count')).over(w))
    .withColumn('percentage', f.col('count')*100/f.col('group_total'))
    .withColumn('category_type', f.lit('sex'))
    .withColumnRenamed('sex', 'category')
)

display(percent_by_sex_ever)

## plot proportions instead of absolute numbers (and ethnicity)

# COMMAND ----------

# DBTITLE 1,Ethnicity
# Ethnicity
# Reformat ethnicity columns
tab(hf_cohort_ever_3, 'name', 'eth5')

display(hf_cohort_ever_3.groupBy('name', 'eth5').count())

# COMMAND ----------

w = Window.partitionBy('name')

percent_by_eth_ever = (
    hf_cohort_ever_3
    .groupBy('name', 'eth5')
    .agg(f.count("*").alias('count'))
    .withColumn("group_total", f.sum(f.col('count')).over(w))
    .withColumn('percentage', f.col('count')*100/f.col('group_total'))
    .withColumn('category_type', f.lit('ethnicity'))
    .withColumnRenamed('eth5', 'category')
)

display(percent_by_eth_ever.limit(10))

## plot proportions instead of absolute numbers (and ethnicity)

# COMMAND ----------

# DBTITLE 1,IMD
# IMD
hf_cohort_ever_imd = (
    hf_cohort_ever_3
    .join(imd_lookup.withColumnRenamed('LSOA_2011', 'lsoa'), 'lsoa', how='left')
)

tab(hf_cohort_ever_imd, 'name', 'IMD_2019_QUINTILES')

display(hf_cohort_ever_imd.groupBy('name', 'IMD_2019_QUINTILES').count())

# COMMAND ----------

w = Window.partitionBy('name')

percent_by_imd_ever = (
    hf_cohort_ever_imd
    .groupBy('name', 'IMD_2019_QUINTILES')
    .agg(f.count("*").alias('count'))
    .withColumn("group_total", f.sum(f.col('count')).over(w))
    .withColumn('percentage', f.col('count')*100/f.col('group_total'))
    .withColumn('category_type', f.lit('IMD'))
    .withColumnRenamed('IMD_2019_QUINTILES', 'category')
)

display(percent_by_imd_ever)

## plot proportions instead of absolute numbers (and ethnicity)

# COMMAND ----------

# DBTITLE 1,Region
# Region
hf_cohort_ever_region = (
    hf_cohort_ever_3
    .join(region_lookup.withColumnRenamed('lsoa_code', 'lsoa'), 'lsoa', how='left')
)

tab(hf_cohort_ever_region, 'name', 'region_name')

display(hf_cohort_ever_region.groupBy('name', 'region_name').count())

# COMMAND ----------

w = Window.partitionBy('name')

percent_by_region_ever = (
    hf_cohort_ever_region
    .withColumn('region_name',f.col('region_name'))
    .groupBy('name', 'region_name')
    .agg(f.count("*").alias('count'))
    .withColumn("group_total", f.sum(f.col('count')).over(w))
    .withColumn('percentage', f.col('count')*100/f.col('group_total'))
    .withColumn('category_type', f.lit('region'))
    .withColumnRenamed('region_name', 'category')
)

display(percent_by_region_ever.limit(10))

## plot proportions instead of absolute numbers (and ethnicity)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Union all

# COMMAND ----------

hf_cohort_ever_unioned = (
    percent_by_sex_ever
    .withColumn('category', f.col('category').cast('string'))
    .unionByName(percent_by_eth_ever.withColumn('category', f.col('category').cast('string')))
    .unionByName(percent_by_imd_ever.withColumn('category', f.col('category').cast('string')))
    .unionByName(percent_by_region_ever.withColumn('category', f.col('category').cast('string')))
    .unionByName(percent_by_age_grp_ever.withColumn('category', f.col('category').cast('string')))
)

display(hf_cohort_ever_unioned.limit(10))

# COMMAND ----------

save_table(hf_cohort_ever_unioned, out_name=f'{proj}_hf_cohort_ever_unioned', save_previous=False)

# COMMAND ----------

# reload
hf_cohort_ever_unioned = spark.table(f'{dsa}.{proj}_hf_cohort_ever_unioned')

# COMMAND ----------

# Round to nearest five (or 0 if < 10)
hf_cohort_ever_unioned_rounded = (
    hf_cohort_ever_unioned
    .withColumn('count_rounded', f.when(f.col("count") < 10, f.lit(0))  # rule 1: <10 → 0
         .otherwise(
             f.round(f.col("count") / 5) * 5  # rule 2: round to nearest 5
         )
    )
    .withColumn('group_total_rounded', f.when(f.col("group_total") < 10, f.lit(0))  # rule 1: <10 → 0
         .otherwise(
             f.round(f.col("group_total") / 5) * 5  # rule 2: round to nearest 5
         )
    )
    .withColumn('percentage_rounded', (f.col('count_rounded')/f.col('group_total_rounded'))*100)
    .drop('percentage') # remove the unrounded percentage column generated from before
)

display(hf_cohort_ever_unioned_rounded)

# COMMAND ----------

save_table(hf_cohort_ever_unioned_rounded, f'{proj}_hf_cohort_ever_unioned_rounded', save_previous=True)

# COMMAND ----------

# to output
x = (
    hf_cohort_ever_unioned_rounded
    .select('name', 'category', 'category_type', 'count_rounded', 'group_total_rounded', 'percentage_rounded')
)

display(x)

# COMMAND ----------

# MAGIC %md
# MAGIC ## First events within study period

# COMMAND ----------

hf_cohort_study_period.columns

# COMMAND ----------

# Reshape to long
hf_cohort_study_period_long = (
    reshape_wide_to_long_multi(hf_cohort_study_period, i = ['person_id', 'study_start_age', 'eth5', 'sex'], j = 'name', stubnames=['pheno_']) # stubnames = ['pheno']
    .withColumnRenamed('pheno_', 'first_event_date')
    .withColumn('name', f.concat(f.lit('pheno_'), f.col('name')))
)

display(hf_cohort_study_period_long.limit(10))

# COMMAND ----------

hf_cohort_study_period_long_2 = (
    hf_cohort_study_period_long
    .filter(f.col('first_event_date').isNotNull())
)

display(hf_cohort_study_period_long_2.limit(10))

# COMMAND ----------

# Check min and max dates
display(hf_cohort_study_period_long_2.select(f.min(f.col("first_event_date")), f.max(f.col('first_event_date'))))

# COMMAND ----------

# DBTITLE 1,Total number of first events within study period
first_event_within_study_period_date_count = (
    hf_cohort_study_period_long_2
    .groupBy('name')
    .agg(f.count(f.col('first_event_date')))
)

display(first_event_within_study_period_date_count)

# COMMAND ----------

# DBTITLE 1,Total number of patients
total_patient_count = (
    hf_cohort_study_period_long_2
    .groupBy("name")
    .agg(f.countDistinct('person_id'))
)

display(total_patient_count)

# COMMAND ----------

# DBTITLE 1,Median age
median_age_2 = (
    hf_cohort_study_period_long_2
    .groupBy('name')
    .agg(f.median(f.col('study_start_age')))
)

# iqr_age_2 = (
#     hf_cohort_study_period
#     .toPandas()
#     .groupby('name')
#     .quantile(0.25).alias('q1')
#     .quantile(0.75).alias('q3')
#     .withColumn('IQR', f.col('q3') - f.col('q1'))
# )

display(median_age_2)
# display(iqr_age_2)

# COMMAND ----------

# DBTITLE 1,IQR (age)
iqr_age_2 = (
    hf_cohort_study_period_long_2
    .groupby("name")
    .agg(
        f.expr("percentile(study_start_age, array(0.25))")[0].alias("Q1"),
        f.expr("percentile(study_start_age, array(0.75))")[0].alias("Q3")
        )
    .withColumn("IQR", f.col("Q3") - f.col("Q1"))
)

display(iqr_age_2)

# COMMAND ----------

# DBTITLE 1,Numbers of each age group
age_grp_count = tab(hf_cohort_study_period_long_2.withColumn('study_start_age_grp', f.when(f.col('study_start_age').between(18, 59), '18-59')
                .when(f.col('study_start_age').between(60,69), '60-69')
                .when(f.col("study_start_age").between(70,79), '70-79')
                .when(f.col('study_start_age').between(80,89), '80-89')
                .when(f.col('study_start_age') >= 90, '90+')), 'name', 'study_start_age_grp')

# COMMAND ----------

# Numbers (& percentage) of each age group
w = Window.partitionBy('name')
percent_by_age_grp_study_period = (
    hf_cohort_study_period_long_2
    .withColumn('study_start_age_grp', f.when(f.col('study_start_age').between(18, 59), '18-59')
                .when(f.col('study_start_age').between(60,69), '60-69')
                .when(f.col("study_start_age").between(70,79), '70-79')
                .when(f.col('study_start_age').between(80,89), '80-89')
                .when(f.col('study_start_age') >= 90, '90+'))
    .groupBy('name', 'study_start_age_grp')
    .agg(f.count("*").alias('count'))
    .withColumn('group_total', f.sum(f.col('count')).over(w))
    .withColumn('percentage', f.col('count')*100/f.col('group_total'))
    .withColumn('category_type', f.lit('age'))
    .withColumnRenamed('study_start_age_grp', 'category')
)
display(percent_by_age_grp_study_period)

# COMMAND ----------

# DBTITLE 1,Sex
w = Window.partitionBy('name')

percent_by_sex_study_period = (
    hf_cohort_study_period_long_2
    .groupBy('name', 'sex')
    .agg(f.count("*").alias('count'))
    .withColumn("group_total", f.sum(f.col('count')).over(w))
    .withColumn('percentage', f.col('count')*100/f.col('group_total'))
    .withColumn('category_type', f.lit('sex'))
    .withColumnRenamed('sex', 'category')
)

display(percent_by_sex_study_period)


# COMMAND ----------

# DBTITLE 1,Ethnicity
# Reformat ethnicity columns
ethnicity_count_2 = (
    hf_cohort_study_period_long_2
    .withColumnRenamed('eth5', 'eth5')
    .withColumn('eth5', f.when(f.col('eth5').isin(['White']), 'White')
                .when(f.col('eth5').isin(['Asian or Asian British']), 'Asian')
                .when(f.col('eth5').isin(['Black, Black British, Caribbean or African']), 'Black')
                .when(f.col('eth5').isin(['Mixed or multiple ethnic groups']), 'Mixed')
                .when(f.col('eth5').isin(['Other ethnic group']), 'Other')
                .when(f.col('eth5').isNull(), 'Unknown')
                .otherwise('999999'))  
)

tab(ethnicity_count_2, 'name', 'eth5')

# COMMAND ----------

w = Window.partitionBy('name')
percent_by_eth_study_period = (
    hf_cohort_study_period_long_2
    .withColumn('study_start_age_grp', f.when(f.col('study_start_age').between(18, 59), '18-59')
                .when(f.col('study_start_age').between(60,69), '60-69')
                .when(f.col("study_start_age").between(70,79), '70-79')
                .when(f.col('study_start_age').between(80,89), '80-89')
                .when(f.col('study_start_age') >= 90, '90+'))
    .groupBy('name', 'eth5')
    .agg(f.count("*").alias('count'))
    .withColumn('group_total', f.sum(f.col('count')).over(w))
    .withColumn('percentage', f.col('count')*100/f.col('group_total'))
    .withColumn('category_type', f.lit('ethnicity'))
    .withColumnRenamed('eth5', 'category')
)
display(percent_by_eth_study_period)

# COMMAND ----------

display(imd_lookup)

# COMMAND ----------

# DBTITLE 1,IMD
hf_cohort_study_period_lsoa = (
    hf_cohort_study_period_long_2
    .join(out_lsoa, 'person_id', how='left')
    .join(imd_lookup.withColumnRenamed('LSOA_2011', 'lsoa'), 'lsoa', how='left')
)

# COMMAND ----------

# Numbers of each IMD quintile and decile, by phenotype ID
# tab(hf_cohort_study_period_lsoa, 'name', 'IMD_2019_DECILES')
tab(hf_cohort_study_period_lsoa, 'name', 'IMD_2019_QUINTILES')

# COMMAND ----------

w = Window.partitionBy('name')

percent_by_imd_study_period = (
    hf_cohort_study_period_lsoa
    .groupBy('name', 'IMD_2019_QUINTILES')
    .agg(f.count("*").alias('count'))
    .withColumn("group_total", f.sum(f.col('count')).over(w))
    .withColumn('percentage', f.col('count')*100/f.col('group_total'))
    .withColumn('category_type', f.lit('IMD'))
    .withColumnRenamed('IMD_2019_QUINTILES', 'category')
)

display(percent_by_imd_study_period)

## plot proportions instead of absolute numbers (and ethnicity)

# COMMAND ----------

display(region_lookup)

# COMMAND ----------

# DBTITLE 1,Region
hf_cohort_study_period_region = (
    hf_cohort_study_period_lsoa
    .join(region_lookup.withColumnRenamed('lsoa_code', 'lsoa'), 'lsoa', how='left')
)

# COMMAND ----------

w = Window.partitionBy('name')

percent_by_region_study_period = (
    hf_cohort_study_period_region
    .groupBy('name', 'region_name')
    .agg(f.count("*").alias('count'))
    .withColumn("group_total", f.sum(f.col('count')).over(w))
    .withColumn('percentage', f.col('count')*100/f.col('group_total'))
    .withColumn('category_type', f.lit('region'))
    .withColumnRenamed('region_name', 'category')
)

display(percent_by_region_study_period.limit(10))

## plot proportions instead of absolute numbers (and ethnicity)

# COMMAND ----------

# DBTITLE 1,Union all
hf_cohort_study_period_unioned = (
    percent_by_sex_study_period
    .withColumn('category', f.col('category').cast('string'))
    .unionByName(percent_by_eth_study_period.withColumn('category', f.col('category').cast('string')))
    .unionByName(percent_by_imd_study_period.withColumn('category', f.col('category').cast('string')))
    .unionByName(percent_by_region_study_period.withColumn('category', f.col('category').cast('string')))
    .unionByName(percent_by_age_grp_study_period.withColumn('category', f.col('category').cast('string')))
)

display(hf_cohort_study_period_unioned.limit(10))

# COMMAND ----------

save_table(hf_cohort_study_period_unioned, f'{proj}_hf_cohort_study_period_unioned', save_previous=False)

# COMMAND ----------

# Reload
hf_cohort_study_period_unioned = spark.table(f'{dsa}.{proj}_hf_cohort_study_period_unioned')

# COMMAND ----------

# Round to nearest five (or 0 if < 10)
hf_cohort_study_period_unioned_rounded = (
    hf_cohort_study_period_unioned
    .withColumn('count', f.col('count'))
    .withColumn('count_rounded', f.when(f.col("count") < 10, f.lit(0))  # rule 1: <10 → 0
         .otherwise(
             f.round(f.col("count") / 5) * 5  # rule 2: round to nearest 5
         )
    )
    .withColumn('group_total_rounded', f.when(f.col("group_total") < 10, f.lit(0))  # rule 1: <10 → 0
         .otherwise(
             f.round(f.col("group_total") / 5) * 5  # rule 2: round to nearest 5
         )
    )
    .withColumn('percentage_rounded', (f.col('count_rounded')/f.col('group_total_rounded'))*100)
    .drop('percentage') # remove the unrounded percentage column generated from before
)

display(hf_cohort_study_period_unioned_rounded)

# COMMAND ----------

save_table(hf_cohort_study_period_unioned_rounded, f'{proj}_hf_cohort_study_period_unioned_rounded', save_previous=True)

# COMMAND ----------

# to output
y = (
    hf_cohort_study_period_unioned_rounded
    .select('name', 'category', 'category_type', 'count_rounded', 'group_total_rounded', 'percentage_rounded')
)
display(y)

# COMMAND ----------

# MAGIC %md
# MAGIC # Computing metrics to compare each phenotype 

# COMMAND ----------

# Reshape from long to wide
hf_cohort_ever_4 = (
    hf_cohort_ever_3
    .withColumn('value', f.when(f.col('event_date_curated').isNotNull(), 1)
                .otherwise(0))
    )

display(hf_cohort_ever_4)

# COMMAND ----------

from pyspark.sql import functions as F

mapping = {
    "pheno_01": "pheno_01_IMR_w_logic",
    "pheno_02": "pheno_02_IIR_w_logic",
    "pheno_05": "pheno_05_IS",
    "pheno_09": "pheno_09_S",
    "pheno_11": "pheno_11_IS",
    "pheno_12": "pheno_12_IS",
    "pheno_13": "pheno_13_I_w_logic",
    "pheno_14": "pheno_14_BNF",
    "pheno_15": "pheno_15_IS",
    "pheno_16": "pheno_16_I_w_logic",
    "pheno_17": "pheno_17_IS",
    "pheno_18": "pheno_18_IIRROU_w_logic",
    "pheno_20": "pheno_20_S",
    "pheno_21": "pheno_21_S",
    "pheno_24": "pheno_24_S",
    "pheno_25": "pheno_25_S",
    "pheno_27": "pheno_27_S",
    "pheno_28": "pheno_28_IS",
    "pheno_29": "pheno_29_IS",
    "pheno_30": "pheno_30_IS",
    "pheno_31": "pheno_31_IS",
    "pheno_33": "pheno_33_I_w_logic",
    "pheno_34": "pheno_34_S",
    "pheno_36": "pheno_36_S",
    "pheno_37": "pheno_37_I_w_logic", # "pheno_37_I_with_logic",
    "pheno_38": "pheno_38_II"
}

mapping_expr = F.create_map([F.lit(x) for kv in mapping.items() for x in kv])

hf_cohort_ever_4 = hf_cohort_ever_4.withColumn("name", mapping_expr[F.col("name")])


# COMMAND ----------

display(hf_cohort_ever_4)

# COMMAND ----------

ordered_labels = [
    # ICD-10 and 'SNOMED CT'
    "pheno_05_IS", "pheno_11_IS", "pheno_12_IS", "pheno_15_IS", "pheno_17_IS",
    "pheno_28_IS", "pheno_29_IS", "pheno_30_IS", "pheno_31_IS",

    # ICD-9/ICD-10 with logic
    "pheno_13_I_w_logic", "pheno_16_I_w_logic", "pheno_33_I_w_logic", "pheno_37_I_w_logic", 

    # ICD-9/ICD-10 without logic
    "pheno_38_II",

    # ICD-9; ICD-10; Read codes with logic
    "pheno_02_IIR_w_logic",

    # ICD-9; ICD-10; read codes (V2 and V3); UK Biobank with logic
    "pheno_18_IIRROU_w_logic",

    # ICD-10; Med codes; Read codes with logic
    "pheno_01_IMR_w_logic",

    # SNOMED
    "pheno_09_S", "pheno_20_S", "pheno_21_S", "pheno_24_S", "pheno_25_S",
    "pheno_27_S", "pheno_34_S", "pheno_36_S",

    # BNF
    "pheno_14_BNF"
]


# COMMAND ----------

# ------------------------------------------------------------
# ONE-CELL PIPELINE: Complete Pairwise Similarity + Heatmap
# ------------------------------------------------------------

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pyspark.sql import functions as F

def phenotype_pairwise_similarity(df_long,
                                  patient_col="person_id",
                                  pheno_col="name",
                                  value_col="Value"):

    pos = (
        df_long
        .filter(F.col(value_col) == 1)
        .select(patient_col, pheno_col)
        .distinct()
    )

    inter = (
        pos.alias("a")
        .join(pos.alias("b"), F.col("a."+patient_col) == F.col("b."+patient_col))
        .select(
            F.col("a."+pheno_col).alias("ph1"),
            F.col("b."+pheno_col).alias("ph2")
        )
        .groupBy("ph1", "ph2")
        .agg(F.count("*").alias("intersection"))
    )

    sizes = (
        pos.groupBy(pheno_col)
        .agg(F.countDistinct(patient_col).alias("set_size"))
    )

    sim = (
        inter
        .join(
            sizes.withColumnRenamed(pheno_col, "ph1")
                 .withColumnRenamed("set_size", "size_ph1"),
            on="ph1"
        )
        .join(
            sizes.withColumnRenamed(pheno_col, "ph2")
                 .withColumnRenamed("set_size", "size_ph2"),
            on="ph2"
        )
        .withColumn("union", F.col("size_ph1") + F.col("size_ph2") - F.col("intersection"))
        .withColumn("Jaccard", F.col("intersection") / F.col("union"))
    )

    return sim

# 1. Compute similarity
sim_df = phenotype_pairwise_similarity(hf_cohort_ever_4)
sim_pd = sim_df.toPandas()

# 2. Extract full phenotype list
phenos = sorted(set(sim_pd["ph1"]).union(sim_pd["ph2"]))

# 3. Build full Cartesian grid
full_grid = pd.MultiIndex.from_product([phenos, phenos], names=["ph1", "ph2"])
full_df = pd.DataFrame(index=full_grid).reset_index()

# 4. Merge similarity results onto full grid
merged = full_df.merge(sim_pd[["ph1", "ph2", "Jaccard"]], on=["ph1", "ph2"], how="left")

# 5. Fill missing similarities with 0
merged["Jaccard"] = merged["Jaccard"].fillna(0)

# 6. Pivot into a complete square matrix
jaccard_matrix = merged.pivot(index="ph1", columns="ph2", values="Jaccard")

jaccard_matrix = jaccard_matrix.astype(float)

jaccard_matrix = jaccard_matrix.loc[ordered_labels, ordered_labels]

# Get same color scheme as phmeatmap in R
pheatmap_colors = sns.color_palette(["blue", "white", "red"], as_cmap=True)

# 7. Plot heatmap at final print size so 6 pt text stays 6 pt on the page
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],  # MIE recommends Arial/Helvetica
    "font.size": 6,
})

fig, ax = plt.subplots(figsize=(4.5, 4.2))   # inches; fits the 12.4 cm type area
sns.heatmap(
    jaccard_matrix,
    cmap="RdYlBu_r",
    vmin=0,
    vmax=1,
    linewidths=0.3,
    linecolor="white",
    square=True,
    xticklabels=True,      # force every label; seaborn otherwise drops some at small sizes
    yticklabels=True,
    cbar_kws={"shrink": 0.7},
    ax=ax,
)
ax.set_ylim(jaccard_matrix.shape[0], 0)
ax.set_xlabel("")          # axis titles dropped; the caption explains the axes
ax.set_ylabel("")
ax.tick_params(axis="both", labelsize=6, length=2, width=0.4)

cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=6, length=2, width=0.4)
cbar.set_label("Jaccard similarity", size=6)

fig.savefig("figure1_heatmap.png", dpi=600, bbox_inches="tight")
fig.savefig("figure1_heatmap.pdf", bbox_inches="tight")   # vector copy, if needed
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # Upset plot

# COMMAND ----------

# Install upsetplot package (via pip install command)

# COMMAND ----------

display(hf_cohort_ever_3)

# COMMAND ----------

hf_cohort_ever_3_wide = (
    hf_cohort_ever_3
    # only select distinct/unique phenos
    .drop('pheno_28', 'pheno_15', 'pheno_17', 'pheno_30', 'pheno_33')
    .withColumn('value', f.when(f.col("event_date_curated").isNull(), f.lit(0))
                .otherwise(f.lit(1)))
    #.withColumn('value', f.col('value').cast('boolean'))
    .groupBy('PERSON_ID')
    .pivot("name")
    .agg(f.first("value"))
    .fillna(0)
    #.fillna(False)
)

display(hf_cohort_ever_3_wide)

# COMMAND ----------

# Create a concat column to indicate the combos
pheno_cols = [c for c in hf_cohort_ever_3_wide.columns if c.startswith("pheno_")]

exprs = [
    f.concat(
        f.lit(f"{c}_"),      # literal prefix "pheno_02_"
        f.col(c).cast("string")  # boolean → "true"/"false"
    )
    for c in pheno_cols
]

hf_cohort_ever_3_wide_2 = (
    hf_cohort_ever_3_wide
    .withColumn('combination', f.concat_ws("|", *exprs))
)

display(hf_cohort_ever_3_wide_2.groupBy('combination').agg(f.countDistinct('person_id')))

# COMMAND ----------

# Rounded
display(hf_cohort_ever_3_wide_2.groupBy('combination').agg(f.countDistinct('person_id')).withColumn('count_rounded', f.when(f.col("count(DISTINCT person_id)") < 10, f.lit(0))  # rule 1: <10 → 0
         .otherwise(
             f.round(f.col("count(DISTINCT person_id)") / 5) * 5  # rule 2: round to nearest 5
         )
    )
        .drop('count(DISTINCT person_id)'))

# COMMAND ----------

# Do the groupBy() to show a summary of the number of combinations --> then can export this outside of the SDE. to be more manageables (this would be underlying data for the upset plot). NB: Make sure to round (but this shouldn't affect the numbers much).

# Generate some dummy data (eg: 2 phenotypes and 10 patients)

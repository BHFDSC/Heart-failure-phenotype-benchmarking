# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # HF_07b_add_hf_prior_and_outcome
# MAGIC
# MAGIC
# MAGIC **Description** Extract CKD before and after index date (study start date). 
# MAGIC
# MAGIC **Authors** Jadene Lewis
# MAGIC
# MAGIC **Reviewers** 
# MAGIC
# MAGIC **Acknowledgements** 
# MAGIC
# MAGIC **Notes**
# MAGIC
# MAGIC **Data Output**
# MAGIC - **``** :  

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # 1. Setup and parameters

# COMMAND ----------

import pyspark.sql.functions as f
import pyspark.sql.types as t
from pyspark.sql import Window

#from functools import reduce

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

# MAGIC %run "/Workspace/Shared/SHDS/common/functions"

# COMMAND ----------

# MAGIC %run "./HF_02_main_parameters"

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # 2. Load the base cohort and extracted HF events

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.1 Base cohort

# COMMAND ----------

df_base = spark.table(f'{dsa}.{proj}_base_cohort')

# COMMAND ----------

count_var(df_base, "PERSON_ID")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.2. Heart failure phenotypes

# COMMAND ----------

# DBTITLE 1,GDPPR
gdppr_hf= spark.table(f'{dsa}.{proj}_gdppr_all_hf_events')
display(gdppr_hf.groupBy("name", "phenotype_source").agg(f.count("*")))

# COMMAND ----------

# DBTITLE 1,HES APC
hes_apc_hf = spark.table(f'{dsa}.{proj}_hes_apc_diag_all_hf_events')
display(hes_apc_hf.groupBy("name", "phenotype_source").agg(f.count("*")))

# COMMAND ----------

# DBTITLE 1,HES APC Procedures
# hes_apc_proc_hf = spark.table(f'{dsa}.{proj}_procedure_phenotypes_hes_apc_all_events')
# display(hes_apc_proc_hf.groupBy("phenotype_name", "phenotype_source").agg(f.count("*")))

# COMMAND ----------

# DBTITLE 1,Primary care meds
pmeds_bnf_hf = spark.table(f'{dsa}.{proj}_pmeds_all_hf_events')
display(pmeds_bnf_hf.groupBy("name", "phenotype_source").agg(f.count("*")))

# COMMAND ----------

# MAGIC %md
# MAGIC # 3. Extract the 'ever' cases for HF

# COMMAND ----------

# Join all events across all datasets
gdppr_hf_sel = gdppr_hf.select(["person_id", "name", "event_date_curated", "phenotype_source"])
hes_apc_hf_sel = hes_apc_hf.select(["person_id", "name", "event_date_curated", "phenotype_source"])
pmeds_hf_sel = pmeds_bnf_hf.select(["person_id", "name", "event_date_curated", "phenotype_source"])

union_hf_1 = gdppr_hf_sel.union(hes_apc_hf_sel).union(pmeds_hf_sel)

# COMMAND ----------

hf_ever = union_hf_1.select(["person_id", "name", "event_date_curated", "phenotype_source"])

# COMMAND ----------

count_var(hf_ever, 'person_id')

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # 3. Find the first event date of HF (heart failure) per patient per phenotype
# MAGIC
# MAGIC

# COMMAND ----------

gdppr_hf_sel = gdppr_hf.select(["person_id", "name", "event_date_curated", "phenotype_source"])
hes_apc_hf_sel = hes_apc_hf.select(["person_id", "name", "event_date_curated", "phenotype_source"])
pmeds_hf_sel = pmeds_bnf_hf.select(["person_id", "name", "event_date_curated", "phenotype_source"])

# COMMAND ----------

# We need first event dates across all datasets
union_hf = gdppr_hf_sel.union(hes_apc_hf_sel).union(pmeds_hf_sel)

# COMMAND ----------

# Add study start and end date
union_hf = (union_hf
             .withColumn("study_start_date", f.to_date(f.lit(study_start_date)))
             .withColumn("study_end_date", f.to_date(f.lit(study_end_date)))
             )
display(union_hf.limit(10))

# COMMAND ----------

# Check
display(union_hf.groupBy("name").agg(f.count("*")))

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Create Flags:
# MAGIC
# MAGIC - Before study start date : Not inclusive of study start date
# MAGIC - >= study start date and <= study end date
# MAGIC - > study end date (drop later)
# MAGIC
# MAGIC - We have cleaned event dates < dob before.

# COMMAND ----------

# DBTITLE 1,Create flags
# # create flags
# union_hf_b =(union_hf
#             .withColumn ("phenotype_ID", 
#                          f.when(f.col("event_date_curated")<f.col("study_start_date"), 
#                                 f.concat(f.col("phenotype_ID"), f.lit("_previous")))
#                          .when((f.col("event_date_curated")>= f.col("study_start_date")) & (f.col("event_date_curated")<=f.col("study_end_date")), f.concat(f.col("phenotype_ID"), f.lit("_outcome")))
#                                   .otherwise(f.concat(f.col("phenotype_ID"), f.lit("_out_of_range"))))
#             )

# COMMAND ----------

# create flags
union_hf_b =(union_hf
            .withColumn ("indicator", 
                         f.when(f.col("event_date_curated")<f.col("study_start_date"), f.lit("before_study_start"))
                         .when((f.col("event_date_curated")>= f.col("study_start_date")) & (f.col("event_date_curated")<=f.col("study_end_date")), f.lit("during_study_period"))
                                  .otherwise(f.lit("after_study_period")))
            )

# COMMAND ----------

display(union_hf_b.groupBy("name").agg(f.count("*")))

# COMMAND ----------

# Display any phenotypes where event date is 'out_of_range' (later than study end date)
display(union_hf_b.filter(f.col('indicator').like('%after_study_period%')))

# COMMAND ----------

# DBTITLE 1,Remove records with out of range admissions
# Drop out of range 
union_hf_c = union_hf_b.filter(~f.col("indicator").like("after_study_period"))

# COMMAND ----------

display(union_hf_c.groupBy("name").agg(f.count("*")))

# COMMAND ----------

# DBTITLE 1,Get first event date per patient per phenotype during study period
# We need first event event dates across all datasets
w = Window.partitionBy("person_id", "name").orderBy(f.col("event_date_curated").asc_nulls_last())
union_hf_long = (
    union_hf_c
    .withColumn("rank", f.row_number().over(w))
    .filter((f.col("rank")==1) & (f.col('indicator') == 'during_study_period')) # get the first event per patient per phenotype which is during the study period
    .drop("rank")
)
display(union_hf_long.limit(5))

# COMMAND ----------

display(union_hf_long.groupBy("name").agg(f.count("*")))

# COMMAND ----------

# Long to wide
hf_wide = union_hf_long.groupBy("person_id").pivot("name").agg(f.first("event_date_curated"))

# COMMAND ----------

display(hf_wide.limit(10))

# COMMAND ----------

# DBTITLE 1,Extract prevalent cases
union_hf_prevalent_long = (
    union_hf_c
    .filter(f.col('indicator') == 'before_study_start')
)

union_hf_incident_long = (
    union_hf_c
    .filter(f.col('indicator') == 'after_study_period')
)

count_var(union_hf_prevalent_long, 'person_id')
count_var(union_hf_incident_long, 'person_id')

# COMMAND ----------

# Long to wide
union_hf_prevalent_wide = (
    union_hf_prevalent_long
    .groupBy("person_id")
    .pivot("name")
    .agg(f.first("event_date_curated"))
)

union_hf_incident_wide = (
    union_hf_incident_long
    .groupBy("person_id")
    .pivot("name")
    .agg(f.first("event_date_curated"))
)

# COMMAND ----------

# MAGIC %md
# MAGIC # 5. Join with cohort IDs

# COMMAND ----------

count_var(df_base, "PERSON_ID")

# COMMAND ----------

count_var(hf_wide, "person_id")

# COMMAND ----------

cohort_ids = df_base.select(["person_id"])
cohort_ids.cache()

# COMMAND ----------

# DBTITLE 1,Prepare final tables
final_table = cohort_ids.join(hf_wide, on= ["person_id"], how= "left") # this table is limited to first event dates within study period

# COMMAND ----------

final_table_2 = df_base.select('PERSON_ID', 'DOB', 'study_start_age', 'sex', 'eth5', 'DOD').join(hf_wide, on= ["person_id"], how= "left") # this table is limited to first event dates within study period (with additional variables)

# COMMAND ----------

final_table_3 = df_base.select('PERSON_ID', 'DOB', 'study_start_age', 'sex', 'eth5', 'DOD').join(hf_ever, on= ["person_id"], how= "left") # this table contains all heart failure events per patient per phenotype across all time

# COMMAND ----------

final_table_4 = df_base.select('PERSON_ID', 'DOB', 'study_start_age', 'sex', 'eth5', 'DOD').join(union_hf_prevalent_wide, on= ["person_id"], how= "left") # this table contains all prevalent HF cases prior to study start date

# COMMAND ----------

final_table_5 = df_base.select('PERSON_ID', 'DOB', 'study_start_age', 'sex', 'eth5', 'DOD').join(union_hf_incident_wide, on= ["person_id"], how= "left") # this table contains all incident HF cases prior to study start date

# COMMAND ----------

# MAGIC %md
# MAGIC # 6. Save

# COMMAND ----------


# COMMAND ----------

save_table(final_table, out_name = f'{proj}_out_hf_study_period', save_previous=True) # contains only person_id with hf event dates (during study period)

# COMMAND ----------

save_table(final_table_2, out_name = f'{proj}_out_hf_study_period_demographics', save_previous=True) # also contains demographic details of patients (during study period)

# COMMAND ----------

save_table(final_table_3, out_name = f'{proj}_out_hf_ever_cases', save_previous=True) # contains all cases identified from the phenotype definition across all time

# COMMAND ----------

save_table(final_table_4, out_name = f'{proj}_out_hf_prevalent_cases', save_previous=True) # contains all prevalent cases (prior to study start) identified from the phenotype definition 

# COMMAND ----------

count_var(final_table, "person_id")

# COMMAND ----------

display(final_table)

# COMMAND ----------


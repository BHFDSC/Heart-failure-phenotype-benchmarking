# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # HF_07b_add_hf_prior_and_outcome_with_logic
# MAGIC
# MAGIC
# MAGIC **Description** Extract all ('ever') heart failure cases and study period heart failure cases (for the base population used)
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
# MAGIC ## 2.1. Base cohort

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

# DBTITLE 1,Primary care meds
pmeds_bnf_hf = spark.table(f'{dsa}.{proj}_pmeds_all_hf_events')
display(pmeds_bnf_hf.groupBy("name", "phenotype_source").agg(f.count("*")))

# COMMAND ----------

# MAGIC %md
# MAGIC # 4. Extract the 'ever' cases for HF

# COMMAND ----------

# DBTITLE 1,Join all events together across datasets
# Join all events across all datasets
gdppr_hf_sel = gdppr_hf.select(["person_id", "name", "event_date_curated", "phenotype_source"])
hes_apc_hf_sel = hes_apc_hf.select(["person_id", "name", "event_date_curated", "phenotype_source"])
pmeds_hf_sel = pmeds_bnf_hf.select(["person_id", "name", "event_date_curated", "phenotype_source"])


# COMMAND ----------

# DBTITLE 1,Apply logic (phenotypes with logic = ['pheno_18', 'pheno_01', 'pheno_02', 'pheno_13', 'pheno_37', 'pheno_16', 'pheno_33'])
pheno_01 = (
    gdppr_hf
    .select(["person_id", "name", "event_date_curated", "phenotype_source"])
    .unionByName(hes_apc_hf.select(["person_id", "name", "event_date_curated", "phenotype_source"]))
    .filter(f.col('name') == 'pheno_01')
    .filter(f.col('phenotype_source').isin(['GDPPR', 'HES_APC_DIAG']))
    .select(["person_id", "name", "event_date_curated", "phenotype_source"])
)

# COMMAND ----------

pheno_02 = (
    gdppr_hf
    .select(["person_id", "name", "event_date_curated", "phenotype_source"])
    .unionByName(hes_apc_hf.select(["person_id", "name", "event_date_curated", "phenotype_source"]))
    .filter(f.col('name') == 'pheno_02')
    .filter(f.col('phenotype_source').isin(['GDPPR', 'HES_APC_DIAG']))
    .select(["person_id", "name", "event_date_curated", "phenotype_source"])
)

# COMMAND ----------

pheno_13 = (
    hes_apc_hf
    .filter(f.col('name') == 'pheno_13')
    .filter(f.col('diag_position') == 2)
    .select(["person_id", "name", "event_date_curated", "phenotype_source"])
)

# COMMAND ----------

pheno_16 = (
    hes_apc_hf
    .filter(f.col('name') == 'pheno_16')
    .filter(f.col("diag_position") == 1)
    .select(["person_id", "name", "event_date_curated", "phenotype_source"])
)

# COMMAND ----------

pheno_18 = (
    hes_apc_hf
    .filter(f.col('name') == 'pheno_18')
    .filter(f.col("diag_position") == 1)
    .select(["person_id", "name", "event_date_curated", "phenotype_source"])
)

# COMMAND ----------

pheno_33 = (
    hes_apc_hf
    .filter(f.col('name') == 'pheno_33')
    .filter(f.col("diag_position") == 1)
    .select(["person_id", "name", "event_date_curated", "phenotype_source"])
)

# COMMAND ----------

pheno_37 = (
    hes_apc_hf
    .filter(f.col('name') == 'pheno_37')
    .filter(f.col("diag_position") == 1)
    .select(["person_id", "name", "event_date_curated", "phenotype_source"])
)

# COMMAND ----------

union_hf_1 = (
    pheno_01
    .unionByName(pheno_02)
    .unionByName(pheno_13)
    .unionByName(pheno_16)
    .unionByName(pheno_18)
    .unionByName(pheno_33)
    .unionByName(pheno_37)
)

# COMMAND ----------

hf_ever = union_hf_1.select(["person_id", "name", "event_date_curated", "phenotype_source"])

# COMMAND ----------

count_var(hf_ever, 'person_id')

# COMMAND ----------

# MAGIC %md
# MAGIC # 5. Find the first event date of HF (heart failure) per patient per phenotype


# COMMAND ----------

# Add study start and end date
hf_first_event_study_period = (
    union_hf_1
    .withColumn("study_start_date", f.to_date(f.lit(study_start_date)))
    .withColumn("study_end_date", f.to_date(f.lit(study_end_date)))
)
display(hf_first_event_study_period.limit(10))

# COMMAND ----------

# Check
display(hf_first_event_study_period.groupBy("name").agg(f.count("*")))

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

# MAGIC %md
# MAGIC ## Create flags


# COMMAND ----------

# create flags
hf_first_event_study_period_2 =(hf_first_event_study_period
            .withColumn ("indicator", 
                         f.when(f.col("event_date_curated")<f.col("study_start_date"), f.lit("before_study_start"))
                         .when((f.col("event_date_curated")>= f.col("study_start_date")) & (f.col("event_date_curated")<=f.col("study_end_date")), f.lit("during_study_period"))
                                  .otherwise(f.lit("after_study_period")))
            )

# COMMAND ----------

display(hf_first_event_study_period_2.groupBy("name").agg(f.count("*")))

# COMMAND ----------

# Display any phenotypes where event date is 'out_of_range' (later than study end date)
display(hf_first_event_study_period_2.filter(f.col('indicator').like('%after_study_period%')))

# COMMAND ----------

# DBTITLE 1,Remove records with out of range admissions
# Drop out of range 
hf_first_event_study_period_3 = hf_first_event_study_period_2.filter(~f.col("indicator").like("after_study_period"))

# COMMAND ----------

display(hf_first_event_study_period_3.groupBy("name").agg(f.count("*")))

# COMMAND ----------

# DBTITLE 1,Get first event date per patient per phenotype during study period
# We need first event event dates across all datasets
w = Window.partitionBy("person_id", "name").orderBy(f.col("event_date_curated").asc_nulls_last())
hf_first_event_study_period_long = (
    hf_first_event_study_period_3
    .withColumn("rank", f.row_number().over(w))
    .filter((f.col("rank")==1) & (f.col('indicator') == 'during_study_period')) # get the first event per patient per phenotype which is during the study period
    .drop("rank")
)
display(hf_first_event_study_period_long.limit(5))

# COMMAND ----------

display(hf_first_event_study_period_long.groupBy("name").agg(f.count("*")))

# COMMAND ----------

# Long to wide
hf_first_event_study_period_wide = hf_first_event_study_period_long.groupBy("person_id").pivot("name").agg(f.first("event_date_curated"))

# COMMAND ----------

display(hf_first_event_study_period_wide.limit(10))


# COMMAND ----------

# MAGIC %md
# MAGIC # 5. Join with cohort IDs

# COMMAND ----------

count_var(df_base, "PERSON_ID")

# COMMAND ----------

count_var(hf_first_event_study_period_wide, "person_id")

# COMMAND ----------

cohort_ids = df_base.select(["person_id"])
cohort_ids.cache()

# COMMAND ----------

# DBTITLE 1,Prepare final tables
final_table = cohort_ids.join(hf_first_event_study_period_wide, on= ["person_id"], how= "left") # this table is limited to first event dates within study period

# COMMAND ----------

final_table_2 = df_base.select('PERSON_ID', 'DOB', 'study_start_age', 'sex', 'eth5', 'DOD').join(hf_first_event_study_period_wide, on= ["person_id"], how= "left") # this table is limited to first event dates within study period (with additional variables)

# COMMAND ----------

final_table_3 = df_base.select('PERSON_ID', 'DOB', 'study_start_age', 'sex', 'eth5', 'DOD').join(hf_ever, on= ["person_id"], how= "left") # this table contains all heart failure events per patient per phenotype across all time


# COMMAND ----------

# MAGIC %md
# MAGIC # 6. Save

# COMMAND ----------

save_table(final_table, out_name = f'{proj}_out_hf_study_period_phenos_with_logic', save_previous=True) # contains only person_id with hf event dates (during study period)

# COMMAND ----------

save_table(final_table_2, out_name = f'{proj}_out_hf_study_period_demographics_phenos_with_logic', save_previous=True) # also contains demographic details of patients (during study period)

# COMMAND ----------

save_table(final_table_3, out_name = f'{proj}_out_hf_ever_cases_phenos_with_logic', save_previous=True) # contains all cases identified from the phenotype definition across all time


# COMMAND ----------

count_var(final_table, "person_id")

# COMMAND ----------

display(final_table)

# COMMAND ----------


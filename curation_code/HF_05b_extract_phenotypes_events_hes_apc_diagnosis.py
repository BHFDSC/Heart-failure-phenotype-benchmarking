# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # HF_05b_extract_phenotypes_events_hes_apc_diagnosis
# MAGIC
# MAGIC
# MAGIC **Description** Extraction of the all diagnostic heart failure events from HES APC 
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
# MAGIC - **`hf_benchmarking_jl_hes_apc_all_diag_hf_events`** :  

# COMMAND ----------

# MAGIC %md
# MAGIC # Setup

# COMMAND ----------

spark.sql('CLEAR CACHE')
spark.conf.set('spark.sql.legacy.allowCreatingManagedTableUsingNonemptyLocation', 'true')

# COMMAND ----------

# DBTITLE 1,Libraries
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

# DBTITLE 1,Functions
# MAGIC %run "/Workspace/Shared/SHDS/common/functions"

# COMMAND ----------

# MAGIC %md
# MAGIC # 0. Parameters

# COMMAND ----------

# MAGIC %run "./HF_02_main_parameters"

# COMMAND ----------

# MAGIC %md
# MAGIC # 1. Data

# COMMAND ----------

hes_apc_clean = spark.table(f'{path_hes_apc_preprocessed}')

# COMMAND ----------

count_var(hes_apc_clean, "person_id")

# COMMAND ----------

# DBTITLE 1,Codelist
codelist_union = spark.table(f'{dsa}.{proj}_combined_hf_codelist')

# COMMAND ----------

display(codelist_union)

# COMMAND ----------

# MAGIC %md
# MAGIC # 2. Prepare

# COMMAND ----------

display(codelist_union.groupBy("name", "terminology").agg(f.count("*")).orderBy(f.col("name")))

# COMMAND ----------

# Keep diagnostic ICD10 codes
diag_codelist = (
    codelist_union
    .filter(f.col("terminology")=="ICD10")
    #.withColumn("code", f.lower(f.col("code")))
    .withColumn("code", f.regexp_replace(f.col("code"), r"\.", ""))
)
display(diag_codelist.groupBy("name", "terminology").agg(f.count("*")).orderBy(f.col("name")))

# COMMAND ----------

print(diag_codelist.filter(f.col("code").contains(".")).count())

# COMMAND ----------

diag_codelist.columns

# COMMAND ----------

display(diag_codelist.limit(5))

# COMMAND ----------

display(hes_apc_clean.limit(5))

# COMMAND ----------

hes_apc_clean = (
    hes_apc_clean
    #.withColumn("code", f.lower(f.col("code")))
    .withColumn("code", f.regexp_replace(f.col("code"), r"\.", ""))
)

# COMMAND ----------

display(hes_apc_clean.limit(5))

# COMMAND ----------

diag_codelist = (
    diag_codelist
    .drop("terminology")
    .drop("term")
)

# COMMAND ----------

# All  events in HES APC
hes_apc_pheno = (
    hes_apc_clean
    .join(f.broadcast(diag_codelist), on=["code"], how="inner")
)

# COMMAND ----------

count_var(hes_apc_pheno, "person_id")

# COMMAND ----------

hes_apc_pheno.columns

# COMMAND ----------

# Make a clean phenotype  dataframe
hes_apc_pheno_sel = hes_apc_pheno.select(
    [
        "person_id",
        #f.col("ID").alias('phenotype_ID'),
        "name", # "phenotype",
        #"subphenotype",
        "code",
        "event_date_curated",
 'epikey',
 'epistart',
 'epiend',
 'admidate',
 'disdate',
 'diag_column',
 'diag_digits',
 'diag_position',
 'date_of_birth',
 'death_flag',
 'date_of_death',
 'flag_death_and_event_date',
 'date_diff_curated_evdt_DOD',
    ]
).withColumn("phenotype_source", f.lit("HES_APC_DIAG"))

# COMMAND ----------

display(hes_apc_pheno_sel.groupBy("name").agg(f.count("*")))

# COMMAND ----------

display(hes_apc_pheno_sel.groupBy("name").agg(f.count("*")))

# COMMAND ----------

display(hes_apc_pheno_sel.groupBy("name", "phenotype_source").agg(f.count("*")))

# COMMAND ----------

# DBTITLE 1,Check
# Check cases where event date is after death date
tab(hes_apc_pheno_sel, 'flag_death_and_event_date')

# Check what the spread of the number of days difference between event date and death date (where  > 0). Because if simply a few number of days, then can keep these records
tabstat(hes_apc_pheno_sel.filter(f.col('date_diff_curated_evdt_DOD') > 0), 'date_diff_curated_evdt_DOD')

# COMMAND ----------

# DBTITLE 1,Filter
# # Filter out events that are after death date
# hes_apc_pheno_sel = (
#     hes_apc_pheno_sel
#     .filter(f.col('flag_death_and_event_date').isin(['alive', 'death_after_event_date']))
# )

# display(hes_apc_pheno_sel)

# COMMAND ----------

# Check
count_var(hes_apc_pheno_sel, 'person_id')

# COMMAND ----------

# MAGIC %md
# MAGIC # 3. Save

# COMMAND ----------

save_table(hes_apc_pheno_sel, f'{proj}_hes_apc_diag_all_hf_events', save_previous=False)

# COMMAND ----------


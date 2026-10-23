# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # HF_05a_extract_phenotype_events_gdppr
# MAGIC
# MAGIC
# MAGIC **Description** Extraction of the all diagnostic heart failure events from GDPPR 
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
# MAGIC - **`hf_benchmarking_jl_gdppr_all_hf_events`** :  

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
# MAGIC # 2. Load clean GDPPR and codelists

# COMMAND ----------

gdppr_clean = spark.table(f'{dsa}.{proj}_gdppr_preprocessed')

# COMMAND ----------

count_var(gdppr_clean, "person_id")

# COMMAND ----------

codelist_union = spark.table(f'{dsa}.{proj}_combined_hf_codelist')

# COMMAND ----------

display(codelist_union.limit(1))

# COMMAND ----------

# check 
tmpt = tab(codelist_union, 'name', 'terminology')

# COMMAND ----------

# Keep diagnostic SNOMED codes
diag_codelist = codelist_union.filter(f.col("terminology")=="SNOMED CT")
display(diag_codelist.groupBy("name", "terminology").agg(f.count("*")).orderBy(f.col("name")))

# COMMAND ----------

display(diag_codelist.groupBy("name", "terminology").agg(f.count("*")).orderBy(f.col("name")))

# COMMAND ----------

diag_codelist.columns

# COMMAND ----------

display(diag_codelist.limit(5))

# COMMAND ----------

display(gdppr_clean.limit(5))

# COMMAND ----------

gdppr_clean = gdppr_clean.withColumnRenamed("CODD", "code")
diag_codelist = diag_codelist.drop("terminology").drop("description")

# COMMAND ----------

# All  events in GDPPR
gdppr_pheno = gdppr_clean.join(f.broadcast(diag_codelist), on=["code"], how="inner")


# COMMAND ----------

count_var(gdppr_pheno, "person_id")

# COMMAND ----------

gdppr_pheno.columns

# COMMAND ----------

# Make a clean phenotype  dataframe

gdppr_pheno_sel = gdppr_pheno.select(
    [
        "person_id",
        "name", 
        "code",
        "event_date_curated",
        "PRACTICE",
        "GP_SYSTEM_SUPPLIER",
        "PROCESSED_TIMESTAMP",
        "REPORTING_PERIOD_END_DATE",
        "DATE",
        "RECORD_DATE",
        "EPISODE_CONDITION",
        "EPISODE_PRESCRIPTION",
        "VALUE1_CONDITION",
        "VALUE2_CONDITION",
        "date_of_birth",
        "death_flag",
        "date_of_death",
        "flag_death_and_event_date",
        "date_diff_curated_evdt_DOD",
    ]
).withColumn("phenotype_source", f.lit("GDPPR"))

# COMMAND ----------

display(gdppr_pheno_sel.groupBy("name").agg(f.count("*")))

# COMMAND ----------

display(gdppr_pheno_sel.groupBy("name").agg(f.count("*")))

# COMMAND ----------

# Mark the order of event dates and save
w = Window.partitionBy("person_id", "name").orderBy(f.col("event_date_curated").asc_nulls_last())
# w_sub = Window.partitionBy("person_id", "name").orderBy(f.col("event_date_curated").asc_nulls_last())
gdppr_pheno_sel = (gdppr_pheno_sel
                        .withColumn("phenotype_rownum", f.row_number().over(w))
                        .withColumn("phenotype_rank", f.rank().over(w))
                        .withColumn("phenotype_dense_rank", f.dense_rank().over(w))
                        # .withColumn("subphenotype_rownum", f.row_number().over(w_sub))
                        # .withColumn("subphenotype_rank", f.rank().over(w_sub))
                        # .withColumn("subphenotype_dense_rank", f.dense_rank().over(w_sub))

                        )


# COMMAND ----------

gdppr_pheno_sel.write.mode("overwrite").saveAsTable(f'{dsa}.{proj}_gdppr_all_hf_events')

# COMMAND ----------

display(gdppr_pheno_sel.groupBy("name", "phenotype_source").agg(f.count("*")))

# COMMAND ----------


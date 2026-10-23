# Databricks notebook source
# MAGIC %md 
# MAGIC # HF_03b_codelist_hf
# MAGIC
# MAGIC **Project** HF
# MAGIC
# MAGIC **Description** This notebook creates the appended codelists for heart failure (across all phenotype definitions used).
# MAGIC
# MAGIC **Authors** Jadene Lewis (Health Data Science Team, BHF Data Science Centre)
# MAGIC
# MAGIC **Reviewers** ⚠ UNREVIEWED
# MAGIC
# MAGIC **Acknowledgements** 
# MAGIC
# MAGIC **Data Output**
# MAGIC - **``** : 

# COMMAND ----------

# MAGIC %md # 0. Setup

# COMMAND ----------

from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

# COMMAND ----------

# DBTITLE 1,Libraries
import pyspark.sql.functions as f
import pyspark.sql.types as t
from pyspark.sql import Window

from functools import reduce

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

# MAGIC %md # 1. Parameters

# COMMAND ----------

# DBTITLE 1,Common functions
# MAGIC %run "./HF_02_main_parameters"

# COMMAND ----------

# MAGIC %md 
# MAGIC # 2. HF Codelists
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.1. All codelists

# COMMAND ----------

pheno_01 = spark.table(f'{dsa}.pheno_01')
pheno_02 = spark.table(f'{dsa}.pheno_02')
pheno_05 = spark.table(f'{dsa}.pheno_05')
pheno_09 = spark.table(f'{dsa}.pheno_09')
pheno_11 = spark.table(f'{dsa}.pheno_11')
pheno_12 = spark.table(f'{dsa}.pheno_12')
pheno_13 = spark.table(f'{dsa}.pheno_13')
pheno_14 = spark.table(f'{dsa}.pheno_14')
pheno_15 = spark.table(f'{dsa}.pheno_15')
pheno_16 = spark.table(f'{dsa}.pheno_16')
pheno_17 = spark.table(f'{dsa}.pheno_17')
pheno_18 = spark.table(f'{dsa}.pheno_18')
pheno_20 = spark.table(f'{dsa}.pheno_20')
pheno_21 = spark.table(f'{dsa}.pheno_21')
pheno_24 = spark.table(f'{dsa}.pheno_24')
pheno_25 = spark.table(f'{dsa}.pheno_25')
pheno_27 = spark.table(f'{dsa}.pheno_27')
pheno_28 = spark.table(f'{dsa}.pheno_28')
pheno_29 = spark.table(f'{dsa}.pheno_29_v2')
pheno_30 = spark.table(f'{dsa}.pheno_30')
pheno_31 = spark.table(f'{dsa}.pheno_31')
pheno_33 = spark.table(f'{dsa}.pheno_33')
# pheno_34 = spark.table(f'{dsa}.pheno_34')
pheno_36 = spark.table(f'{dsa}.pheno_36_v2')
pheno_37 = spark.table(f'{dsa}.pheno_37')

# COMMAND ----------

# LOADING PHENO_34
import os 
codelist_path = f'/Heart failure phenotyping assessment/pheno_34.csv'
paths = [codelist_path]

spark_dfs = []

for path in paths:
    pandas_df = pd.read_csv(path, keep_default_na=False)
    spark_df = spark.createDataFrame(pandas_df)
    spark_dfs.append(spark_df)

if spark_dfs:
    codelist = spark_dfs[0]
    for df in spark_dfs[1:]:
        codelist = codelist.union(df)

pheno_34 = codelist #.select(f.col("code").alias("CODE"))

# COMMAND ----------

# LOADING pheno_38
import os 
codelist_path = f'/Heart failure phenotyping assessment/pheno_38.csv'
paths = [codelist_path]

spark_dfs = []

for path in paths:
    pandas_df = pd.read_csv(path, keep_default_na=False)
    spark_df = spark.createDataFrame(pandas_df)
    spark_dfs.append(spark_df)

if spark_dfs:
    codelist = spark_dfs[0]
    for df in spark_dfs[1:]:
        codelist = codelist.union(df)

pheno_38 = codelist #.select(f.col("code").alias("CODE"))

# COMMAND ----------

display(pheno_36)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Prepare

# COMMAND ----------

# Identify all dataframes in current workspace starting with 'pheno_'
pheno_dfs = {name: value for name, value in globals().items() if name.startswith('pheno_')}

# COMMAND ----------

print(pheno_dfs)

# COMMAND ----------

display(pheno_34)

# COMMAND ----------

# Define the target column set and order
cols = ['ID', 'name', 'phenotype_name', 'code', 'conceptId', 'term', 'description', 'terminology', 'coding_system', 'system']

# Loop through and adjust each dataframe
for name, df in pheno_dfs.items():
    # Add any missing columns with nulls (None)
    for c in cols:
        if c not in df.columns:
            df = df.withColumn(c, f.lit(None))

    # Populate ID column
    df = df.withColumn('ID', f.lit(name))

    # Reorder columns to match specified order
    df = df.select(cols)
    # Replace the original dataframe in memory with formatted version
    globals()[name] = df

print(f"Formatted {len(pheno_dfs)} dataframes: {list(pheno_dfs.keys())}")

# COMMAND ----------

# DBTITLE 1,Union all codelists
df_names = [f'pheno_{i:02d}' for i in range(1, 39)]

# Get DataFrames from globals by name, ignoring missing ones
pheno_dfs = [globals()[name] for name in df_names if name in globals()]

# Union all DataFrames safely allowing for missing columns
if pheno_dfs:
    combined_codelist = reduce(
        lambda df1, df2: df1.unionByName(df2, allowMissingColumns=True),
        pheno_dfs
    )
    globals()['combined_codelist'] = combined_codelist
    print(f"Union complete across {len(pheno_dfs)} dataframes.")
    combined_codelist.show(5)
else:
    print("No pheno_01 to pheno_37 dataframes found.")

# COMMAND ----------

display(combined_codelist)

# COMMAND ----------

display(combined_codelist.filter((f.col('coding_system').isNull()) & (f.col('terminology').isNull())))

# COMMAND ----------

# Use only the ID, name, code, term and terminology columns
combined_codelist_2 = (
    combined_codelist
    .withColumn('code', f.when(f.col('code').isNotNull(), f.col('code'))
                .otherwise(f.col('conceptId')))
    .withColumn('term', f.when(f.col('term').isNotNull(), f.col('term'))
                .otherwise(f.col('description')))
    .withColumn('terminology', f.when(f.col('terminology').isNotNull(), f.col('terminology'))
                .otherwise(f.col('coding_system')))
    .withColumn('terminology', f.when(f.col('terminology').isin(['BNF codes']), 'BNF')
                .when(f.col('terminology').isin(['SNOMED', 'SNOMED CT', 'SNOMED  CT codes']), 'SNOMED CT')
                .when(f.col('terminology').isin(['ICD10', 'ICD 10', 'ICD10 codes']), 'ICD10')
                .when(f.col('terminology').isin(['ICD9 codes']), 'ICD9')
                .when(f.col('terminology').isNull(), 'SNOMED CT') # NB: This is for pheno_20 and pheno_34 who have SNOMED codes only
                .otherwise(f.col('terminology')))
    .select(f.col('ID').alias('name'), 'code', 'term', 'terminology') 
) 

# COMMAND ----------

# check
tab(combined_codelist_2, 'terminology')

# COMMAND ----------

# check
count_var(combined_codelist_2, 'code')

# COMMAND ----------

display(combined_codelist_2.filter(f.col('code').isNull()))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.2. HF SNOMED
# MAGIC The following phenotypes have SNOMED codes:  
# MAGIC - 

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.3. HF ICD10
# MAGIC The following phenotypes have ICD10 codes only

# COMMAND ----------

# # Filter for phenotypes with only ICD10 codes
# hf_icd10_only = (
#     combined_codelist
#     .filter((f.col('ID').isin(['pheno_13', 'pheno_16', 'pheno_33', 'pheno_33', 'pheno_37'])))
#     .withColumn('name', f.when(f.col('name').isNull(), f.col('phenotype_name'))
#                 .when(f.col('name').isNotNull(), f.col('name'))
#                 .otherwise(f.lit(999999)))
#     .withColumn('term', f.when(f.col('term').isNull(), f.col('description'))
#                 .when(f.col('term').isNotNull(), f.col('term'))
#                 .otherwise(f.lit(999999)))
#     .withColumn('terminology', f.when(f.col('terminology').isNull(), f.col('coding_system'))
#                 .when(f.col('terminology').isNotNull(), f.col('terminology'))
#                 .otherwise(f.lit(999999)))
#     .withColumn('terminology', f.lit('ICD10'))
#     .select('ID', 'name', 'code', 'term', 'terminology')
# )

# # Check
# tab(hf_icd10_only, 'ID')
# tab(hf_icd10_only, 'name')
# tab(hf_icd10_only, 'term')
# tab(hf_icd10_only, 'terminology')

# COMMAND ----------

# display(hf_icd10_only)

# COMMAND ----------

# MAGIC %md
# MAGIC # 3. Quality check

# COMMAND ----------

# check 
tmpt = tab(combined_codelist, 'name', 'terminology')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.1. Check SNOMED code duplication

# COMMAND ----------

# Are snomed codes unique?
snomed_grouped = combined_codelist_2.filter(f.col("terminology") == "SNOMED").groupBy("name", "code").agg(f.count("*").alias("count"))
display(snomed_grouped)

# COMMAND ----------

duplicated_codes = snomed_grouped.filter(f.col("count") > 1).select("code").collect()
duplicated_codes

# COMMAND ----------

for item in duplicated_codes:
    print(ckd_codelist.filter(f.col("code") == item[0]).show())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.2. Check ICD10 code duplication

# COMMAND ----------

# Are ICD10 codes unique?
icd_grouped = combined_codelist_2.filter(f.col("terminology") == "SNOMED").groupBy("name", "code").agg(f.count("*").alias("count"))
display(icd_grouped)

# COMMAND ----------

duplicated_codes = icd_grouped.filter(f.col("count") > 1).select("code").collect()
duplicated_codes

# COMMAND ----------

for item in duplicated_codes:
    print(ckd_codelist.filter(f.col("code") == item[0]).show())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.3. Check BNF code duplication

# COMMAND ----------

# Are OPCS4  codes unique?
bnf_grouped = combined_codelist_2.filter(f.col("terminology") == "BNF").groupBy("name", "code").agg(f.count("*").alias("count"))
display(bnf_grouped)

# COMMAND ----------

duplicated_codes = bnf_grouped.filter(f.col("count") > 1).select("code").collect()
duplicated_codes

# COMMAND ----------

for item in duplicated_codes:
    print(combined_codelist_2.filter(f.col("code") == item[0]).show())

# COMMAND ----------

# MAGIC %md ## 4 How many of the HF codes exist in the SNOMED refset 

# COMMAND ----------

# df_refset = spark.table(path_gdppr_refset)

# COMMAND ----------

# DBTITLE 1,Declare COCP BNF codes
# df_refset_2 = df_refset.withColumnRenamed("ConceptId", "CODE").withColumn("isin_refset", f.lit(1))
# df_snomed = ckd_codelist.filter(f.col("terminology")=="SNOMED").withColumn("isin_snomed_codelist", f.lit(1))
# df_refset_join = df_snomed.join(df_refset_2, on= "CODE", how="left").fillna({"isin_snomed_codelist":0, "isin_refset":0})
            
# display(df_refset_join.groupby("isin_snomed_codelist", "isin_refset").agg(f.count("*")))

# COMMAND ----------

# display(df_refset_join.filter(f.col("isin_refset")==0))

# COMMAND ----------

# MAGIC %md
# MAGIC # Filter for only unique phenotypes

# COMMAND ----------

# Filter for only UNIQUE phenotypes
codes_by_pheno = (
    combined_codelist_2.groupBy("name")
      .agg(f.collect_set("code").alias("codes"))
)

# Self-join to find pairs of phenotypes with identical code sets
same_codelists = (
    codes_by_pheno.alias("a")
    .join(codes_by_pheno.alias("b"),
          f.col("a.codes") == f.col("b.codes"))
    .where(f.col("a.name") != f.col("b.name"))  # avoid duplicates/self matches
    .select(f.col("a.name").alias("pheno_a"),
            f.col("b.name").alias("pheno_b"),
            f.col("a.codes").alias("shared_codes"))
)

display(same_codelists)

# COMMAND ----------

# # Remove either pheno_16 or pheno_33 from combined codelist
# combined_codelist_3 = (
#     combined_codelist_2
#     .filter(~f.col('name').isin(['pheno_33']))
# )

# # check
# tab(combined_codelist_3, 'name')

# COMMAND ----------

# MAGIC %md # 4. Check

# COMMAND ----------

# check 
tmpt = tab(combined_codelist_2, 'name', 'terminology')

# COMMAND ----------

tab(combined_codelist_2.filter(f.col('code').like('%I50%')), 'code')

# COMMAND ----------

# Do codes I502, I508, I502, I503 exist in SDE?
hes_apc_curated = spark.table(hds_curated_assets_hes_apc_long)
display(hes_apc_curated.filter(f.col('CODE').isin(['I502', 'I508', 'I503', 'I504'])))

# COMMAND ----------

# MAGIC %md # 5. Save

# COMMAND ----------

save_table(combined_codelist_2, out_name=f'{proj}_combined_hf_codelist', save_previous=True)

# COMMAND ----------


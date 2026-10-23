# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # HF_05c_extract_phenotypes_events_medications
# MAGIC
# MAGIC
# MAGIC **Description** Extraction of the all heart failure medication phenotype events from primary care dispensed medication
# MAGIC
# MAGIC **Authors** Mehrdad Mizani
# MAGIC
# MAGIC **Reviewers** 
# MAGIC
# MAGIC **Acknowledgements** 
# MAGIC
# MAGIC **Notes**
# MAGIC
# MAGIC **Data Output**
# MAGIC - **`hf_benchmarking_jl_pmeds_all_hf_events`** :  

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
# MAGIC # 2. Load clean PMEDS and codelists

# COMMAND ----------

pmeds_clean = spark.table(f'{dsa}.{proj}_pmeds_preprocessed')

# COMMAND ----------

count_var(pmeds_clean, "person_id")

# COMMAND ----------

display(pmeds_clean.limit(10))

# COMMAND ----------

codelist_union = spark.table(f'{dsa}.{proj}_combined_hf_codelist')

# COMMAND ----------

display(codelist_union.limit(1))

# COMMAND ----------

display(codelist_union.groupBy("phenotype", "terminology").agg(f.count("*")).orderBy(f.col("phenotype")))

# COMMAND ----------

# Keep BNF codes
bnf_codelist = codelist_union.filter(f.col("terminology")=="BNF")
display(bnf_codelist.groupBy("name", "terminology").agg(f.count("*")).orderBy(f.col("name")))

# COMMAND ----------

# Check length of codes
bnf_codelist = bnf_codelist.withColumn("code_length", f.length(f.col("code")))
display(bnf_codelist.groupBy("code_length").agg(f.count("*")))

# COMMAND ----------

# MAGIC %md
# MAGIC All codes are 9 character long (chemical substance level). 

# COMMAND ----------

bnf_codelist.columns

# COMMAND ----------

display(bnf_codelist.limit(5))

# COMMAND ----------

display(pmeds_clean.limit(5))

# COMMAND ----------

pmeds_clean = pmeds_clean.withColumn("code_length", f.length(f.col("PaidBNFCode")))
display(pmeds_clean.groupBy("code_length").agg(f.count("*")))


# COMMAND ----------

pmeds_clean = pmeds_clean.withColumn("code", f.substring(f.col("PaidBNFCode"), 1, 9)).withColumn("code_length_2", f.length(f.col("code")))
display(pmeds_clean.groupBy("code_length",   "code_length_2").agg(f.count("*")))

# COMMAND ----------

display(pmeds_clean.select("person_id", "PaidBNFCode", "code", "PaidBNFName").limit(30))

# COMMAND ----------

bnf_codelist = bnf_codelist.drop("terminology").drop("code_length")

# COMMAND ----------

# All  events in PMEDS
pmeds_pheno = pmeds_clean.join(f.broadcast(bnf_codelist.withColumn("code", f.substring(f.col("code"), 1, 9))), on=["code"], how="inner")

# COMMAND ----------

display(pmeds_pheno.select("person_id", "PaidBNFCode", "code","event_date_curated", "PaidBNFName", "term", "name"))

# COMMAND ----------

count_var(pmeds_pheno, "person_id")

# COMMAND ----------

pmeds_pheno.columns

# COMMAND ----------

# Make a clean phenotype  dataframe
pmeds_pheno_sel = pmeds_pheno.select(
    [
        "person_id",
        "name",
        "code",
        "event_date_curated",
 'BSAPrescriptionID',
 'ChargeStatus',
 'CostCentreLSOA',
 'CostCentreODSCode',
 'CostCentreSubType',
 'CostCentreType',
 'DispensedCountryCode',
 'DispensedPharmacyLSOA',
 'DispensedPharmacyODSCode',
 'DispensedPharmacyType',
 'EPSPrescriptionID',
 'EPSPrescriptionIndicator',
 'ExemptionCode',
 'HighVolVaccineIndicator',
 'ItemActualCost',
 'ItemID',
 'ItemNIC',
 'MaternityExemptionFlag',
 'NotDispensedIndicator',
 'OutOfHoursIndicator',
 'PaidACBSIndicator',
 'PaidBNFName',
 'PaidCDIndicator',
 'PaidDisallowedReason',
 'PaidDissallowedIndicator',
 'PaidDrugStrength',
 'PaidFlavourIndicator',
 'PaidFormulation',
 'PaidPADMIndicator',
 'PaidQuantity',
 'PaidSpecContIndicator',
 'PaidSupplierName',
 'PaiddmdCode',
 'PatientAge',
 'PatientGender',
 'PrescribedBNFCode',
 'PrescribedBNFName',
 'PrescribedCountryCode',
 'PrescribedFormulation',
 'PrescribedMedicineStrength',
 'PrescribedQuantity',
 'PrescribedSupplierName',
 'PrescribeddmdCode',
 'PrescriberType',
 'PrivatePrescriptionIndicator',
 'ProcessedPeriod',
 'ProcessingPeriodDate',
 'date_of_birth',
 'death_flag',
 'date_of_death',
 'flag_death_and_event_date',
 'date_diff_curated_evdt_DOD',
 'term',
 'code_length',
 'code_length_2'
    ]
).withColumn("phenotype_source", f.lit("PMEDS"))

# COMMAND ----------

display(pmeds_pheno_sel.groupBy("name").agg(f.count("*")))

# COMMAND ----------

display(pmeds_pheno_sel.groupBy("name").agg(f.count("*")))

# COMMAND ----------

# Mark the order of event dates and save
w = Window.partitionBy("person_id", "name").orderBy(f.col("event_date_curated").asc_nulls_last())
# w_sub = Window.partitionBy("person_id", "phenotype", "subphenotype").orderBy(f.col("event_date_curated").asc_nulls_last())
pmeds_pheno_sel = (pmeds_pheno_sel
                        .withColumn("phenotype_rownum", f.row_number().over(w))
                        .withColumn("phenotype_rank", f.rank().over(w))
                        .withColumn("phenotype_dense_rank", f.dense_rank().over(w))
                        # .withColumn("subphenotype_rownum", f.row_number().over(w_sub))
                        # .withColumn("subphenotype_rank", f.rank().over(w_sub))
                        # .withColumn("subphenotype_dense_rank", f.dense_rank().over(w_sub))

                        )


# COMMAND ----------

pmeds_pheno_sel.write.mode("overwrite").saveAsTable(f'{dsa}.{proj}_pmeds_all_hf_events')

# COMMAND ----------


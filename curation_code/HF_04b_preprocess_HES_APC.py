# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # HF_benchmarking_04b_preprocess_HES_APC
# MAGIC
# MAGIC **Description** The notebook preprocessed HES APC long by applying basic quality contorls including:
# MAGIC
# MAGIC - Dropping null IDs
# MAGIC - Dropping records where all epistart, epiend, admidate, disdate are null 
# MAGIC - Dropping records where all epistart, epiend, admidate, disdate before date of birth
# MAGIC - Dropping record with null CODE
# MAGIC - Create a curated event date from epistart, epiend, admidate, disdate
# MAGIC - Flag recrods where the event date is larger than date of death 
# MAGIC - Keep columns needed for phenotyping
# MAGIC - Save the table
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
# MAGIC - **`hf_benchmarking_jl_hes_apc_preprocessed`** :

# COMMAND ----------

# MAGIC %md
# MAGIC # 0. Setup

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

# MAGIC %sql
# MAGIC set spark.databricks.pyspark.pythonPriority=HIGH;

# COMMAND ----------

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
# MAGIC # 3. Load long HES APC

# COMMAND ----------

hes_apc_long = spark.table(hds_curated_assets_hes_apc_long)

# COMMAND ----------

display(hes_apc_long.limit(10))

# COMMAND ----------

# display HES APC
# display(hes_apc.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.1 Any null IDs?

# COMMAND ----------

count_var(hes_apc_long, "person_id")

# COMMAND ----------

# MAGIC %md
# MAGIC Action: No missing IDs found. 

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.2 Missing date columns

# COMMAND ----------

# All missing
null_date = (hes_apc_long
              .filter(
                  (f.col("epistart").isNull()) &
                  (f.col("epiend").isNull()) &
                  (f.col("admidate").isNull()) &
                  (f.col("disdate").isNull())
                  )
              )
count_var(null_date, "person_id")

# COMMAND ----------

display(null_date.limit(10))

# COMMAND ----------

# Flag these records, drop later

df_3_2 = (
    hes_apc_long
    .withColumn("qc_missing_date", 
                f.when(
                    (f.col("epistart").isNull()) &
                    (f.col("epiend").isNull()) &
                    (f.col("admidate").isNull()) &
                    (f.col("disdate").isNull()),
                    f.lit(1)
                ).otherwise(
                    f.lit(0)
                )
                )
    )

# COMMAND ----------

display(df_3_2.groupBy(f.col("qc_missing_date")).agg(f.count("*")))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.3 Missing code
# MAGIC

# COMMAND ----------


null_code = (df_3_2
              .filter(
                  (f.col("code").isNull()) 
                  )
              )
count_var(null_code, "person_id")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.4. Missing date of birth or DOB>DOD

# COMMAND ----------

df_demo = spark.table(hds_curated_assets_demographic)

# COMMAND ----------

df_3_4 = (df_3_2
          .join(
              df_demo.select(["person_id", "date_of_birth", 
                              "death_flag", "date_of_death"])
              , on="person_id"
              , how="left"))

# COMMAND ----------

# Null DOB
df_3_4 = (df_3_4
          .withColumn("qc_null_dob", 
                      f.when(
                          f.col("date_of_birth").isNull(), f.lit(1))
                      .otherwise(f.lit(0))
                      )
          )

# COMMAND ----------

display(df_3_4.groupBy("death_flag").agg(f.count("*")))

# COMMAND ----------

# any issues with the death _flag?
df_3_4.filter(f.col("death_flag").isNull()).filter(f.col("date_of_death").isNotNull()).count()

# COMMAND ----------

# Death flag =1 and DOB > DOD
# And missing death flag
df_3_4 = (df_3_4
          .withColumn("qc_dob_after_dod", 
                      f.when(
                          (f.col("date_of_death").isNotNull()) & 
                          (f.col("date_of_birth")>f.col("date_of_death"))
                   , f.lit(1))
                      .otherwise(f.lit(0))
                      )
          .withColumn("qc_missing_dod", 
                      f.when(
                          (f.col("death_flag").isNotNull())&
                          (f.col("date_of_death").isNull())
                          ,f.lit(1)
                      )
                      .otherwise(f.lit(0)))
          )

# COMMAND ----------

display(df_3_4.groupBy("qc_null_dob").agg(f.count("*")))

# COMMAND ----------



# COMMAND ----------

count_var(df_3_4.filter(f.col("qc_null_dob")==1), "person_id")

# COMMAND ----------

display(df_3_4.groupBy("qc_dob_after_dod").agg(f.count("*")))

# COMMAND ----------

count_var(df_3_4.filter(f.col("qc_dob_after_dod")==1), "person_id")

# COMMAND ----------

display(df_3_4.groupBy("qc_missing_dod").agg(f.count("*")))

# COMMAND ----------

count_var(df_3_4.filter(f.col("qc_missing_dod")==1), "person_id")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.5. Curated event date and further QC
# MAGIC
# MAGIC Curate from valid dates in this order:
# MAGIC - epistart
# MAGIC - epiend
# MAGIC - admidate
# MAGIC - disdate
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC Add validity flag:
# MAGIC
# MAGIC - Null
# MAGIC - Before DOB
# MAGIC - After DOD

# COMMAND ----------

df_3_5 = (df_3_4
          .withColumn("qc_epistart", 
                      f.when(f.col("epistart").isNull(), f.lit("null_date"))
                      .when(f.col("epistart")<f.col("date_of_birth"), f.lit("before_dob"))
                      .when(f.col("epistart")>f.col("date_of_death"), f.lit("after_dod"))
                      .otherwise("valid"))
)

df_3_5 = (df_3_5
          .withColumn("qc_epiend", 
                      f.when(f.col("epiend").isNull(), f.lit("null_date"))
                      .when(f.col("epiend")<f.col("date_of_birth"), f.lit("before_dob"))
                      .when(f.col("epiend")>f.col("date_of_death"), f.lit("after_dod"))
                      .otherwise("valid"))
)

df_3_5 = (df_3_5
          .withColumn("qc_admidate", 
                      f.when(f.col("admidate").isNull(), f.lit("null_date"))
                      .when(f.col("admidate")<f.col("date_of_birth"), f.lit("before_dob"))
                      .when(f.col("admidate")>f.col("date_of_death"), f.lit("after_dod"))
                      .otherwise("valid"))
)

df_3_5 = (df_3_5
          .withColumn("qc_disdate", 
                      f.when(f.col("disdate").isNull(), f.lit("null_date"))
                      .when(f.col("disdate")<f.col("date_of_birth"), f.lit("before_dob"))
                      .when(f.col("disdate")>f.col("date_of_death"), f.lit("after_dod"))
                      .otherwise("valid"))
)

# COMMAND ----------

df_3_5.write.mode('overwrite').saveAsTable(f'{dsa}.hf_benchmarking_hes_apc_3_5_temp')

# COMMAND ----------

df_3_5_reload = spark.table(f'{dsa}.hf_benchmarking_hes_apc_3_5_temp')

# COMMAND ----------

display(df_3_5_reload.groupBy("qc_epistart").agg(f.count("*")))

# COMMAND ----------

display(df_3_5_reload.groupBy("qc_epiend").agg(f.count("*")))

# COMMAND ----------

display(df_3_5_reload.groupBy("qc_admidate").agg(f.count("*")))

# COMMAND ----------

display(df_3_5_reload.groupBy("qc_disdate").agg(f.count("*")))

# COMMAND ----------

df_3_5_a = (df_3_5_reload
          .withColumn("epistart_temp", f.when(f.col("epistart")<f.col("date_of_birth"), f.lit(None)).otherwise(f.col("epistart")))
          .withColumn("epiend_temp", f.when(f.col("epiend")<f.col("date_of_birth"), f.lit(None)).otherwise(f.col("epiend")))
          .withColumn("admidate_temp", f.when(f.col("admidate")<f.col("date_of_birth"), f.lit(None)).otherwise(f.col("admidate")))
          .withColumn("disdate_temp", f.when(f.col("disdate")<f.col("date_of_birth"), f.lit(None)).otherwise(f.col("disdate")))
          
          )

# COMMAND ----------

display(df_3_5_a.limit(10))

# COMMAND ----------

df_3_5_a = (df_3_5_a
          .withColumn("event_date_curated", 
                      f.coalesce(
                          f.col("epistart_temp"),
                          f.col("epiend_temp"),
                          f.col("admidate_temp"),
                          f.col("disdate_temp")
                          )
                      )
          )

# COMMAND ----------

count_var(df_3_5_a.filter(f.col("event_date_curated").isNull()),"person_id")

# COMMAND ----------

count_var(df_3_5_a,"person_id")

# COMMAND ----------

# check
display(df_3_5_a.filter(f.col("event_date_curated").isNull()).limit(50))

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Flag null  and <DOB curated event dates

# COMMAND ----------

df_3_5_b = (df_3_5_a
          .withColumn("qc_null_curated_event_date", f.when(f.col("event_date_curated").isNull(), f.lit(1)).otherwise(f.lit(0)))
          .withColumn("qc_curated_event_date_before_dob", f.when(f.col("event_date_curated")<f.col("date_of_birth"), f.lit(1)).otherwise(f.lit(0)))
          )

# COMMAND ----------

display(df_3_5_b.groupBy("qc_null_curated_event_date").agg(f.count("*")))

# COMMAND ----------

count_var(df_3_5_b.filter(f.col("qc_null_curated_event_date")==1),"person_id")

# COMMAND ----------

display(df_3_5_b.groupBy("qc_curated_event_date_before_dob").agg(f.count("*")))

# COMMAND ----------

count_var(df_3_5_b.filter(f.col("qc_curated_event_date_before_dob")==1),"person_id")

# COMMAND ----------

# MAGIC %md
# MAGIC ####Flag records with an eventdate > date of death

# COMMAND ----------

df_3_5_b = (df_3_5_b
          .withColumn("flag_death_and_event_date", f.lit("alive"))
          .withColumn("flag_death_and_event_date", 
                      f.when( (f.col("date_of_death").isNotNull()) & (f.col("date_of_death")>= f.col("event_date_curated")) , f.lit("death_after_event_date")  )
                      .when( (f.col("date_of_death").isNotNull()) & (f.col("date_of_death")< f.col("event_date_curated")) , f.lit("death_before_event_date"))
                      .when((f.col("death_flag")==1) & (f.col("date_of_death").isNull()), f.lit("date_of_death_missing"))
                      .otherwise(f.col("flag_death_and_event_date"))
                      )
          )

# COMMAND ----------

display(df_3_5_b.filter(f.col("flag_death_and_event_date")=="date_of_death_missing"))

# COMMAND ----------

display(df_3_5_b.groupBy("flag_death_and_event_date").agg(f.count("*")))

# COMMAND ----------

# How many patients with invalid date of death?
count_var(df_3_5_b.filter(f.col("flag_death_and_event_date").isin(["death_before_event_date", "date_of_death_missing"])), "person_id")

# COMMAND ----------


count_var(df_3_5_b.filter(f.col("flag_death_and_event_date")== "date_of_death_missing"), "person_id")

# COMMAND ----------

#date diff
df_3_5_c = df_3_5_b.withColumn("date_diff_curated_evdt_DOD", f.datediff(f.col("event_date_curated"), f.col("date_of_death")))

# COMMAND ----------

count_var(df_3_5_c.filter(f.col("date_diff_curated_evdt_DOD")<0), "person_id")

# COMMAND ----------

display(df_3_5_c.groupBy("flag_death_and_event_date").agg(f.count("*")))

# COMMAND ----------

display(df_3_5_c.filter(f.col("date_diff_curated_evdt_DOD")<0).groupBy("flag_death_and_event_date").agg(f.count("*")))

# COMMAND ----------

# MAGIC %md
# MAGIC # Apply final QC and save the table

# COMMAND ----------

df_3_5_c.columns 

# COMMAND ----------

count_var(df_3_5_c, "person_id")

# COMMAND ----------

# Missing DATE and RECORD_DATE
display(df_3_5_c.groupBy("qc_missing_date").agg(f.count("*")))
# Drop missing dates. Drop this column later 

# COMMAND ----------

df_3_5_d = df_3_5_c.filter(f.col("qc_missing_date")==0)
count_var(df_3_5_d, "person_id")

# COMMAND ----------

# Missing CODE (no missing code was found before)
#display(df_3_5_d.groupBy("qc_missing_code").agg(f.count("*")))
# Drop rows with missing code. Drop this column later
#df_3_5_e = df_3_5_d.filter(f.col("qc_missing_code")==0)
#count_var(df_3_5_e, "person_id")
df_3_5_e = df_3_5_d

# COMMAND ----------

# Missing Date of brith 
display(df_3_5_e.groupBy("qc_null_dob").agg(f.count("*")))
# Drop missing DOB.Drop this column later. 

# COMMAND ----------

display(df_3_5_e.filter(f.col("qc_null_dob")==1))

# COMMAND ----------

df_3_5_f = df_3_5_e.filter(f.col("qc_null_dob")==0)
count_var(df_3_5_f, "person_id")

# COMMAND ----------

# date of birth after date of death
display(df_3_5_f.groupBy("qc_dob_after_dod").agg(f.count("*")))
# drop invalid rows, drop this column later

# COMMAND ----------

df_3_5_g = df_3_5_f.filter(f.col("qc_dob_after_dod")==0)
count_var(df_3_5_g, "person_id")

# COMMAND ----------

# death flag = 1 and date of death missing
display(df_3_5_g.groupBy("qc_missing_dod").agg(f.count("*")))
# drop invalid rows, drop this column later

# COMMAND ----------

df_3_5_h = df_3_5_g.filter(f.col("qc_missing_dod")==0)
count_var(df_3_5_h, "person_id")

# COMMAND ----------

# epistart status
display(df_3_5_h.groupBy("qc_epistart").agg(f.count("*")))
# drop this column

# COMMAND ----------

# epiend status
display(df_3_5_h.groupBy("qc_epiend").agg(f.count("*")))
# drop this column

# COMMAND ----------

# admidate status
display(df_3_5_h.groupBy("qc_admidate").agg(f.count("*")))
# drop this column

# COMMAND ----------

# disdate status
display(df_3_5_h.groupBy("qc_disdate").agg(f.count("*")))
# drop this column

# COMMAND ----------

# Check final curated event date

display(df_3_5_h.groupBy("qc_null_curated_event_date").agg(f.count("*")))

# COMMAND ----------

count_var(df_3_5_h.filter(f.col("qc_null_curated_event_date")==1), "person_id")


# COMMAND ----------

count_var(df_3_5_h, "person_id")

# COMMAND ----------

df_3_5_i = df_3_5_h.filter(f.col("qc_null_curated_event_date")==0)
count_var(df_3_5_i, "person_id")

# COMMAND ----------

# Check final curated event date <dob

display(df_3_5_i.groupBy("qc_curated_event_date_before_dob").agg(f.count("*")))

# COMMAND ----------

df_3_5_j = df_3_5_i.filter(f.col("qc_curated_event_date_before_dob")==0)
count_var(df_3_5_j, "person_id")

# COMMAND ----------

# Death
display(df_3_5_j.groupBy("death_flag").agg(f.count("*")))
# replace null with 0

# COMMAND ----------

df_3_5_k = df_3_5_j.withColumn("death_flag", f.when(f.col("death_flag").isNull(), 0).otherwise(f.col("death_flag")))
display(df_3_5_k.groupBy("death_flag").agg(f.count("*")))

# COMMAND ----------

display(df_3_5_k.groupBy("flag_death_and_event_date").agg(f.count("*")))
# keep this columns

# COMMAND ----------

# Final ckecks
df_check = df_3_5_k
print(df_check.filter(f.col("event_date_curated").isNull()).count())
print(df_check.filter(f.col("date_of_birth").isNull()).count())
print(df_check.filter(f.col("CODE").isNull()).count())

print(df_check.filter(f.col("death_flag")==1).filter(f.col("date_of_death").isNull()).count())
print(df_check.filter(f.col("death_flag")==0).filter(f.col("date_of_death").isNotNull()).count())

print(df_check.filter(f.col("event_date_curated")<f.col("date_of_birth")).count())

print(df_check.filter(f.col("event_date_curated")>f.col("date_of_death")).count())

# COMMAND ----------

df_3_5_k.columns

# COMMAND ----------

df_out = df_3_5_k.select([
 'person_id',
 'epikey',
 'epistart',
 'epiend',
 'admidate',
 'disdate',

 'event_date_curated',
 'diag_column',
 'code',
 'diag_digits',
 'diag_position',
 'date_of_birth',
 'death_flag',
 'date_of_death',
 
 'flag_death_and_event_date',
 'date_diff_curated_evdt_DOD'
])

# COMMAND ----------

# hf_benchmarking_params.path_hes_apc_preprocessed

# COMMAND ----------

df_out.write.mode('overwrite').saveAsTable(f'{path_hes_apc_preprocessed}')

# COMMAND ----------

df_out = spark.table(f'{path_hes_apc_preprocessed}')
display(df_out.limit(10))

# COMMAND ----------


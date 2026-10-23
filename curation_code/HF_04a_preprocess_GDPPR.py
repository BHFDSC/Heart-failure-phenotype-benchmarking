# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # HF_04a_preprocess_GDPPR
# MAGIC
# MAGIC **Description** The notebook preprocessed GDPPR by applying basic quality contorls including:
# MAGIC
# MAGIC - Flag null IDs
# MAGIC - Flag records where both DATE and RECORD_DATE are null
# MAGIC - Flag record with null CODE
# MAGIC - Flag patients with null DoB
# MAGIC - Flag records where both DATE and RECORD_DATE are before date of birth
# MAGIC - Create a curated event date from DATE and RECORD_DATE
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
# MAGIC - **`hf_benchmarking_jl_gdppr_preprocessed`** : 

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
# MAGIC # 3. Load GDPPR

# COMMAND ----------

gdppr = extract_batch_from_archive(parameters_df_datasets, 'gdppr')

# COMMAND ----------

# GDPPR
display(gdppr.limit(10))

# COMMAND ----------

# check if gdppr is filtered for the correct archived_on_date
# Note: There should be only one row in the outputs and the date must be checked against the archived_on date in the protocol
display(gdppr.groupBy("archived_on").agg(f.count("*")))

# COMMAND ----------

# Rename ID columns
gdppr = gdppr.withColumnRenamed("NHS_NUMBER_DEID", "person_id")

# COMMAND ----------

# MAGIC %md
# MAGIC # 4. Quality control flags

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.1. Null IDs

# COMMAND ----------

# Check the original gdppr
count_var(gdppr, "person_id")

# COMMAND ----------

# MAGIC %md
# MAGIC Action: No missing IDs found. 

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.2. Missing date columns 
# MAGIC

# COMMAND ----------

# Check: both dates missing
null_date = gdppr.filter(f.col("DATE").isNull()).filter(f.col("RECORD_DATE").isNull())
count_var(null_date, "person_id")

# COMMAND ----------


df_4_2 = (gdppr.withColumn("qc_missing_date", 
                           f.when(
                               (f.col("DATE").isNull() 
                                &
                                (f.col("RECORD_DATE").isNull())
                           ), f.lit(1)).otherwise(f.lit(0))) 
)


# COMMAND ----------

display(df_4_2.groupBy(f.col("qc_missing_date")).agg(f.count("*")))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.3. Missing code column
# MAGIC

# COMMAND ----------

# Check
null_codes = df_4_2.filter(f.col("CODE").isNull())
count_var(null_codes, "person_id")

# COMMAND ----------

df_4_3 = (df_4_2
          .withColumn("qc_missing_code", 
                      f.when(
                        f.col("CODE").isNull(), f.lit(1)
                      ).otherwise(f.lit(0))
                      )
          )
       


# COMMAND ----------

display(df_4_3.groupBy(f.col("qc_missing_code")).agg(f.count("*")))

# COMMAND ----------

# MAGIC %md
# MAGIC ###  Select columns and save a temporary checkpoint (drop later)

# COMMAND ----------

df_4_3.columns 

# COMMAND ----------

df_4_3_temp =(df_4_3
         .select([
             "archived_on", 
             "person_id",
             "PRACTICE", 
             "GP_SYSTEM_SUPPLIER",
             "PROCESSED_TIMESTAMP",
             "REPORTING_PERIOD_END_DATE",
             "DATE",
             "RECORD_DATE",
             "CODE",
             "EPISODE_CONDITION",
             "EPISODE_PRESCRIPTION",
             "VALUE1_CONDITION",
             "VALUE2_CONDITION",
             "qc_missing_date", 
             "qc_missing_code"
         ]))

# COMMAND ----------

df_4_3_temp.write.mode('overwrite').saveAsTable(f'{dsa}.{proj}_gdppr_4_3_temp')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.4. Relaod the temporary table and link with patient characteristics table

# COMMAND ----------

df_4_3_reload  = spark.table(f'{dsa}.{proj}_gdppr_4_3_temp')

# COMMAND ----------

count_var(df_4_3_reload, "person_id")

# COMMAND ----------

df_demo = spark.table(hds_curated_assets_demographic)

# COMMAND ----------

display(df_demo.limit(10))

# COMMAND ----------

df_4_4 = (df_4_3_reload
          .join(
              df_demo.select(["person_id", "date_of_birth", 
                              "death_flag", "date_of_death"])
              , on="person_id"
              , how="left"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.5. Null date of birth or DOB>DOD

# COMMAND ----------

df_4_5 = (df_4_4
          .withColumn("qc_null_dob", 
                      f.when(
                          f.col("date_of_birth").isNull(), f.lit(1))
                      .otherwise(f.lit(0))
                      )
          )


# COMMAND ----------

display(df_4_5.groupBy("death_flag").agg(f.count("*")))

# COMMAND ----------

# any issues with the death _flag?
df_4_5.filter(f.col("death_flag").isNull()).filter(f.col("date_of_death").isNotNull()).count()

# COMMAND ----------

# Death flag =1 and DOB > DOD
# And missing death flag
df_4_5 = (df_4_5
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

display(df_4_5.groupBy("qc_null_dob").agg(f.count("*")))

# COMMAND ----------

count_var(df_4_5.filter(f.col("qc_null_dob")==1), "person_id")

# COMMAND ----------

display(df_4_5.groupBy("qc_dob_after_dod").agg(f.count("*")))

# COMMAND ----------

count_var(df_4_5.filter(f.col("qc_dob_after_dod")==1), "person_id")

# COMMAND ----------

display(df_4_5.groupBy("qc_missing_dod").agg(f.count("*")))

# COMMAND ----------

count_var(df_4_5.filter(f.col("qc_missing_dod")==1), "person_id")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Curated event date and further QC
# MAGIC
# MAGIC DATE is the primary event date column. RECORD_DATE will be used for imputation and date correction were DATE is null or < date of birth
# MAGIC
# MAGIC

# COMMAND ----------

# Add a flag to check the quality of the event date
df_5 = (df_4_5
        .withColumn("qc_date_status", f.lit("DATE_and_RECORD_DATE_not_null"))
        .withColumn("qc_date_status",
                    f.when((f.col("DATE").isNull()) & (f.col("RECORD_DATE").isNull()), f.lit("DATE_and_RECORD_DATE_null"))
                    .when((f.col("DATE").isNull()) & (f.col("RECORD_DATE").isNotNull()), f.lit("DATE_null_RECORD_DATE_not_null"))
                    .when((f.col("DATE").isNotNull()) & (f.col("RECORD_DATE").isNull()), f.lit("DATE_not_null_RECORD_DATE_null"))
                    .otherwise(f.col("qc_date_status"))
                    )
        )


df_5 = (df_5
        .withColumn("qc_date_dob_status", f.lit("valid"))
        .withColumn(
               "qc_date_dob_status", 
               f.when(f.col("DATE").isNull() & f.col("RECORD_DATE").isNull(), "DATE_and_RECORD_DATE_null")
               .when(
                      (f.col("DATE").isNotNull()) 
                      & (f.col("DATE") < f.col("date_of_birth")) 
                      & ((f.col("RECORD_DATE").isNull()) | (f.col("RECORD_DATE") >= f.col("date_of_birth"))), "DATE_before_DoB"
                      )
               .when(
                      (f.col("RECORD_DATE").isNotNull()) 
                      & (f.col("RECORD_DATE") < f.col("date_of_birth")) 
                      & ((f.col("DATE").isNull()) | (f.col("DATE") >= f.col("date_of_birth"))), "RECORD_DATE_before_DoB"
                      )
               .when(
                      (f.col("DATE").isNotNull()) 
                      & (f.col("RECORD_DATE").isNotNull()) 
                      & (f.col("DATE") < f.col("date_of_birth")) & (f.col("RECORD_DATE") < f.col("date_of_birth")), "DATE_and_RECORD_DATE_before_DoB")
                       .otherwise(f.col("qc_date_dob_status")
               )
        
         )

       )



df_5 = (df_5
        .withColumn("qc_date_dod_status", f.lit("valid"))
        .withColumn(
               "qc_date_dod_status", 
               f.when(f.col("DATE").isNull() & f.col("RECORD_DATE").isNull(), "DATE_and_RECORD_DATE_null")
               .when(
                      (f.col("DATE").isNotNull()) 
                      & (f.col("DATE") > f.col("date_of_death")) 
                      & ((f.col("RECORD_DATE").isNull()) | (f.col("RECORD_DATE") <= f.col("date_of_death"))), "DATE_after_DoD"
                      )
               .when(
                      (f.col("RECORD_DATE").isNotNull()) 
                      & (f.col("RECORD_DATE") > f.col("date_of_death")) 
                      & ((f.col("DATE").isNull()) | (f.col("DATE") <= f.col("date_of_death"))), "RECORD_DATE_after_DoD"
                      )
               .when(
                      (f.col("DATE").isNotNull()) 
                      & (f.col("RECORD_DATE").isNotNull()) 
                      & (f.col("DATE") > f.col("date_of_death")) & (f.col("RECORD_DATE") > f.col("date_of_death")), "DATE_and_RECORD_DATE_after_Dod")
                      .otherwise(f.col("qc_date_dod_status")
               )
         
         )

       )

# COMMAND ----------

#spark.sql(f"DROP TABLE IF EXISTS {hf_benchmarking_params.dbc}.hf_benchmarking_gdppr_5_temp")

# COMMAND ----------

df_5.write.mode('overwrite').saveAsTable(f'{dsa}.hf_benchmarking_gdppr_5_temp')

# COMMAND ----------

df_5_reload = spark.table(f'{dsa}.hf_benchmarking_gdppr_5_temp')

# COMMAND ----------

# check 
display(df_5_reload.groupBy("qc_date_status").agg(f.count("*")))

# COMMAND ----------

# check 
display(df_5_reload.groupBy("qc_date_dob_status").agg(f.count("*")))

# COMMAND ----------

# check 
display(df_5_reload.groupBy("qc_date_dod_status").agg(f.count("*")))

# COMMAND ----------

# Only Date null 
display(df_5_reload.filter(f.col("qc_date_status")=="DATE_null_RECORD_DATE_not_null").limit(100))

# COMMAND ----------

#  Date or RECORD_DATE < date of birth 
display(df_5_reload.filter(f.col("qc_date_dob_status")=="DATE_and_RECORD_DATE_before_DoB"))

# COMMAND ----------

#  Date or RECORD_DATE > date of death 
display(df_5_reload.filter(f.col("qc_date_dod_status")=="DATE_after_DoD").limit(100))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Date difference between DOB/DOD and DATE and RECORD_DATE

# COMMAND ----------

# date diffs
df_5_2 = df_5_reload.withColumn("date_diff_DATE_DOB", f.datediff(f.col("DATE"), f.col("date_of_birth")))
df_5_2 = df_5_2.withColumn("date_diff_RECORD_DATE_DOB", f.datediff(f.col("RECORD_DATE"), f.col("date_of_birth")))
df_5_2 = df_5_2.withColumn("date_diff_DATE_DOD", f.datediff(f.col("DATE"), f.col("date_of_death")))
df_5_2 = df_5_2.withColumn("date_diff_RECORD_DATE_DOD", f.datediff(f.col("RECORD_DATE"), f.col("date_of_death")))

# COMMAND ----------

display(df_5_2.filter(f.col("date_diff_DATE_DOB")<0))

# COMMAND ----------

display(df_5_2.filter(f.col("date_diff_RECORD_DATE_DOB")<0))

# COMMAND ----------

display(df_5_2.filter(f.col("date_diff_DATE_DOD")>0))

# COMMAND ----------

display(df_5_2.filter(f.col("date_diff_RECORD_DATE_DOD")>0))

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 6. Curated  event date and status flag
# MAGIC
# MAGIC Strategy:
# MAGIC
# MAGIC - Use only DATE or RECORD_DATE >= DOB 
# MAGIC - Use DATE, if valid
# MAGIC - Otherwise use RECORD_DATE, if valid
# MAGIC
# MAGIC - Keep a flag showing the date different between the final event date and DOD for later QC
# MAGIC - Drop records with missing date of death (if death flag is 1) and save the final table

# COMMAND ----------

df_6_a = (df_5_2
                .withColumn("temp_DATE",
                             f.when(f.col("DATE")>=f.col("date_of_birth"),f.col("DATE"))
                             .otherwise(f.lit(None)))
                .withColumn("temp_RECORD_DATE", 
                            f.when(f.col("RECORD_DATE")>=f.col("date_of_birth"),f.col("RECORD_DATE"))
                            .otherwise(f.lit(None)))

)


# COMMAND ----------

display(df_6_a)

# COMMAND ----------

display(df_6_a.filter(f.col("temp_DATE").isNull()))

# COMMAND ----------

display(df_6_a.filter(f.col("temp_RECORD_DATE").isNull()))

# COMMAND ----------

display(df_6_a.filter(f.col("temp_RECORD_DATE").isNull()).filter(f.col("temp_DATE").isNull()))

# COMMAND ----------

df_6_a = df_6_a.withColumn("event_date_curated", f.coalesce(f.col("temp_DATE"), f.col("temp_RECORD_DATE")))

# COMMAND ----------

count_var(df_6_a.filter(f.col("event_date_curated").isNull()),"person_id")

# COMMAND ----------

count_var(df_6_a,"person_id")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Flag null  and <DOB curated event dates

# COMMAND ----------

df_6_b = (df_6_a
          .withColumn("qc_null_curated_event_date", f.when(f.col("event_date_curated").isNull(), f.lit(1)).otherwise(f.lit(0)))
          .withColumn("qc_curated_event_date_before_dob", f.when(f.col("event_date_curated")<f.col("date_of_birth"), f.lit(1)).otherwise(f.lit(0)))
          )



# COMMAND ----------

display(df_6_b.groupBy("qc_null_curated_event_date").agg(f.count("*")))

# COMMAND ----------

count_var(df_6_b.filter(f.col("qc_null_curated_event_date")==1),"person_id")

# COMMAND ----------

display(df_6_b.groupBy("qc_curated_event_date_before_dob").agg(f.count("*")))

# COMMAND ----------

count_var(df_6_b.filter(f.col("qc_curated_event_date_before_dob")==1),"person_id")

# COMMAND ----------

# MAGIC %md
# MAGIC ####Flag records with an eventdate > date of death

# COMMAND ----------

display(df_6_b.limit(10))

# COMMAND ----------

df_6_b = (df_6_b
          .withColumn("flag_death_and_event_date", f.lit("alive"))
          .withColumn("flag_death_and_event_date", 
                      f.when( (f.col("date_of_death").isNotNull()) & (f.col("date_of_death")>= f.col("event_date_curated")) , f.lit("death_after_event_date")  )
                      .when( (f.col("date_of_death").isNotNull()) & (f.col("date_of_death")< f.col("event_date_curated")) , f.lit("death_before_event_date"))
                      .when((f.col("death_flag")==1) & (f.col("date_of_death").isNull()), f.lit("date_of_death_missing"))
                      .otherwise(f.col("flag_death_and_event_date"))
                      )
          )

# COMMAND ----------

display(df_6_b.filter(f.col("flag_death_and_event_date")=="date_of_death_missing"))

# COMMAND ----------

display(df_6_b.groupBy("flag_death_and_event_date").agg(f.count("*")))

# COMMAND ----------

# How many patients with invalid date of death?
count_var(df_6_b.filter(f.col("flag_death_and_event_date").isin(["death_before_event_date", "date_of_death_missing"])), "person_id")

# COMMAND ----------


count_var(df_6_b.filter(f.col("flag_death_and_event_date")== "date_of_death_missing"), "person_id")

# COMMAND ----------

#date diff
df_6_c = df_6_b.withColumn("date_diff_curated_evdt_DOD", f.datediff(f.col("event_date_curated"), f.col("date_of_death")))


# COMMAND ----------

count_var(df_6_c.filter(f.col("date_diff_curated_evdt_DOD")<0), "person_id")

# COMMAND ----------


display(df_6_c.groupBy("flag_death_and_event_date").agg(f.count("*")))


# COMMAND ----------

display(df_6_c.filter(f.col("date_diff_curated_evdt_DOD")<0).groupBy("flag_death_and_event_date").agg(f.count("*")))

# COMMAND ----------

# MAGIC %md
# MAGIC # Apply final QC and save the table

# COMMAND ----------

df_6_c.columns

# COMMAND ----------

count_var(df_6_c, "person_id")

# COMMAND ----------

# Missing DATE and RECORD_DATE
display(df_6_c.groupBy("qc_missing_date").agg(f.count("*")))
# Drop missing dates. Drop this column later 

# COMMAND ----------

df_6_d = df_6_c.filter(f.col("qc_missing_date")==0)
count_var(df_6_d, "person_id")

# COMMAND ----------

# Missing CODE
display(df_6_d.groupBy("qc_missing_code").agg(f.count("*")))
# Drop rows with missing code. Drop this column later

# COMMAND ----------

df_6_e = df_6_d.filter(f.col("qc_missing_code")==0)
count_var(df_6_e, "person_id")

# COMMAND ----------

 
# Missing Date of brith 
display(df_6_e.groupBy("qc_null_dob").agg(f.count("*")))
# Drop missing DOB.Drop this column later. 

# COMMAND ----------

df_6_f = df_6_e.filter(f.col("qc_null_dob")==0)
count_var(df_6_f, "person_id")

# COMMAND ----------

# date of birth after date of death
display(df_6_f.groupBy("qc_dob_after_dod").agg(f.count("*")))
# drop invalid rows, drop this column later

# COMMAND ----------

df_6_g = df_6_f.filter(f.col("qc_dob_after_dod")==0)
count_var(df_6_g, "person_id")

# COMMAND ----------

# death flag = 1 and date of death missing
display(df_6_g.groupBy("qc_missing_dod").agg(f.count("*")))
# drop invalid rows, drop this column later

# COMMAND ----------

df_6_h = df_6_g.filter(f.col("qc_missing_dod")==0)
count_var(df_6_h, "person_id")

# COMMAND ----------

# DATE and RECORD_DATE status
display(df_6_h.groupBy("qc_date_status").agg(f.count("*")))
# drop this column

# COMMAND ----------

# DATE, RECORD_DATE, DOB
display(df_6_h.groupBy("qc_date_dob_status").agg(f.count("*")))
# drop this column

# COMMAND ----------

# DATE, RECORD_DATE, DOD
display(df_6_h.groupBy("qc_date_dod_status").agg(f.count("*")))
# drop this column

# COMMAND ----------

# Check final curated event date

display(df_6_h.groupBy("qc_null_curated_event_date").agg(f.count("*")))

# COMMAND ----------

count_var(df_6_h.filter(f.col("qc_null_curated_event_date")==1), "person_id")

# COMMAND ----------

count_var(df_6_h, "person_id")

# COMMAND ----------

df_6_i = df_6_h.filter(f.col("qc_null_curated_event_date")==0)
count_var(df_6_i, "person_id")

# COMMAND ----------

# Check final curated event date <dob

display(df_6_i.groupBy("qc_curated_event_date_before_dob").agg(f.count("*")))

# COMMAND ----------

df_6_j = df_6_i.filter(f.col("qc_curated_event_date_before_dob")==0)
count_var(df_6_j, "person_id")

# COMMAND ----------

# Death
display(df_6_j.groupBy("death_flag").agg(f.count("*")))
# replace null with 0

# COMMAND ----------

df_6_k = df_6_j.withColumn("death_flag", f.when(f.col("death_flag").isNull(), 0).otherwise(f.col("death_flag")))
display(df_6_k.groupBy("death_flag").agg(f.count("*")))


# COMMAND ----------

display(df_6_k.groupBy("flag_death_and_event_date").agg(f.count("*")))
# keep this columns

# COMMAND ----------

# Final ckecks
df_check = df_6_k
print(df_check.filter(f.col("event_date_curated").isNull()).count())
print(df_check.filter(f.col("date_of_birth").isNull()).count())
print(df_check.filter(f.col("CODE").isNull()).count())

print(df_check.filter(f.col("death_flag")==1).filter(f.col("date_of_death").isNull()).count())
print(df_check.filter(f.col("death_flag")==0).filter(f.col("date_of_death").isNotNull()).count())

print(df_check.filter(f.col("event_date_curated")<f.col("date_of_birth")).count())

print(df_check.filter(f.col("event_date_curated")>f.col("date_of_death")).count())

# COMMAND ----------

df_6_k.columns

# COMMAND ----------

# Select columns
df_6_out = (df_6_k.select(['person_id',
 'archived_on',
 'PRACTICE',
 'GP_SYSTEM_SUPPLIER',
 'PROCESSED_TIMESTAMP',
 'REPORTING_PERIOD_END_DATE',
 'DATE',
 'RECORD_DATE',
 'event_date_curated',
 'CODE',
 'EPISODE_CONDITION',
 'EPISODE_PRESCRIPTION',
 'VALUE1_CONDITION',
 'VALUE2_CONDITION',
 'date_of_birth',
 'death_flag',
 'date_of_death',
 'flag_death_and_event_date',
 'date_diff_curated_evdt_DOD']
))

# COMMAND ----------

df_6_out.write.mode('overwrite').saveAsTable(f'{dsa}.hf_benchmarking_gdppr_clean_jl_251020')

# COMMAND ----------

df_6_out_reload = spark.table(f'{dsa}.hf_benchmarking_gdppr_clean_jl_251020')

# COMMAND ----------

df_6_out_reload.write.mode('overwrite').saveAsTable(f'{dsa}.hf_benchmarking_jl_gdppr_preprocessed')

# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # HF_04d_preprocess_PMEDS
# MAGIC
# MAGIC **Description** The notebook preprocess primary care medicaitons (PMEDS) by applying basic quality contorls including:
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
# MAGIC - **`hf_benchmarking_jl_pmeds_preprocessed`** : 

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
# MAGIC # 3. Load PMEDS

# COMMAND ----------

pmeds = extract_batch_from_archive(parameters_df_datasets, 'pmeds')

# COMMAND ----------

# PMEDS
display(pmeds.limit(10))

# COMMAND ----------

# check if pmeds is filtered for the correct archived_on_date
# Note: There should be only one row in the outputs and the date must be checked against the archived_on date in the protocol
display(pmeds.groupBy("archived_on").agg(f.count("*")))

# COMMAND ----------

# Rename ID columns
pmeds = pmeds.withColumnRenamed("Person_ID_DEID", "person_id")

# COMMAND ----------

# MAGIC %md
# MAGIC # 4. Quality control flags

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.1. Null IDs

# COMMAND ----------

# Check the original pmeds
count_var(pmeds, "person_id")

# COMMAND ----------

df_4_1 = (pmeds.withColumn("qc_missing_id", 
                           f.when(
                               (f.col("person_id").isNull() 
                               
                           ), f.lit(1)).otherwise(f.lit(0))) 
)

# COMMAND ----------

display(df_4_1.groupBy(f.col("qc_missing_id")).agg(f.count("*")))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.2. Missing date columns 
# MAGIC

# COMMAND ----------


df_4_2 = (df_4_1.withColumn("qc_missing_date", 
                           f.when(
                               (f.col("ProcessingPeriodDate").isNull() 
                             
                           ), f.lit(1)).otherwise(f.lit(0))) 
)


# COMMAND ----------

display(df_4_2.groupBy(f.col("qc_missing_date")).agg(f.count("*")))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.3. Missing code column
# MAGIC

# COMMAND ----------

df_4_3 = (df_4_2
          .withColumn("qc_missing_code", 
                      f.when(
                        f.col("PaidBNFCode").isNull(), f.lit(1)
                      ).otherwise(f.lit(0))
                      )
          )
       


# COMMAND ----------

display(df_4_3.groupBy(f.col("qc_missing_code")).agg(f.count("*")))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.4. Link with pateint characteristic table

# COMMAND ----------

df_demo = spark.table(hds_curated_assets_demographic)

# COMMAND ----------

display(df_demo.limit(10))

# COMMAND ----------

df_4_4 = (df_4_3
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

# MAGIC %md
# MAGIC ## Save a temporary table to speed up the operations

# COMMAND ----------

df_4_5.write.mode("overwrite").saveAsTable(f'{dsa}.{proj}_meds_temp_4_5')

# COMMAND ----------

df_4_5_reload = spark.table(f'{dsa}.{proj}_meds_temp_4_5')

# COMMAND ----------

display(df_4_5_reload.groupBy("qc_null_dob").agg(f.count("*")))

# COMMAND ----------

count_var(df_4_5_reload.filter(f.col("qc_null_dob")==1), "person_id")

# COMMAND ----------

display(df_4_5_reload.groupBy("qc_dob_after_dod").agg(f.count("*")))

# COMMAND ----------

count_var(df_4_5_reload.filter(f.col("qc_dob_after_dod")==1), "person_id")

# COMMAND ----------

display(df_4_5_reload.groupBy("qc_missing_dod").agg(f.count("*")))

# COMMAND ----------

count_var(df_4_5_reload.filter(f.col("qc_missing_dod")==1), "person_id")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Curated event date and further QC
# MAGIC  We know ProcessingPeriodDate is not null. Check if ProcessingPeriodDate < date of birth, or dob is null
# MAGIC

# COMMAND ----------

# Add a flag to check the quality of the event date

df_5 = (df_4_5_reload
        .withColumn("qc_date_dob_status", f.lit("valid"))
        .withColumn(
               "qc_date_dob_status", 
               f.when(f.col("ProcessingPeriodDate").isNull() , "ProcessingPeriodDate_null")
               .when(
                      (f.col("ProcessingPeriodDate").isNotNull()) 
                      & (f.col("ProcessingPeriodDate") < f.col("date_of_birth")) 
                     , "ProcessingPeriodDate_before_DoB"
                      )
              .otherwise(f.col("qc_date_dob_status"))
              
        
         )

       )



df_5 = (df_5
        .withColumn("qc_date_dod_status", f.lit("valid"))
        .withColumn(
               "qc_date_dod_status", 
               f.when(f.col("ProcessingPeriodDate").isNull(), "ProcessingPeriodDate_null")
               .when(
                      (f.col("ProcessingPeriodDate").isNotNull()) 
                      & (f.col("ProcessingPeriodDate") > f.col("date_of_death")) 
                      , "ProcessingPeriodDate_after_DoD"
                      )
               .otherwise(f.col("qc_date_dod_status"))
          
         
         )

       )

# COMMAND ----------

# check 
display(df_5.groupBy("qc_date_dob_status").agg(f.count("*")))

# COMMAND ----------

# check 
display(df_5.groupBy("qc_date_dod_status").agg(f.count("*")))

# COMMAND ----------

# ProcessingPeriodDate < date of birth 
display(df_5.filter(f.col("qc_date_dob_status")=="ProcessingPeriodDate_before_DoB").limit(20))

# COMMAND ----------

#  ProcessingPeriodDate > date of death 
display(df_5.filter(f.col("qc_date_dod_status")=="ProcessingPeriodDate_after_DoD").limit(20))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Date difference between DOB/DOD and ProcessingPeriodDate

# COMMAND ----------

# date diffs
df_5_2 = df_5.withColumn("date_diff_ProcessingPeriodDate_DOB", f.datediff(f.col("ProcessingPeriodDate"), f.col("date_of_birth")))

df_5_2 = df_5_2.withColumn("date_diff_ProcessingPeriodDate_DOD", f.datediff(f.col("ProcessingPeriodDate"), f.col("date_of_death")))

# COMMAND ----------

display(df_5_2.filter(f.col("date_diff_ProcessingPeriodDate_DOB")<0).limit(20))

# COMMAND ----------

display(df_5_2.filter(f.col("date_diff_ProcessingPeriodDate_DOD")>0).limit(20))

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 6. Curated  event date and status flag
# MAGIC
# MAGIC Strategy:
# MAGIC
# MAGIC - Use only ProcessingPeriodDate >= DOB 
# MAGIC
# MAGIC - Keep a flag showing the date different between the final event date and DOD for later QC
# MAGIC - Drop records with missing date of death (if death flag is 1) and save the final table
# MAGIC
# MAGIC Note: some of the checks after this step are already done. We kept them here just in  case a new curated event date is created from multiple columns at a later run. 

# COMMAND ----------

 df_6_a = df_5_2.withColumn("event_date_curated", f.col("ProcessingPeriodDate"))

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

display(df_6_b.groupBy("flag_death_and_event_date").agg(f.count("*")))

# COMMAND ----------



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

# Missing id
display(df_6_c.groupBy("qc_missing_id").agg(f.count("*")))
# Drop missing IDs. Drop this column later 

# COMMAND ----------

count_var(df_6_c.filter(f.col("qc_missing_id")==1), "person_id")

# COMMAND ----------

df_6_d = df_6_c.filter(f.col("qc_missing_id")==0)
count_var(df_6_d, "person_id")

# COMMAND ----------

# Missing ProcessingPeriodDate
display(df_6_d.groupBy("qc_missing_date").agg(f.count("*")))
# Drop missing dates. Drop this column later 

# COMMAND ----------

#df_6_e = df_6_d.filter(f.col("qc_missing_date")==0)
#count_var(df_6_e, "person_id")

# COMMAND ----------

# Missing CODE
display(df_6_d.groupBy("qc_missing_code").agg(f.count("*")))
# Drop rows with missing code. Drop this column later

# COMMAND ----------

#df_6_e = df_6_d.filter(f.col("qc_missing_code")==0)
#count_var(df_6_e, "person_id")

# COMMAND ----------

 
# Missing Date of brith 
display(df_6_d.groupBy("qc_null_dob").agg(f.count("*")))
# Drop missing DOB.Drop this column later. 

# COMMAND ----------

count_var(df_6_d.filter(f.col("qc_null_dob")==1), "person_id")

# COMMAND ----------

df_6_f = df_6_d.filter(f.col("qc_null_dob")==0)
count_var(df_6_f, "person_id")

# COMMAND ----------

# date of birth after date of death
display(df_6_f.groupBy("qc_dob_after_dod").agg(f.count("*")))
# drop invalid rows, drop this column later

# COMMAND ----------

count_var(df_6_f.filter(f.col("qc_dob_after_dod")==1), "person_id")

# COMMAND ----------

df_6_g = df_6_f.filter(f.col("qc_dob_after_dod")==0)
count_var(df_6_g, "person_id")

# COMMAND ----------

# death flag = 1 and date of death missing
display(df_6_g.groupBy("qc_missing_dod").agg(f.count("*")))
# drop invalid rows, drop this column later

# COMMAND ----------

count_var(df_6_g.filter(f.col("qc_missing_dod")==1), "person_id")

# COMMAND ----------

df_6_h = df_6_g.filter(f.col("qc_missing_dod")==0)
count_var(df_6_h, "person_id")

# COMMAND ----------

#  DOB
display(df_6_h.groupBy("qc_date_dob_status").agg(f.count("*")))
# drop this column

# COMMAND ----------

#  DOD
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

#df_6_i = df_6_h.filter(f.col("qc_null_curated_event_date")==0)
#count_var(df_6_i, "person_id")
df_6_i = df_6_h

# COMMAND ----------

# Check final curated event date <dob

display(df_6_i.groupBy("qc_curated_event_date_before_dob").agg(f.count("*")))

# COMMAND ----------

count_var(df_6_i.filter(f.col("qc_curated_event_date_before_dob")==1), "person_id")

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
print(df_check.filter(f.col("PaidBNFCode").isNull()).count())

print(df_check.filter(f.col("death_flag")==1).filter(f.col("date_of_death").isNull()).count())
print(df_check.filter(f.col("death_flag")==0).filter(f.col("date_of_death").isNotNull()).count())

print(df_check.filter(f.col("event_date_curated")<f.col("date_of_birth")).count())

print(df_check.filter(f.col("event_date_curated")>f.col("date_of_death")).count())

# COMMAND ----------

df_6_k.columns

# COMMAND ----------

# Select columns
df_6_out = (df_6_k.select(['person_id',
 'event_date_curated',
 'PaidBNFCode',
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
 'date_diff_curated_evdt_DOD']
))

# COMMAND ----------

df_6_out.write.mode('overwrite').saveAsTable(f'{dsa}.{proj}_pmeds_preprocessed')

# COMMAND ----------


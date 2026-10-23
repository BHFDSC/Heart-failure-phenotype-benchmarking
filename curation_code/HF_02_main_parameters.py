# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # HF_02_main_parameters
# MAGIC
# MAGIC **Project** Heart failure benchmarking assessment
# MAGIC
# MAGIC **Description** This notebook defines a set of parameters, which is loaded in each notebook in the data curation pipeline, so that helper functions and parameters are consistently available.
# MAGIC
# MAGIC **Author(s)** Jadene Lewis (Health Data Science Team, BHF Data Science Centre)
# MAGIC
# MAGIC **Reviewers** ⚠ UNREVIEWED
# MAGIC
# MAGIC **Acknowledgements** Based on CCU004_01-D01-parameters (Tom Bolton, Fionna Chalmers, Anna Stevenson)
# MAGIC
# MAGIC **Notes** This pipeline has an initial production date of 2025-10-13 (`pipeline_production_date` == `2025-10-13`) and the `archived_on` dates used for each dataset correspond to the latest (most recent) batch of data before this date. Should the pipeline and all the notebooks that follow need to be updated and rerun, then this notebook should be rerun directly (before being called by subsequent notebooks) with `pipeline_production_date` updated and `run_all_toggle` switched to True. After this notebook has been rerun the `run_all_toggle` should be reset to False to prevent subsequent notebooks that call this notebook from having to rerun the 'archived_on' section. Rerunning this notebook with the updated `pipeline_production_date` will ensure that the `archived_on` dates used for each dataset are updated with these dates being saved for reference in the collabortion database.
# MAGIC
# MAGIC **Versions** Version 1 as at '2025-10-13'
# MAGIC
# MAGIC **Data Output** 
# MAGIC - **`hf_benchmarking_jl_parameters_df_datasets`**: table of `archived_on` dates for each dataset that can be used consistently throughout the pipeline 

# COMMAND ----------

# MAGIC %md 
# MAGIC # 0. Setup

# COMMAND ----------

run_all_toggle = False # True

# COMMAND ----------

spark.conf.set('spark.sql.legacy.allowCreatingManagedTableUsingNonemptyLocation', 'true')

# COMMAND ----------

# MAGIC %md
# MAGIC # 1. Libraries

# COMMAND ----------

import pyspark.sql.functions as f
import pyspark.sql.types as t
import pandas as pd
import re
import datetime

# COMMAND ----------

# MAGIC %md
# MAGIC # 2. Common Functions

# COMMAND ----------

# MAGIC %run "Workspace/Shared/SHDS/common/functions"

# COMMAND ----------

# MAGIC %md
# MAGIC # 4. Paths and Variables

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.1 Set Project Specific Variables

# COMMAND ----------

# Please set and check the variables below

# -----------------------------------------------------------------------------
# Databases
# -----------------------------------------------------------------------------
db = ''
dbc = f'{db}_collab'
dsa = f''
dss = f''

# -----------------------------------------------------------------------------
# Project
# -----------------------------------------------------------------------------
proj = 'hf_benchmarking_jl' 


# -----------------------------------------------------------------------------
# Dates- 
# -----------------------------------------------------------------------------
study_start_date = '2020-01-01' 
study_end_date   = '2024-12-31' 


# -----------------------------------------------------------------------------
# Pipeline production date
# -----------------------------------------------------------------------------
# date at which pipeline was created and archived_on dates for datasets have been selected based on
pipeline_production_date = '2025-10-13' 


# -----------------------------------------------------------------------------
# Datasets
# -----------------------------------------------------------------------------
# data frame of datasets
datasets = [
  # -----------------------------------------------------------------------------
  # Datasets requested by the project
  # -----------------------------------------------------------------------------  
    ['hes_apc',       dbc, f'hes_apc_all_years_archive',      'PERSON_ID_DEID',                 'EPISTART']
  , ['deaths',        dbc, f'deaths_{db}_archive',            'DEC_CONF_NHS_NUMBER_CLEAN_DEID', 'REG_DATE_OF_DEATH']
  , ['nicor_nhfa_v4',          dbc, f'nicor_hf_{db}_archive',              '1_03_NHS_NUMBER_DEID',          '2_00_DATE_OF_VISIT']
  , ['nicor_nhfa_v5',          dbc, f'nicor_hf_v5_{db}_archive',              '1_03_NHS_NUMBER_DEID',          '2_00_DATE_OF_VISIT']            
  , ['sgss',          dbc, f'sgss_{db}_archive',              'PERSON_ID_DEID',                 'Specimen_Date']
  , ['pillar2',       dbc, f'covid_antigen_testing_pillar2_{db}_archive', 'Person_ID_DEID',  'TestStartDate']
  , ['pillar3',       dbc, f'covid_antibody_testing_pillar3_{db}_archive','Person_ID_DEID',  'TestStartDate']
  , ['vacc',          dbc, f'vaccine_status_{db}_archive',    'PERSON_ID_DEID',                'DATE_AND_TIME'] 
  , ['pmeds',         dbc, f'primary_care_meds_{db}_archive', 'Person_ID_DEID',                 'ProcessingPeriodDate'] 
  # -----------------------------------------------------------------------------
  # Additonal datasets needed for the data curation pipeline for this project
  # -----------------------------------------------------------------------------
  , ['gdppr',         dbc, f'gdppr_{db}_archive',             'NHS_NUMBER_DEID',                'DATE']  
  , ['hes_ae',        dbc, f'hes_ae_all_years_archive',       'PERSON_ID_DEID',                 'ARRIVALDATE']
  , ['hes_op',        dbc, f'hes_op_all_years_archive',       'PERSON_ID_DEID',                 'APPTDATE']
  , ['hes_cc',        dbc, f'hes_cc_all_years_archive',       'PERSON_ID_DEID',                 'CCSTARTDATE'] 
  , ['chess',         dbc, f'chess_{db}_archive',             'PERSON_ID_DEID',                 'InfectionSwabDate']
  , ['sus',           dbc, f'sus_{db}_archive',               'NHS_NUMBER_DEID',                'EPISODE_START_DATE'] 
  
  # -----------------------------------------------------------------------------
  # Datasets not required for this project
  # -----------------------------------------------------------------------------  
#   , ['icnarc',        dbc, f'icnarc_{db}_archive',            'NHS_NUMBER_DEID',               'Date_of_admission_to_your_unit']  
#   , ['minap',         dbc, f'minap_{db}_archive',             'NHS_NUMBER_DEID',               'ARRIVAL_AT_HOSPITAL'] 
# , ['ssnap',         dbc, f'ssnap_{db}_archive',             'Person_ID_DEID',                 'S1FIRSTARRIVALDATETIME']
#   , ['nvra',          dbc, f'nvra_{db}_archive',              'NHS_NUMBER_DEID',               'DATE']  
]

tmp_df_datasets = pd.DataFrame(datasets, columns=['dataset', 'database', 'table', 'id', 'date']).reset_index()
# Let’s discuss how to reset the index in Pandas DataFrame. Often We start with a huge data frame in Pandas and after manipulating/filtering the data frame, we end up with a much smaller data frame. When we look at the smaller data frame, it might still carry the row index of the original data frame. If the original index is numbers, now we have indexes that are not continuous.

if(run_all_toggle):
  print('tmp_df_datasets:\n', tmp_df_datasets.to_string())

tmp_df_datasets.count()


# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.2 Datasets Archived States

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.2.1 Create

# COMMAND ----------

type(tmp_df_datasets)

# COMMAND ----------

display(tmp_df_datasets)

# COMMAND ----------

# DBTITLE 1,All archived states
# for each dataset in tmp_df_datasets, 
#   tabulate all archived_on dates (for information)
#   find the latest (most recent) archived_on date before the pipeline_production_date
#   create a table containing a row with the latest archived_on date and count of the number of records for each dataset
  
# this will not run each time the Parameters notebook is run in annother notebook - will only run if the toggle is switched to True
if(run_all_toggle):

  latest_archived_on = []
  lsoa_1st = []
  for index, row in tmp_df_datasets.iterrows():
    # initial  
    dataset = row['dataset']
    path = row['database'] + '.' + row['table']
    print(index, dataset, path); print()

    # point to table
    tmpd = spark.table(path)

    # tabulate all archived_on dates
    tmpt = tab(tmpd, 'archived_on')
    
    # extract latest (most recent) archived_on date before the pipeline_production_date
    tmpa = (
      tmpd
      .groupBy('archived_on')
      .agg(f.count(f.lit(1)).alias('n'))
      .withColumn('n', f.format_number('n', 0))
      .where(f.col('archived_on') <= pipeline_production_date)
      .orderBy(f.desc('archived_on'))
      .limit(1)
      .withColumn('dataset', f.lit(dataset))
      .select('dataset', 'archived_on', 'n')
      .toPandas()
    )
    tmpaa = tmpa.iloc[0]['archived_on']
    print(tmpaa)

    # get count
    tmpn = tmpd.where(f.col('archived_on') == tmpaa).count()
    tmpa['n'] = tmpn
    print(tmpa.to_string())
    
    # # extract closest archived_on date that comes after study_start_date
    # if(dataset=="gdppr"):
    #   tmpb = (
    #     tmpd
    #     .groupBy('archived_on')
    #     .agg(f.count(f.lit(1)).alias('n'))
    #     .withColumn('n', f.format_number('n', 0))
    #     .where(f.col('archived_on') >= study_start_date)
    #     .orderBy(f.asc('archived_on'))
    #     .limit(1)
    #     .withColumn('dataset', f.lit(dataset))
    #     .select('dataset', 'archived_on', 'n')
    #     .toPandas()
    #   )
      
    #   if(index == 0): lsoa_1st = tmpb
    #   else: lsoa_1st = pd.concat([lsoa_1st, tmpb])
    #   # else: lsoa_1st = lsoa_1st.unionByName(tmpb)
    
    # append results
    if(index == 0): latest_archived_on = tmpa
    else: latest_archived_on = pd.concat([latest_archived_on, tmpa])
    # else: latest_archived_on = latest_archived_on.unionByName(tmpa)
    print()
    


  # check
  print('Latest (most recent) archived_on date before pipeline_production_date')
  print(latest_archived_on.to_string())
  # print('\nClosest (1st) GDPPR archived_on date following study_start_date')
  # print(lsoa_1st.to_string())

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.2.2 Check

# COMMAND ----------

# DBTITLE 1,Latest archived states
# this will not run each time the Parameters notebook is run in annother notebook - will only run if the toggle is switched to True 
# if(run_all_toggle):
#   # check
#   # display(latest_archived_on)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.2.3 Prepare

# COMMAND ----------

# prepare the tables to be saved

# this will not run each time the Parameters notebook is run in annother notebook - will only run if the toggle is switched to True
if(run_all_toggle):
  
  # merge the datasets dataframe with the latest_archived_on
  tmp_df_datasets_sp = spark.createDataFrame(tmp_df_datasets) 
  latest_archived_on_sp = spark.createDataFrame(latest_archived_on) 
  parameters_df_datasets = merge(tmp_df_datasets_sp, latest_archived_on_sp, ['dataset'], validate='1:1', assert_results=['both'], indicator=0).orderBy('index'); print()
  
  # check  
  print(parameters_df_datasets.toPandas().to_string())


# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.2.4 Save

# COMMAND ----------

# save the parameters_df_datasets table
# which we can simply import below when not running all and calling this notebook in subsequent notebooks

# this will not run each time the Parameters notebook is run in annother notebook - will only run if the toggle is switched to True
if(run_all_toggle):
  parameters_df_datasets = (
    parameters_df_datasets
    .withColumn('n', f.format_number('n', 0))
  )
  save_table(df=parameters_df_datasets, out_name=f'{proj}_parameters_df_datasets', save_previous=False)

# COMMAND ----------

# # save the parameters_df_datasets table
# # which we can simply import below when not running all and calling this notebook in subsequent notebooks

# # this will not run each time the Parameters notebook is run in annother notebook - will only run if the toggle is switched to True
# if(run_all_toggle):
#   # save name
#   outName = f'{proj}_parameters_df_datasets'.lower()
#   print(outName)

#   # save previous version for comparison purposes
#   tmpt = spark.sql(f"""SHOW TABLES FROM {dbc}""")\
#     .select('tableName')\
#     .where(f.col('tableName') == outName)\
#     .collect()
#   if(len(tmpt)>0):
#     _datetimenow = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#     outName_pre = f'{outName}_pre{_datetimenow}'.lower()
#     print(outName_pre)
#     spark.table(f'{dbc}.{outName}').write.mode('overwrite').saveAsTable(f'{dbc}.{outName_pre}')
#     spark.sql(f'ALTER TABLE {dbc}.{outName_pre} OWNER TO {dbc}')

#   # save with production date appended
#   parameters_df_datasets.withColumn('pipeline_production_date', f.lit(pipeline_production_date)).write.mode('overwrite').saveAsTable(f'{dbc}.{outName}')
    
#   spark.sql(f'ALTER TABLE {dbc}.{outName} OWNER TO {dbc}')

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.2.5 Import

# COMMAND ----------

# import the parameters_df_datasets table 
# convert to a Pandas dataframe and transform archived_on to a string (to conform to the input that the extract_batch_from_archive function is expecting)

spark.sql(f'REFRESH TABLE {dsa}.{proj}_parameters_df_datasets')
parameters_df_datasets = (
  spark.table(f'{dsa}.{proj}_parameters_df_datasets')
  .orderBy('index')
  .toPandas()
)
parameters_df_datasets['archived_on'] = parameters_df_datasets['archived_on'].astype(str)

# COMMAND ----------

display(parameters_df_datasets)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.3 Curated Data Paths

# COMMAND ----------

# -----------------------------------------------------------------------------
# These are paths to data tables curated in subsequent notebooks that may be
# needed in subsequent notebooks from which they were curated
# -----------------------------------------------------------------------------

# note: the below is largely listed in order of appearance within the pipeline:  

# reference tables
path_ref_bhf_phenotypes  = 'bhf_cvd_covid_uk_byod.bhf_covid_uk_phenotypes_20210127'
path_ref_geog            = f'{dss}.ons_chd_geo_listings'
path_ref_gp_refset       = f'{dss}.gpdata_snomed_refset_full'
path_ref_gdppr_refset    = f'{dss}.gdppr_cluster_refset'
path_ref_icd10           = f'{dss}.icd10_group_chapter_v01'
path_ref_opcs4           = f'{dss}.opcs_codes_v02'
path_ref_imd             = '.hds_cur_lsoa_2011_imd_lookup'
path_ref_region          = '.hds_cur_lsoa_region_lookup'  

# curated tables
path_cur_deaths_sing       = f'{dsa}.{proj}_cur_deaths_sing'
path_cur_deaths_long       = f'{dsa}.{proj}_cur_deaths_long'
path_cur_hes_apc_long      = f'{dsa}.{proj}_cur_hes_apc_long' 

path_cur_lsoa_ruc          = f'{dsa}.{proj}_cur_lsoa_ruc_lookup'
path_cur_lsoa_multisource        = f'{dsa}.{proj}_cur_lsoa_multisource'
path_cur_covid_inf         = f'{dsa}.{proj}_cur_covid_inf'
path_cur_covid_vacc        = f'{dsa}.{proj}_cur_covid_vacc'
path_cur_covid_vacc_first        = f'{dsa}.{proj}_cur_covid_vacc_first' 
path_cur_covid_vacc_qa           = f'{dsa}.{proj}_cur_covid_vacc_qa'
path_cur_covid_vacc_reshaped     = f'{dsa}.{proj}_cur_covid_vacc_reshaped'

# Curated within HF benchmarking
path_gdppr_preprocessed = f"{dsa}.{proj}_gdppr_preprocessed"
path_hes_apc_preprocessed = f"{dsa}.{proj}_hes_apc_preprocessed"
path_hes_apc_procedure_preprocessed = f"{dsa}.{proj}_hes_apc_procedure_preprocessed"
path_hf_icd10_only_codelist  = f'{dsa}.{proj}_hf_icd10_only_codelist'
path_hf_codelist_all = f'{dsa}.{proj}_hf_codelist_all'
path_cur_hes_apc_diag_all_hf_events  = f'{dsa}.{proj}_hes_apc_diag_all_hf_events'
path_cur_gdppr_diag_all_hf_events = f'{dsa}.{proj}_gdppr_all_hf_events'

# HDS curated assets: Demographics table
hds_curated_assets_demographic = f'{dsa}.hds_curated_assets__demographics_2025_07_28' # HDS team key patient characteristics table

# HDS curated assets: Multisource table
hds_curated_assets_lsoa_multisource = f'{dsa}.hds_curated_assets__lsoa_multisource_2025_07_28'

# HDS curated assets: curated deaths (single row) table
hds_curated_assets_deaths_sing = f'{dsa}.hds_curated_assets__deaths_single_2025_07_28'

# HDS curated assets: curated cause of death (long) table
hds_curated_assets_deaths_long = f'{dsa}.hds_curated_assets__deaths_cause_of_death_2025_07_28'

# HDS curated assets: curated HES APC table (long)
hds_curated_assets_hes_apc_long = f'{dsa}.hds_curated_assets__hes_apc_diagnosis_2025_07_28'

# HDS curated assets: curated COVID diagnoses table
hds_curated_assets_covid_positive = f'{dsa}.hds_curated_assets__covid_positive_2025_07_28'

# Patient ID table (with flag column for valid NHS Numbers)
patient_id_table = f'.token_pseudo_id_lookup'

# # temporary tables
path_tmp_skinny_unassembled             = f'{dsa}.{proj}_tmp_kpc_harmonised_1'
path_tmp_skinny_assembled               = f'{dsa}.{proj}_tmp_kpc_selected'
path_tmp_skinny                         = f'{dsa}.{proj}_tmp_skinny'

path_tmp_quality_assurance_hx_1st_wide  = f'{dsa}.{proj}_tmp_quality_assurance_hx_1st_wide'
path_tmp_quality_assurance_hx_1st       = f'{dsa}.{proj}_tmp_quality_assurance_hx_1st'
path_tmp_quality_assurance_qax          = f'{dsa}.{proj}_tmp_quality_assurance_qax'
path_tmp_quality_assurance              = f'{dsa}.{proj}_tmp_quality_assurance'

path_tmp_demographics                   = f'{dsa}.{proj}_tmp_demographics'
path_tmp_lsoa                           = f'{dsa}.{proj}_tmp_lsoa'
path_tmp_lsoa_inc_exc                   = f'{dsa}.{proj}_tmp_lsoa_inc_exc'

path_tmp_test_cases                     = f'{dsa}.{proj}_tmp_test_cases'
path_tmp_test_controls                  = f'{dsa}.{proj}_tmp_test_controls'
path_tmp_test_controls_lsoa             = f'{dsa}.{proj}_tmp_test_controls_lsoa'
path_tmp_test_controls_covid_inf        = f'{dsa}.{proj}_tmp_test_controls_covid_inf'
path_tmp_test_controls_covid_vacc       = f'{dsa}.{proj}_tmp_test_controls_covid_vacc'

path_tmp_cases                          = f'{dsa}.{proj}_tmp_cases'
path_tmp_pcontrols                      = f'{dsa}.{proj}_tmp_pcontrols'

path_tmp_inc_exc_flow                   = f'{dsa}.{proj}_tmp_inc_exc_flow'
path_tmp_inc_exc_flow_cohort            = f'{dsa}.{proj}_tmp_inc_exc_flow_cohort'

path_tmp_hx_nonfatal                    = f'{dsa}.{proj}_tmp_hx_nonfatal'

path_tmp_inc_exc_2_cohort                 = f'{dsa}.{proj}_tmp_inc_exc_2_cohort'
path_tmp_inc_exc_2_flow                   = f'{dsa}.{proj}_tmp_inc_exc_2_flow'


# out tables
path_out_codelist_quality_assurance      = f'{dsa}.{proj}_out_codelist_quality_assurance'
path_out_codelist_covid                  = f'{dsa}.{proj}_out_codelist_covid'
path_out_codelist_covariates             = f'{dsa}.{proj}_out_codelist_covariates'

path_out_hf_cohort                         = f'{dsa}.{proj}_out_hf_cohort'
# path_out_covariates                        = f'{dsa}.{proj}_out_covariates'

# path_out_demographics                      = f'{dsa}.{proj}_out_demographics'
# path_out_lsoa                              = f'{dsa}.{proj}_out_lsoa'

path_out_covid_inf                         = f'{dsa}.{proj}_out_covid_inf'
path_out_covid_vacc                        = f'{dsa}.{proj}_out_covid_vacc'

path_out_outcomes                          = f'{dsa}.{proj}_out_outcomes'

# Controls tables
path_out_controls                          = f'{dsa}.{proj}_out_controls'
path_out_controls_lsoa                     = f'{dsa}.{proj}_out_controls_lsoa'
path_out_controls_covid_inf                = f'{dsa}.{proj}_out_controls_covid_inf' 
path_out_controls_covid_vacc               = f'{dsa}.{proj}_out_controls_covid_vacc'

# COMMAND ----------

# Leftover code

# -----------------------------------------------------------------------------
# out tables 
# -----------------------------------------------------------------------------
# tmp_out_date = '99999999'
# data = [
# #    ['codelist',  tmp_out_date]
# #  , ['cohort',  tmp_out_date]
# #  , ['skinny',    tmp_out_date]  
# #  , ['covariates', tmp_out_date]  
# #  , ['exposures',  tmp_out_date]      
# #  , ['outcomes',   tmp_out_date]  
# #  , ['analysis',   tmp_out_date]    
# ]
# df_out = pd.DataFrame(data, columns = ['dataset', 'out_date'])
# # df_out

# COMMAND ----------

# MAGIC %md
# MAGIC # 5. Summary

# COMMAND ----------

# # print variables, archived_on table, and paths for reference in subsequent notebooks
# print(f'Project:')
# print("  {0:<22}".format('proj') + " = " + f'{proj}')
# print("  {0:<22}".format('cohort') + " = " + f'{cohort}') 
# print(f'')
# print(f'Study dates:')    
# print("  {0:<22}".format('study_start_date') + " = " + f'{study_start_date}') 
# print("  {0:<22}".format('study_end_date') + " = " + f'{study_end_date}') 
# print(f'')
# print(f'Databases:')
# print("  {0:<22}".format('db') + " = " + f'{db}') 
# print("  {0:<22}".format('dbc') + " = " + f'{dbc}') 
# print(f'')
# print(f'Pipeline production date:')    
# print("  {0:<22}".format('pipeline_production_date') + " = " + f'{pipeline_production_date}') 
# print(f'')
# print(f'parameters_df_datasets:')
# print(parameters_df_datasets[['dataset', 'database', 'table', 'archived_on', 'n']].to_string())
# print(f'')
# print(f'Paths:')
# # with pd.option_context('display.max_rows', None, 'display.max_columns', None): 
# #  print(df_paths_raw_data[['dataset', 'database', 'table']])
# # print(f'')
# # print(f'  df_archive')
# # print(df_archive[['dataset', 'database', 'table', 'productionDate']].to_string())
# tmp = vars().copy()
# for var in list(tmp.keys()):
#   if(re.match('^path_.*$', var)):
#     print("  {0:<22}".format(var) + " = " + tmp[var])    
 
  
# #print(f'composite_events:')   
# #for i, c in enumerate(composite_events):
#   #print('  ', i, c, '=', composite_events[c])
# #print(f'')
# # print(f'Out dates:')
# # with pd.option_context('display.max_rows', None, 'display.max_columns', None): 
# #  print(df_paths_raw_data[['dataset', 'database', 'table']])
# # print(f'')
# # print(f'  df_out')
# #print(df_out[['dataset', 'out_date']].to_string())
# #print(f'')

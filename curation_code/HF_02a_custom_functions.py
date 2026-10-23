# Databricks notebook source
# MAGIC %md # CCU087_01-D01b-custom_functions
# MAGIC
# MAGIC This notebook contains a variety of custom functions, some of which have been specifically developed for project CCU085

# COMMAND ----------

# MAGIC %md
# MAGIC # LSOA function
# MAGIC Create a function to get LSOA closest to stroke admission date (to be used in controls_selection: in stage 1, 2, 3)

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window

def join_closest_lsoa(df, lsoa_multisource):
    # Join the dataframes
    joined_df = df.join(
        lsoa_multisource.withColumnRenamed('person_id', 'CONTROL_PERSON_ID'),
        on='CONTROL_PERSON_ID',
        how='left'
    )

    # Calculate the date difference and add England flag
    joined_df = (
      joined_df
      .withColumn('date_diff', f.abs(f.datediff(f.col('CASE_DOE'), f.col('lsoa_multisource_record_date'))))
    )

    # Add England flag
    joined_df = (
      joined_df
      .withColumnRenamed('lsoa_multisource', 'CONTROL_LSOA')
      .withColumnRenamed('in_england', 'ind_control_lsoa_england')
      .withColumn('NON_ENGLAND_CONTROL_LSOA', f.when(f.col('ind_control_lsoa_england') == 0, 1)
              .otherwise(0))
    )

    # Adding a 'lsoa_tie_flag' column- flags where there is a tie by date_diff (two distinct LSOA values for same date_diff). Arbritraily chooses one of the flags
    tmp_ties = (
        joined_df
        .groupBy('CONTROL_PERSON_ID', 'CASE_DOE', 'date_diff')
        .agg(f.countDistinct(f.col('CONTROL_LSOA')).alias('n_distinct_lsoa'))
    )

    joined_df = (
        joined_df
        .join(tmp_ties, ['CONTROL_PERSON_ID', 'CASE_DOE', 'date_diff'], how = 'left')
        .withColumn('lsoa_tie_flag', f.when(f.col('n_distinct_lsoa') > 1, '1')
                    .when(f.col('n_distinct_lsoa') == 1, '0')
                    .otherwise(None)
    ))

    # Select the LSOA closest to stroke admission date: apply an LSOA function (to make more efficient)
    window_spec = Window.partitionBy('CONTROL_PERSON_ID', 'CASE_DOE').orderBy('date_diff', 'CONTROL_LSOA') # random selection where there are multiple LSOA with same date_diff. order by both columns to keep stable

    result_df = (
        joined_df
        .withColumn('row_num', f.row_number().over(window_spec))
        .filter(f.col('row_num') == 1)
        .drop('row_num', 'date_diff')
    )

    # check
    tmpt = tab(joined_df, 'ind_control_lsoa_england')
    tmpt = tab(joined_df, 'ind_lsoa_null')

    return result_df

# COMMAND ----------

# MAGIC %md # LSOA function (pandas)
# MAGIC Create a function to get LSOA closest to stroke admission date (to be used in controls_selection: in stage 1, 2, 3). But using pandas dataframe as input instead of Spark dataframe

# COMMAND ----------

import pandas as pd
import numpy as np

def join_closest_lsoa_p(df, lsoa_multisource):

    # Merge the dataframes
    merged_df = pd.merge(df, lsoa_multisource, on='CONTROL_PERSON_ID', how='left')

    # Calculate the date difference
    merged_df['date_diff'] = abs((merged_df['CASE_DOE'] - merged_df['lsoa_multisource_record_date']).dt.days)

    # Rename columns
    merged_df = merged_df.rename(columns={'lsoa_multisource': 'CONTROL_LSOA', 'in_england': 'ind_control_lsoa_england'})

    # Add England flag
    merged_df['NON_ENGLAND_CONTROL_LSOA'] = np.where(merged_df['ind_control_lsoa_england'] == 0, 1, 0)

    # Sort by columns
    merged_df_sorted = merged_df.sort_values(['CONTROL_PERSON_ID', 'CASE_DOE', 'date_diff', 'CONTROL_LSOA'])

    # Then, create a groupby object
    grouped = merged_df_sorted.groupby(['CONTROL_PERSON_ID', 'CASE_DOE'])

    # Now you can apply window-like operations on this grouped object
    # For example, to get row numbers within each group:
    merged_df_sorted['row_num'] = grouped.cumcount() + 1

    # Take rows where row_num == 1
    result_df = merged_df_sorted[merged_df_sorted['row_num'] == 1]

    # Drop the temporary date_diff column
    result_df = result_df.drop(columns=['date_diff'])

    return result_df

# COMMAND ----------

# cur_lsoa_multisource

# # Preparing lsoa_multisource
# # Reformatting record_date column in the lsoa_multisource table to ensure that all dates are converted from string to date format (consistent)
# lsoa_multisource_ssnap = (
#     lsoa_multisource_ssnap
#     .filter(f.col('lsoa').isNotNull()) # drop null lsoa
#     .filter(f.col('record_date').isNotNull()) # drop null record date 
#     .withColumn('record_date', f.when(f.col('record_date').rlike("^\d{4}-\d{2}-\d{2}$"),  # Matches 'yyyy-MM-dd'
#         f.to_date(f.col('record_date'), "yyyy-MM-dd"))
#                 .when(f.col('record_date').rlike("^\d{8}$"),  # Matches 'yyyyMMdd'
#         f.to_date(f.col('record_date'), "yyyyMMdd")))
#     # indicator
#     .withColumn('ind_record_date', 
#                 f.when(f.col('record_date').rlike("^\d{4}-\d{2}-\d{2}$"), f.lit(1))
#                 # .when(f.col('record_date').rlike("^\d{8}$"), f.lit(2))
#                 .otherwise(f.lit(0)))
#     .withColumnRenamed('person_id', 'PERSON_ID')
#     .withColumnRenamed('lsoa', 'lsoa_multisource')
#     .withColumnRenamed('record_date', 'lsoa_multisource_record_date')          
#     .withColumnRenamed('data_source', 'lsoa_multisource_data_source')
#     .drop('archived_on')
#     )

# COMMAND ----------

# ssnap_id_date
# cur_lsoa_multisource = spark.table() 


# arguments

# df_cohort = ssnap_id_date
# df_lsoa_multisource = cur_lsoa_multisource

# option to broadcast join cohort

# # think about multiple rows per PERSON_ID

# # date = '2020-01-01' , date = DATE

# # join two tables

# window_dense_rank = Window.partitionBy('PERSON_ID', 'DATE').orderBy('abs_datediff', f.desc('date_diff')) 
# window_collect_list = Window.partitionBy('PERSON_ID', 'DATE') 
# window_row_number = Window.partitionBy('PERSON_ID', 'DATE').orderBy('lsoa_multisource') # stable ordering

# tmp = (
#     df_lsoa_multisource
#     .join(f.broadcast(df_cohort), on='PERSON_ID', how='inner')
#     .withColumn('date_diff', f.datediff(f.col('lsoa_multisource_record_date'), f.col('DATE')))
#     .withColumn('date_diff_abs', f.abs(f.datediff(f.col('lsoa_multisource_record_date'), f.col('DATE'))))
#     .withColumn('dense_rank', f.dense_rank().over(window_dense_rank))
#     .where(f.col('dense_rank') == 1)    
#     .drop('dense_rank')
#     .withColumn('lsoa_ties', f.sort_array(f.collect_list(f.col('lsoa_multisource')).over(window_ties)))
#     .withColumn('lsoa_ties_n', f.size(f.col('lsoa_ties'))) # add distinct before count?
#     .withColumn('row_num', f.row_number().over(window_spec))
#     .where(f.col('row_num') == 1)
#     .drop('row_num')
# )





# COMMAND ----------

# MAGIC %md
# MAGIC # Hometime function
# MAGIC Function to calculate hometime pre and post stroke admission (across a specified number of days)

# COMMAND ----------

# TODO: Consider passing a list of days to avoid repeated calls 

def hometime(cohort_df, hes_apc_spells_df, out_name, days, data_coverage_end_date, same_day_spells=0.5, run_checks=True, nrows=30):

  # add docstring etc.
  """
  Author:             Tom Bolton
  Last updated:       30/09/2024
  Contact:            

  Description:
    Returns the hometime calculation for...

  Args:
  - cohort_df (DataFrame): The input cohort dataframe 
  - hes_apc_spells_df: Curated patient spells dataframe (derived from HES APC)
  - out_name: The name of the temporary table to be saved
  - days: The number of days over which hometime should be calculated (use positive or negative values for follow-up or lookback periods, respectively)
  - data_coverage_end_date: The cut-off date for follow-up period (based on HES APC coverage)
  - run_checks: If set to True, this runs a series of checks
  
  Returns:
    DataFrame: The cohort dataframe with additional hometime columns (as per the specified follow-up and lookback periods).

"""
  # input - cohort dataframe - PERSON_ID, DATE, DOD
  #       - hes_apc_spells dataframe = PERSON_ID, spell_start_date, spell_end_date
  #

  print('----------------------------------------------------------------------------------------------')
  print('prepare cohort')
  print('----------------------------------------------------------------------------------------------')
  # add data_coverage_end_date
  cohort_df_1 = (
    cohort_df    
    .orderBy('PERSON_ID', 'DATE', 'DOD')
    .withColumn('data_coverage_end_date', f.to_date(f.lit(data_coverage_end_date)))
  )

  # add period start and end dates relating to the days parameter
  # post event
  if(days > 0):
    print('days > 0\n')
    # start date = DATE (date of event)
    # end date = minimum of [DATE + days, DOD, data coverage end date]
    # if end date less than DATE + days, then create an indicator to show whether this was due to DOD or data coverage end date 
    cohort_df_1 = (
      cohort_df_1  
      .withColumn('period_start_date', f.col('DATE')) 
      .withColumn('period_end_date_max', f.date_add('DATE', days))   
      .withColumn('period_end_date', f.least('period_end_date_max', 'DOD', 'data_coverage_end_date'))
      .withColumn('period_end_date_source', 
        f.when(f.col('period_end_date') < f.col('period_end_date_max'),
          f.when(f.col('period_end_date') == f.col('DOD'), f.lit('DOD'))
          .when(f.col('period_end_date') == f.col('data_coverage_end_date'), f.lit('data_coverage_end_date'))
          .otherwise(f.lit('999999999999999'))
        )
        .otherwise(f.lit('NA'))
      ) 
      .drop('period_end_date_max')   
    )

  # pre event
  elif(days < 0): 
    print('days < 0\n')
    # end date = DATE (date of event)
    # start date = DATE + days
    # note: no need to truncate start date, since this is an adult cohort with historical data coverage, might be needed for other cohorts
    cohort_df_1 = (
      cohort_df_1  
      .withColumn('period_start_date', f.date_add('DATE', days))   
      .withColumn('period_end_date', f.date_add(f.col('DATE'), -1)) 
      .withColumn('period_end_date_source', f.lit('NA'))
    )    

  # add period days
  cohort_df_1 = (
    cohort_df_1
    .withColumn(f'period_days', f.datediff(f.col('period_end_date'), f.col('period_start_date')))
  )

  # checks
  if run_checks:
    count_var(cohort_df_1, 'PERSON_ID'); print()
    count_varlist(cohort_df_1, ['PERSON_ID', 'DATE'])
    print(cohort_df_1.orderBy('PERSON_ID', 'DATE').limit(nrows).toPandas().to_string()); print()


  print('----------------------------------------------------------------------------------------------')
  print('join spells')
  print('----------------------------------------------------------------------------------------------')
  # join hes_apc_spells
  tmp1 = (
    hes_apc_spells
    .join(f.broadcast(cohort_df_1), on='PERSON_ID', how='inner')
  )

  # temp save
  tmp1 = temp_save(df=tmp1, out_name=f'{proj}_tmp_hometime_tmp1'); print() 

  # check
  if run_checks:
    count_var(tmp1, 'PERSON_ID'); print()
    count_varlist(tmp1, ['PERSON_ID', 'DATE'])
    print(tmp1.columns)
    print(tmp1.orderBy('PERSON_ID', 'DATE').limit(nrows).toPandas().to_string()); print()


  print('----------------------------------------------------------------------------------------------')
  print('filter spells')
  print('----------------------------------------------------------------------------------------------')
  tmp2 = (
    tmp1
    .where((f.col('spell_end_date') >= f.col('period_start_date')) & (f.col('spell_start_date') <= f.col('period_end_date')))   
    .withColumn('chk', f.when(f.col('spell_start_date') < f.lag(f.col('spell_end_date'), 1).over(Window.partitionBy('PERSON_ID', 'DATE').orderBy('spell_start_date', 'spell_end_date')), 1)) 
  )

  # temp save
  tmp2 = temp_save(df=tmp2, out_name=f'{proj}_tmp_hometime_tmp2'); print() 

  # check
  if run_checks:
    count_var(tmp2, 'PERSON_ID'); print()
    count_varlist(tmp2, ['PERSON_ID', 'DATE'])
    tmpt = tab(tmp2, 'spell_start_gt_end'); print() # add assert statement (infrequent!)
    tmpt = tab(tmp2, 'chk'); print() # add assert statement
    print(tmp2.orderBy('PERSON_ID', 'DATE').limit(nrows).toPandas().to_string()); print()
    

  print('----------------------------------------------------------------------------------------------')
  print('calculate days in hospital')
  print('----------------------------------------------------------------------------------------------')
  tmp3 = (
    tmp2
    .where(f.col('spell_start_gt_end') != 1)
    .drop('spell_start_gt_end', 'chk', 'data_coverage_end_date')
    .withColumn('spell_start_date_truncated', f.greatest('spell_start_date', 'period_start_date'))
    .withColumn('spell_end_date_truncated', f.least('spell_end_date', 'period_end_date'))    
    .withColumn('hospital_days', f.datediff(f.col('spell_end_date_truncated'), f.col('spell_start_date_truncated')))
    .withColumn('hospital_days', f.when(f.col('hospital_days') == 0, f.lit(same_day_spells)).otherwise(f.col('hospital_days'))) # # # CHECK - moved from below
    # to distinguish between patients who do not have any hospital stay and patients that have a day or less in hospital (eg: spell_start = spell_end; where the length of stay could be anywhere between >0 and 1 day)
  )

  # check
  if run_checks:
    count_var(tmp3, 'PERSON_ID'); print()
    count_varlist(tmp3, ['PERSON_ID', 'DATE'])
    tmpt = tabstat(tmp3, f'hospital_days'); print()
    print(tmp3.orderBy('PERSON_ID', 'DATE').limit(nrows).toPandas().to_string()); print()


  print('----------------------------------------------------------------------------------------------') 
  print('summarise overall days in hospital in period')
  print('----------------------------------------------------------------------------------------------')
  tmp4 = (
    tmp3  
    .groupBy('PERSON_ID', 'DATE', 'DOD', 'period_start_date', 'period_end_date', 'period_end_date_source', 'period_days') 
    .agg(f.sum('hospital_days').alias('period_hospital_days'))
    .withColumn('period_hospital_days', f.least('period_hospital_days', 'period_days')) # check - chained 0.5 days?
  )

    # check
  if run_checks:
    count_var(tmp4, 'PERSON_ID'); print()
    count_varlist(tmp4, ['PERSON_ID', 'DATE'])    
    tmpt = tabstat(tmp4, f'period_hospital_days'); print()
    print(tmp4.orderBy('PERSON_ID', 'DATE').limit(nrows).toPandas().to_string()); print()


  print('----------------------------------------------------------------------------------------------')
  print('Calculate days at home')
  print('----------------------------------------------------------------------------------------------')
  tmp5 = (
    tmp4
    .withColumn(f'period_home_days',
      f.when(f.col('period_start_date') == f.col('period_end_date'), f.lit(0)) # check
      .otherwise(f.col('period_days') - f.col('period_hospital_days')) 
    )
  )

  # check
  if run_checks:
    count_var(tmp5, 'PERSON_ID'); print()
    count_varlist(tmp5, ['PERSON_ID', 'DATE'])
    tmpt = tabstat(tmp5, f'period_home_days'); print()
    print(tmp5.orderBy('PERSON_ID', 'DATE').limit(nrows).toPandas().to_string()); print()


  print('----------------------------------------------------------------------------------------------')
  print('Output')
  print('----------------------------------------------------------------------------------------------')
  tmp6 = (
    tmp5
    .select('PERSON_ID', 'DATE', 'period_home_days')
  )
  
  # temp save
  tmp6 = temp_save(df=tmp6, out_name=f'{proj}_tmp_hometime_tmp6'); print() 

  # merge
  tmp7 = merge(tmp6, cohort_df_1, ['PERSON_ID', 'DATE'], validate='1:1', assert_results=['both', 'right_only'], broadcast_right=1, indicator=0); print() # validate='1:1'

  # add
  tmp7 = (
    tmp7
    .withColumn('period_home_days', f.when(f.col('period_home_days').isNull(), f.col('period_days')).otherwise(f.col('period_home_days')))
  )
  
  # rename cols
  # days_str
  if(days>0): suffix = f'post_{days}d'
  elif(days<0): suffix = f'pre_{days*-1}d'  
  renames = {
    'period_start_date':        f'{suffix}_start_date'
    , 'period_end_date':        f'{suffix}_end_date'
    , 'period_end_date_source': f'{suffix}_end_date_trunc_source'
    , 'period_days':            f'{suffix}_days_available'
    , 'period_home_days':       f'{suffix}_days_at_home'
  }
  tmp7 = rename_columns(tmp7, renames); print()

  # check
  if run_checks:
    count_var(tmp7, 'PERSON_ID'); print()
    count_varlist(tmp7, ['PERSON_ID', 'DATE'])
    tmpt = tabstat(tmp7, f'{suffix}_days_at_home'); print()
    print(tmp7.orderBy('PERSON_ID', 'DATE').limit(nrows).toPandas().to_string()); print()

  # temp save
  tmp7 = temp_save(df=tmp7, out_name=f'{out_name}'); print() 

  # drop
  spark.sql(f"DROP TABLE {dsa}.{proj}_tmp_hometime_tmp1")
  spark.sql(f"DROP TABLE {dsa}.{proj}_tmp_hometime_tmp2")
  spark.sql(f"DROP TABLE {dsa}.{proj}_tmp_hometime_tmp6")

  # return 
  return tmp7


# COMMAND ----------

# MAGIC %md # Washout function
# MAGIC Function (taken from CCU076) to apply a washout period to a dataframe (in the context of CCU085, we applied this to curated COVID data)

# COMMAND ----------

from pyspark.sql import DataFrame

def washout(df: DataFrame, days: int) -> DataFrame:
  
  # person_id: str, name: str, date: str, code: str, 
  
  '''
  Applies washout period
  
  Args:
    df: Spark DataFrame.
    person_id: NOT CODED YET
    name: NOT CODED YET
    date: NOT CODED YET
    code: NOT CODED YET
    days: washout period (e.g., 30 days)   
    
  Example usage:
    washout(df, days = 30)
    >> 
    
  Notes:
    ...
  '''
  
  print('=================================================================================')
  print('washout')
  print('=================================================================================')  
  
  print('Note: This function currently uses PERSON_ID, name, DATE, CODE - arguments will be added at a later date to assign these')
  
  # check columns are not in df - as these will be overwritten and dropped
  common_cols = [col for col in df.columns if col in ['_rownum_DATE', '_rownum', '_diff', '_diff_cumsum', '_diff_cumsum_flag']]
  assert len(common_cols) == 0, f'common_cols = {common_cols}'
  
  # define windows to calculate row numbers and date differences with patient/outcome record sets
  # window for duplicate DATE
  _win_DATE = Window\
    .partitionBy('PERSON_ID', 'name', 'DATE')\
    .orderBy('CODE')
  # main window (with stable ordering for duplicate DATE)
  _win = Window\
    .partitionBy('PERSON_ID', 'name')\
    .orderBy('DATE', '_rownum_DATE')

  # initialise
  _df_master = []
  _df_washedout = []
  _df_working = df
  _counter_working = _df_working.count()
  print(f'{_counter_working:,} events in total initially'); print()

  # iterate through
  i = 0
  while _counter_working > 0:
    i += 1
    
    print('---------------------------------------------------------------------------------')
    print(f'iteration = {i}') 
    print('---------------------------------------------------------------------------------')
    # calculate row numbers, differences and flag
    _df_working = (
      _df_working
      .withColumn('_rownum_DATE', f.row_number().over(_win_DATE))
      .withColumn('_rownum', f.row_number().over(_win))
      .withColumn('_diff', f.datediff(f.col('DATE'), f.lag(f.col('DATE'), 1).over(_win)))
      .withColumn('_diff_cumsum', f.sum(f.col('_diff')).over(_win))
      .withColumn('_diff_cumsum_flag', f.when(f.col('_rownum') == 1, f.lit('0_first')).when(f.col('_diff_cumsum') > days, '2_gt_washout').otherwise('1_le_washout'))
    )

    # calculate maximum row number
    _rownum_max = _df_working.agg(f.max(f.col('_rownum'))).collect()[0][0]

    # print number of records and maximum row number for this iteration
    # the latter provides an idea of expected number of subsequent iterations
    print(f'{_counter_working:,} events in total (_rownum_max per person_id and name = {_rownum_max})'); print()

    # tabulate of records identified as within washout by name
    tmpt = tab(_df_working, 'name', '_diff_cumsum_flag'); print()

    # output first row for this iteration to the master dataframe
    _df_out = (
      _df_working
      .where(f.col('_rownum') == 1)
      .drop('_rownum_DATE', '_rownum', '_diff', '_diff_cumsum', '_diff_cumsum_flag')
    )
    _counter_firstevents = _df_out.count()
    if(i == 1): 
      _df_master = _df_out
    else:
      _df_master = (
        _df_master
        .unionByName(_df_out)
      )

    # remove the first row and rows within the washout period from the working dataframe (for the next iteration)
    # remove the first row
    _df_working = (
      _df_working
      .where(f.col('_rownum') != 1)
    )
    print(f'{_counter_firstevents:,} events stored for being the first event per person_id and name of iteration {i}')

    # count the rows within the washout period before removing below
    _df_washedout = (
      _df_working
      .where(f.col('_diff_cumsum') <= days)
    )
    _counter_washedout = _df_washedout.count()
    print(f'{_counter_washedout:,} events excluded for being within {days} days of the first event per person_id and name of iteration {i}'); print()

    # remove the rows within the washout period
    _df_working = (
      _df_working\
      .where(f.col('_diff_cumsum') > days)
    )
    _counter_working = _df_working.count()


  print('---------------------------------------------------------------------------------')
  print(f'check') 
  print('---------------------------------------------------------------------------------')      
  # check that differences between events are now all greater than the washout period
  
  # calculate row numbers, differences  
  _df_tmp = (
    _df_master
    .withColumn('_rownum_DATE', f.row_number().over(_win_DATE))
    .withColumn('_rownum', f.row_number().over(_win))
    .withColumn('_diff', f.datediff(f.col('DATE'), f.lag(f.col('DATE'), 1).over(_win)))
  )
  
  # check
  tmpt = tabstat(_df_tmp, '_diff', byvar='name'); print()
  _diff_min = _df_tmp.agg(f.min(f.col('_diff'))).collect()[0][0]
  assert _diff_min > days, '# # # # something went wrong - differences remain that are still less than the washout period # # # #'  
  print(f'differences between events per person_id and name > {days} days is satisfied')
  
  return _df_master

# COMMAND ----------

# MAGIC %md # Save table
# MAGIC Function to save tables to the dsa database

# COMMAND ----------

# def save_table(df, out_name:str, save_previous=True, data_base:str=f''):
  
#   # assert that df is a dataframe
#   assert isinstance(df, f.DataFrame), 'df must be of type dataframe' #isinstance(df, pd.DataFrame) | 
#   # if a pandas df then convert to spark
#   #if(isinstance(df, pd.DataFrame)):
#     #df = (spark.createDataFrame(df))
  
#   # save name check
#   if(any(char.isupper() for char in out_name)): 
#     print(f'Warning: {out_name} converted to lowercase for saving')
#     out_name = out_name.lower()
#     print('out_name: ' + out_name)
#     print('')
  
#   # df name
#   df_name = [x for x in globals() if globals()[x] is df][0]
  
#   # ------------------------------------------------------------------------------------------------
#   # save previous version for comparison purposes
#   if(save_previous):
#     tmpt = (
#       spark.sql(f"""SHOW TABLES FROM {data_base}""")
#       .select('tableName')
#       .where(f.col('tableName') == out_name)
#       .collect()
#     )
#     if(len(tmpt)>0):
#       # save with production date appended
#       _datetimenow = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#       out_name_pre = f'{out_name}_pre{_datetimenow}'.lower()
#       print(f'saving (previous version):')
#       print(f'  {out_name}')
#       print(f'  as')
#       print(f'  {out_name_pre}')
#       spark.table(f'{data_base}.{out_name}').write.mode('overwrite').saveAsTable(f'{data_base}.{out_name_pre}')
#       #spark.sql(f'ALTER TABLE {data_base}.{out_name_pre} OWNER TO {data_base}')
#       print('saved')
#       print('') 
#     else:
#       print(f'Warning: no previous version of {out_name} found')
#       print('')
#   # ------------------------------------------------------------------------------------------------  
  
#   # save new version
#   print(f'saving:')
#   print(f'  {df_name}')
#   print(f'  as')
#   print(f'  {out_name}')
#   df.write.mode('overwrite').option("overwriteSchema", "True").saveAsTable(f'{data_base}.{out_name}')
#   #spark.sql(f'ALTER TABLE {data_base}.{out_name} OWNER TO {data_base}')
#   print('saved')

# COMMAND ----------

# MAGIC %md # Extract batch from archive
# MAGIC Function to select the most recent batch of a dataset 

# COMMAND ----------

# Updated function that compares the number of rows expected (the number that were found when running the parameters notebook in full) against the number of rows observed (the number that were found when extracting the data from the archive in a subsequent notebook). This would alert us to the number of rows being changed in the archive tables, which the data wranglers control.

# function to extract the batch corresponding to the pre-defined archived_on date - will be used in subsequent notebooks

from pyspark.sql import DataFrame
def extract_batch_from_archive(_df_datasets: DataFrame, _dataset: str):
  
  # get row from df_archive_tables corresponding to the specified dataset
  _row = _df_datasets[_df_datasets['dataset'] == _dataset]
  
  # check one row only
  assert _row.shape[0] != 0, f"dataset = {_dataset} not found in _df_datasets (datasets = {_df_datasets['dataset'].tolist()})"
  assert _row.shape[0] == 1, f"dataset = {_dataset} has >1 row in _df_datasets"
  
  # create path and extract archived on
  _row = _row.iloc[0]
  _path = _row['database'] + '.' + _row['table']  
  _archived_on = _row['archived_on']
  _n_rows_expected = _row['n']  
  print(_path + ' (archived_on = ' + _archived_on + ', n_rows_expected = ' + _n_rows_expected + ')')
  
  # check path exists # commented out for runtime
#   _tmp_exists = spark.sql(f"SHOW TABLES FROM {_row['database']}")\
#     .where(f.col('tableName') == _row['table'])\
#     .count()
#   assert _tmp_exists == 1, f"path = {_path} not found"

  # extract batch
  _tmp = spark.table(_path)\
    .where(f.col('archived_on') == _archived_on)  
  
  # check number of records returned
  _n_rows_observed = _tmp.count()
  print(f'  n_rows_observed = {_n_rows_observed:,}')
  assert _n_rows_observed > 0, f"_n_rows_observed == 0"
  assert f'{_n_rows_observed:,}' == _n_rows_expected, f"_n_rows_observed != _n_rows_expected ({_n_rows_observed:,} != {_n_rows_expected})"

  # return dataframe
  return _tmp

# COMMAND ----------

# MAGIC %md # Union list 
# MAGIC Function to join multiple Python lists (and remove duplicate values)

# COMMAND ----------

def union_list(spark, *lists):
    # Create RDDs from the input lists
    rdds = [spark.sparkContext.parallelize(lst) for lst in lists]
    
    # Union all RDDs
    union_rdd = rdds[0].union(*rdds[1:])
    
    # Remove duplicates
    distinct_rdd = union_rdd.distinct()
    
    # Collect the result back to a list
    result = distinct_rdd.collect()
    
    return result
# Load libraries
library(tidyverse)
library(tableone)
library(gt) 
library(webshot2)
library(dplyr)
library(pagedown)
library(gtExtras)
library(purrr)

# Set working directory
setwd(r"(C:\Users\JadeneLewis\OneDrive - Health Data Research UK\Documents\Heart failure phenotyping work\Analysis\Input)")

# Load data
demo_summary_ever_cases <- read.csv(r"(hf_benchmarking_analysis_demo_summaries_ever_cases_rounded_20261022.csv)")
demo_summary_study_period_cases <- read.csv(r"(hf_benchmarking_analysis_demo_summaries_study_period_cases_rounded_20261022.csv)") # updated to contain revised demo summaries (With correct values of IMD)
iqr_age_ever_cases <- read.csv(r"(hf_cohort_ever_iqr_age.csv)")
iqr_age_study_period <- read.csv(r"(hf_cohort_study_period_iqr_age.csv)")
median_age_study_period <- read.csv(r"(hf_cohort_study_period_median_age.csv)")
median_age_ever_cases <- read.csv(r"(hf_cohort_ever_median_age.csv)")
total_records_ever_cases <- read.csv(r"(hf_cohort_ever_total_record_count.csv)")
distinct_patient_count_ever_cases <- read.csv(r"(hf_cohort_ever_total_patient_counts.csv)")
total_records_study_period <- read.csv(r"(hf_cohort_study_period_total_record_count.csv)")
distinct_patient_count_ever_cases <- read.csv(r"(hf_cohort_study_period_total_patient_count.csv)")

# Reformat IQR and median age dataframes
iqr_age_ever_cases <- iqr_age_ever_cases %>%
  mutate(Demographic = "Age (IQR)",
         value = as.character(IQR)) %>%
  select(name, Demographic, value) %>%
  pivot_wider(names_from = name, values_from = value)

median_age_ever_cases <- median_age_ever_cases %>%
  mutate(Demographic = "Age (Median)",
         value = as.character(median.study_start_age.)) %>%
  select(name, Demographic, value) %>%
  pivot_wider(names_from = name, values_from = value)

iqr_age_study_period <- iqr_age_study_period %>%
  mutate(Demographic = "Age (IQR)",
         value = as.character(IQR)) %>%
  select(name, Demographic, value) %>%
  pivot_wider(names_from = name, values_from = value)

median_age_study_period <- median_age_study_period %>%
  mutate(Demographic = "Age (Median)",
         value = as.character(median.study_start_age.)) %>%
  select(name, Demographic, value) %>%
  pivot_wider(names_from = name, values_from = value)

# # # EVER CASE # # #
demo_summary_ever_cases <- demo_summary_ever_cases %>%
  mutate(value = sprintf("%s (%.2f%%)", count_rounded, percentage_rounded))

wide_table <- demo_summary_ever_cases %>%
  mutate(
    category_type = case_when(
      category_type == "region" ~ "Region",
      category_type == "sex" ~ "Sex",
      category_type == "ethnicity" ~ "Ethnicity",
      category_type == "age" ~ "Age",
      TRUE ~ category_type
    ),
    category = case_when(
      category_type == "Sex" & category == "1" ~ "Male",
      category_type == "Sex" & category == "2" ~ "Female",
      TRUE ~ category
    )
  ) %>%
  unite("Demographic", category_type, category, sep = " - ") %>%
  select(Demographic, name, value, percentage_rounded) %>%
  pivot_wider(names_from = name, values_from = c(value, percentage_rounded)) %>%
  arrange(Demographic)

prepare_age_df <- function(df, reference_df) {
  df <- df %>% rename_with(~ paste0("value_", .x), .cols = -matches("Demographic"))
  pct_cols <- grep("^percentage_rounded_", names(reference_df), value = TRUE)
  for (col in pct_cols) if (!col %in% names(df)) df[[col]] <- 0
  df
}

iqr_age_ever_cases   <- prepare_age_df(iqr_age_ever_cases, wide_table)
median_age_ever_cases <- prepare_age_df(median_age_ever_cases, wide_table)

wide_table <- bind_rows(wide_table, iqr_age_ever_cases, median_age_ever_cases) %>%
  arrange(Demographic)

colnames(wide_table) <- gsub("^value_", "", colnames(wide_table))
colnames(wide_table) <- gsub("^percentage_rounded_", "pct_", colnames(wide_table))

all_cols <- names(wide_table)
value_cols <- setdiff(all_cols[!startsWith(all_cols, "pct_")], "Demographic")
pct_cols   <- grep("^pct_", all_cols, value = TRUE)

chunks <- split(value_cols, cut(seq_along(value_cols), 3, labels = FALSE))
pct_chunks <- split(pct_cols,  cut(seq_along(pct_cols),  3, labels = FALSE))

# --- Build gt tables (EVER CASES) ---
# --- Build gt tables (EVER CASES) ---
gt_tables <- lapply(seq_along(chunks), function(i) {
  value_set <- chunks[[i]]
  pct_set   <- pct_chunks[[i]]
  
  tbl <- wide_table %>%
    select(all_of(c("Demographic", value_set, pct_set)))
  
  # --- Format numeric counts with commas ONLY when a percentage is present ---
  library(purrr)
  
  format_value_cell <- function(col) {
    map_chr(col, function(x) {
      if (!str_detect(x, "\\(")) return(x)
      num_raw <- str_extract(x, "^[0-9,]+") %>% str_replace_all(",", "")
      pct     <- str_extract(x, "\\(.*\\)")
      num_val <- suppressWarnings(as.numeric(num_raw))
      formatted_num <- if (!is.na(num_val) && num_val >= 1000)
        format(num_val, big.mark = ",", scientific = FALSE)
      else num_raw
      paste0(formatted_num, " ", pct)
    })
  }
  
  tbl <- tbl %>%
    mutate(across(all_of(value_set), format_value_cell))
  
  gt_obj <- tbl %>%
    select(all_of(c("Demographic", value_set))) %>%
    gt() %>%
    tab_header(
      title = "Phenotype × Demographic Summary (Ever Cases)",
      subtitle = "Counts and percentages for each phenotype and demographic category"
    ) %>%
    tab_options(
      table.font.size = px(14),
      data_row.padding = px(4),
      heading.align = "left"
    ) %>%
    opt_table_outline() %>%
    opt_row_striping()
  
  # --- Apply diverging blue → white → red gradient centred on row mean ---
  # Below row average = blue, at average = white, above average = red
  # Gradient driven by count values, not percentages
  count_data <- tbl %>% select(all_of(value_set))
  count_mat <- count_data %>%
    mutate(across(everything(),
                  ~ suppressWarnings(as.numeric(gsub(",", "", str_extract(.x, "^[0-9,]+")))))) %>%
    as.matrix()
  
  n_rows  <- nrow(count_mat)
  div_pal <- colorRampPalette(c("#08519c", "#ffffff", "#a50f15"))(201)
  
  for (r in seq_len(n_rows)) {
    row_vals <- count_mat[r, ]
    if (length(row_vals) == 0 || all(is.na(row_vals))) next
    
    row_mean   <- mean(row_vals, na.rm = TRUE)
    deviations <- row_vals - row_mean
    max_dev    <- max(abs(deviations), na.rm = TRUE)
    # Skip rows with no variation
    if (max_dev == 0) next
    
    # Scale deviations symmetrically: -max_dev → 1 (blue), 0 → 101 (white), +max_dev → 201 (red)
    scaled_idx <- scales::rescale(deviations, to = c(1, 201), from = c(-max_dev, max_dev))
    
    for (j in seq_along(value_set)) {
      si <- scaled_idx[j]
      if (is.na(si) || is.nan(si)) next
      col <- div_pal[max(1L, min(201L, round(si)))]
      if (!is.na(col) && nzchar(col)) {
        gt_obj <- gt_obj %>%
          tab_style(
            style = cell_fill(color = col),
            locations = cells_body(columns = all_of(value_set[j]), rows = r)
          )
      }
    }
  }
  
  # --- Colour Age (IQR) and Age (Median) rows from their numeric values ---
  age_rows <- which(startsWith(tbl$Demographic, "Age ("))
  if (length(age_rows) > 0) {
    for (r in age_rows) {
      row_vals <- tbl[r, value_set] |>
        unlist() |>
        (\(x) suppressWarnings(as.numeric(gsub(",", "", x))))()
      row_mean   <- mean(row_vals, na.rm = TRUE)
      deviations <- row_vals - row_mean
      max_dev    <- max(abs(deviations), na.rm = TRUE)
      if (max_dev == 0) next
      scaled_idx <- scales::rescale(deviations, to = c(1, 201), from = c(-max_dev, max_dev))
      for (j in seq_along(value_set)) {
        si <- scaled_idx[j]
        if (is.na(si) || is.nan(si)) next
        col <- div_pal[max(1L, min(201L, round(si)))]
        if (!is.na(col) && nzchar(col)) {
          gt_obj <- gt_obj %>%
            tab_style(
              style = cell_fill(color = col),
              locations = cells_body(columns = all_of(value_set[j]), rows = r)
            )
        }
      }
    }
  }
  
  gt_obj
})

# Save PNGs
for (i in seq_along(gt_tables)) {
  gtsave(
    data = gt_tables[[i]],
    filename = paste0("phenotype_table_demographic_summaries_ever_cases_part", i, "_20261022.png"),
    vwidth = 6000, vheight = 3500, zoom = 1
  )
}


# # # STUDY PERIOD # # #
demo_summary_study_period_cases <- demo_summary_study_period_cases %>%
  mutate(value = sprintf("%s (%.2f%%)", count_rounded, percentage_rounded))

wide_table <- demo_summary_study_period_cases %>%
  mutate(
    category_type = case_when(
      category_type == "region" ~ "Region",
      category_type == "sex" ~ "Sex",
      category_type == "ethnicity" ~ "Ethnicity",
      category_type == "age" ~ "Age",
      TRUE ~ category_type
    ),
    category = case_when(
      category_type == "Sex" & category == "1" ~ "Male",
      category_type == "Sex" & category == "2" ~ "Female",
      TRUE ~ category
    )
  ) %>%
  unite("Demographic", category_type, category, sep = " - ") %>%
  select(Demographic, name, value, percentage_rounded) %>%
  pivot_wider(names_from = name, values_from = c(value, percentage_rounded)) %>%
  arrange(Demographic)

iqr_age_study_period   <- prepare_age_df(iqr_age_study_period, wide_table)
median_age_study_period <- prepare_age_df(median_age_study_period, wide_table)

wide_table <- bind_rows(wide_table, iqr_age_study_period, median_age_study_period) %>%
  arrange(Demographic)

colnames(wide_table) <- gsub("^value_", "", colnames(wide_table))
colnames(wide_table) <- gsub("^percentage_rounded_", "pct_", colnames(wide_table))

all_cols <- names(wide_table)
value_cols <- setdiff(all_cols[!startsWith(all_cols, "pct_")], "Demographic")
pct_cols   <- grep("^pct_", all_cols, value = TRUE)

chunks <- split(value_cols, cut(seq_along(value_cols), 3, labels = FALSE))
pct_chunks <- split(pct_cols,  cut(seq_along(pct_cols),  3, labels = FALSE))

# --- Build gt tables (STUDY PERIOD) ---
gt_tables <- lapply(seq_along(chunks), function(i) {
  value_set <- chunks[[i]]
  pct_set   <- pct_chunks[[i]]
  
  tbl <- wide_table %>% select(all_of(c("Demographic", value_set, pct_set)))
  
  # --- Format numeric counts with commas (only if >= 1000) ---
  # --- Format numeric counts with commas ONLY when a percentage is present ---
  library(purrr)
  
  format_value_cell <- function(col) {
    map_chr(col, function(x) {
      
      # If no percentage present → return unchanged
      if (!str_detect(x, "\\(")) {
        return(x)
      }
      
      # Extract numeric part before " (xx%)"
      num_raw <- str_extract(x, "^[0-9,]+") %>% str_replace_all(",", "")
      pct     <- str_extract(x, "\\(.*\\)")  # keep "(xx%)"
      
      num_val <- suppressWarnings(as.numeric(num_raw))
      
      # Format only if >= 1000
      if (!is.na(num_val) && num_val >= 1000) {
        formatted_num <- format(num_val, big.mark = ",", scientific = FALSE)
      } else {
        formatted_num <- num_raw
      }
      
      paste0(formatted_num, " ", pct)
    })
  }
  
  tbl <- tbl %>%
    mutate(across(all_of(value_set), format_value_cell))
  
  gt_obj <- tbl %>%
    select(all_of(c("Demographic", value_set))) %>%
    gt() %>%
    tab_header(
      title = "Phenotype × Demographic Summary (Study Period Cases)",
      subtitle = "Counts and percentages for each phenotype and demographic category"
    ) %>%
    tab_options(
      table.font.size = px(14),
      data_row.padding = px(4),
      heading.align = "left"
    ) %>%
    opt_table_outline() %>%
    opt_row_striping()
  
  # --- Apply diverging blue → white → red gradient centred on row mean ---
  # Below row average = blue, at average = white, above average = red
  # Gradient driven by count values, not percentages
  count_data <- tbl %>% select(all_of(value_set))
  count_mat <- count_data %>%
    mutate(across(everything(),
                  ~ suppressWarnings(as.numeric(gsub(",", "", str_extract(.x, "^[0-9,]+")))))) %>%
    as.matrix()
  
  n_rows  <- nrow(count_mat)
  div_pal <- colorRampPalette(c("#08519c", "#ffffff", "#a50f15"))(201)
  
  for (r in seq_len(n_rows)) {
    row_vals <- count_mat[r, ]
    if (length(row_vals) == 0 || all(is.na(row_vals))) next
    
    row_mean   <- mean(row_vals, na.rm = TRUE)
    deviations <- row_vals - row_mean
    max_dev    <- max(abs(deviations), na.rm = TRUE)
    # Skip rows with no variation
    if (max_dev == 0) next
    
    # Scale deviations symmetrically: -max_dev → 1 (blue), 0 → 101 (white), +max_dev → 201 (red)
    scaled_idx <- scales::rescale(deviations, to = c(1, 201), from = c(-max_dev, max_dev))
    
    for (j in seq_along(value_set)) {
      si <- scaled_idx[j]
      if (is.na(si) || is.nan(si)) next
      col <- div_pal[max(1L, min(201L, round(si)))]
      if (!is.na(col) && nzchar(col)) {
        gt_obj <- gt_obj %>%
          tab_style(
            style = cell_fill(color = col),
            locations = cells_body(columns = value_set[j], rows = r)
          )
      }
    }
  }
  
  # --- Colour Age (IQR) and Age (Median) rows from their numeric values ---
  age_rows <- which(startsWith(tbl$Demographic, "Age ("))
  if (length(age_rows) > 0) {
    for (r in age_rows) {
      row_vals <- tbl[r, value_set] |>
        unlist() |>
        (\(x) suppressWarnings(as.numeric(gsub(",", "", x))))()
      row_mean   <- mean(row_vals, na.rm = TRUE)
      deviations <- row_vals - row_mean
      max_dev    <- max(abs(deviations), na.rm = TRUE)
      if (max_dev == 0) next
      scaled_idx <- scales::rescale(deviations, to = c(1, 201), from = c(-max_dev, max_dev))
      for (j in seq_along(value_set)) {
        si <- scaled_idx[j]
        if (is.na(si) || is.nan(si)) next
        col <- div_pal[max(1L, min(201L, round(si)))]
        if (!is.na(col) && nzchar(col)) {
          gt_obj <- gt_obj %>%
            tab_style(
              style = cell_fill(color = col),
              locations = cells_body(columns = value_set[j], rows = r)
            )
        }
      }
    }
  }
  
  gt_obj
})

# Save PNGs
for (i in seq_along(gt_tables)) {
  gtsave(
    data = gt_tables[[i]],
    filename = paste0("phenotype_table_demographic_summaries_study_period_part", i, "_20261022.png"),
    vwidth = 6000, vheight = 3500, zoom = 1
  )
}

## UPSET PLOT
upset_plot <- read.csv(r"(upset_plot_20260702.csv)")
upset_plot <- upset_plot %>%
  mutate(.combo_id = row_number())


install.packages("UpSetR")
library(UpSetR)

install.packages("ComplexUpset")
library(ComplexUpset)
library(ggplot2)

df_long <- upset_plot %>%
  mutate(.combo_id = row_number()) %>%
  separate_rows(combination, sep = "\\|") %>%
  mutate(combination = str_trim(combination)) %>%
  filter(combination != "") %>%
  extract(
    combination,
    into = c("pheno", "value"),
    regex = "^(.*)_(0|1)$"
  ) %>%
  mutate(value = as.integer(value)) %>%
  drop_na(pheno, value)

df_sets <- df_long %>%
  group_by(.combo_id, pheno) %>%
  summarise(value = max(value), .groups = "drop") %>%
  pivot_wider(
    names_from = pheno,
    values_from = value,
    values_fill = 0
  ) %>%
  left_join(upset_plot, by = ".combo_id") 

df_sets_clean <- df_sets %>%
  mutate(across(starts_with("pheno_"), ~ .x == 1))

# df_sets_top <- df_sets_clean %>%
#   slice_max(order_by = count_rounded, n = 30) %>%
#   arrange(desc(count_rounded))

df_expanded <- df_sets_clean %>%
  uncount(count_rounded, .remove = FALSE)

## Plot distribution in order to determine the min_size to use in upset plot
ggplot(df_expanded, aes(count_rounded)) +
  geom_histogram(bins = 50) +
  scale_x_log10() +
  labs(title = "Distribution of intersection sizes")

# Elbow method (cumulative coverage plot): Look for the elbow — the point where adding more intersections barely increases total coverage.
# That elbow is your minimum size.
df_elbow <- df_sets_clean %>%
  arrange(desc(count_rounded)) %>%
  mutate(
    cum_prop = cumsum(count_rounded) / sum(count_rounded),
    rank = row_number()
  )

ggplot(df_elbow, aes(rank, cum_prop)) +
  geom_line() +
  labs(
    title = "Cumulative coverage of intersections (sorted)",
    x = "Intersection rank (largest → smallest)",
    y = "Cumulative proportion"
  )

summary(df_sets_clean$count_rounded)

df_elbow %>% slice(20) %>% pull(count_rounded) 

df_sets_clean_filtered <- df_sets_clean %>% filter(count_rounded >= 34275)

## PLOTTING UPSET PLOT
set_cols <- setdiff(names(df_sets_clean_filtered), c("count_rounded", ".combo_id", "combination"))

upset(
  df_sets_clean_filtered,
  set_cols,
  name='phenotype',
  base_annotations = list(
    "Intersection size" =
      (
        intersection_size(
          mapping = aes(y = count_rounded),
          text = list(size = 0)  # hide default labels
        )
        +
          geom_text(
            aes(label = count_rounded),
            vjust = -0.3,
            size  = 3
          )
      )
  ),
  wrap = TRUE,
  sort_sets = 'descending',
  sort_intersections = 'descending' #,
  #set_sizes = TRUE 
)



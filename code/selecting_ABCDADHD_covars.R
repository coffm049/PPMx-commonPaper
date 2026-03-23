# interactively, with
# srun -N 1 --mem 80g --time=1:00:00 -p interactive --pty bash
# module load R/4.4.0-openblas-rocky8
# test <- readRDS("/panfs/jay/groups/6/faird/shah0726/ABCD/nda.Rds") %>% names()
# test[grep("zyg", test, ignore.case = T)]
# test[grep("twin", test, ignore.case = T)]
# test[grep("gen", test, ignore.case = T)]
# test[grep("rel_relation", test, ignore.case = T)]

library(tidyverse)
df <- readRDS("/panfs/jay/groups/6/faird/shah0726/ABCD/nda.Rds")
nn <- names(df)
nn[grepl("bis", nn)]

df %>%
  select(
    1:3, rel_family_id, abcd_site, age, female, household.income, high.educ, race_ethnicity, eventname,
    medhx_9a_anesthesia_p_l, medhx_9b_anesthesia_times_p_l, medhx_9c_anesthesia_age_p_l,
    medhx_ss_9b_anesthesia_times_p_l,
    medhx_ss_9b_anesthesia_times_p, medhx_9a_anesthesia_p, medhx_9b_anesthesia_times_p, medhx_9c_anesthesia_age_p,
    mri_info_manufacturer,
    #interview_age.baseline_year_1_arm_1.x,
    # sex.baseline_year_1_arm_1.x,
    contains("bisbas"),
    contains("upps"),
    contains("nihtbx"),
    contains("ADHD"),

    # contains("zyg"), contains("twin"),
    rel_relationship,
    contains("genetic_zygosity_status_1")
  ) %>% # select mri machine manu
  rename(c(IID = subjectid, FID = rel_family_id)) %>%
  filter(grepl("baseline", eventname)) %>%
  relocate(FID, IID, .before = 1) %>%
  select(-c(src_subject_id, eventname)) %>%
  write_csv("~/software/ProductPartitionModels.jl/ABCD-adhd/ADHDCovars.tsv")





# Faceted figure of the distribution of the modeled covariates in the real
# ABCD data (FRF-total analysis), good enough for a supplement.
#
# Run on HPC (from analyses/):
#   module load R/4.4.0-openblas-rocky8
#   Rscript covariateDist-ABCD.R
#
# Output: output/covariateDist.png

library(tidyverse)

# cleaned + standardized covariates (z-scores) for the FRF-total model
modelVars <- c("age", "female", "uPosUrg", "uLplanning", "uLpers",
               "uNegUrg", "bbRR", "bbFS", "bbSum")

fullDf <- read_csv("output/sampledDF.csv", show_col_types = FALSE)

covLabels <- c(
  age       = "Age",
  female    = "Female (0/1)",
  uPosUrg   = "Positive urgency",
  uLplanning = "Lack of planning",
  uLpers    = "Lack of perseverance",
  uNegUrg   = "Negative urgency",
  bbRR      = "BAS reward responsiveness",
  bbFS      = "BAS fun seeking",
  bbSum     = "BIS total"
)

longDf <- fullDf %>%
  select(all_of(modelVars)) %>%
  pivot_longer(everything(), names_to = "covariate", values_to = "value") %>%
  mutate(covariate = factor(covariate, levels = modelVars,
                            labels = covLabels[modelVars]))

p <- longDf %>%
  ggplot(aes(x = value)) +
  geom_histogram(bins = 40, fill = "#4DBBD5", color = "white", linewidth = 0.3) +
  facet_wrap(~covariate, scales = "free", ncol = 3) +
  labs(x = "Standardized value", y = "Count") +
  theme_minimal(base_size = 11) +
  theme(
    strip.text = element_text(size = 10),
    axis.title.x = element_text(margin = margin(t = 8)),
    panel.grid.minor = element_blank()
  )

ggsave("output/covariateDist.png", plot = p, width = 9, height = 7, dpi = 300)
print("wrote output/covariateDist.png")
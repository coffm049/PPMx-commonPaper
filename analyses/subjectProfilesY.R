# Subject-level predicted outcome (y) by SALSO subset.
#
# Source : ppmx-analysis/analyses/output/openTotal/subjectProfilesK5-v2/postPred.csv
#          columns: IID, salso, nTotal_pred
# Output : figures/subject_profiles.{png,pdf}
#
# DESIGN NOTES
#   * Single panel. The earlier version wrapped two panels (raw scale and robustly
#     standardised) side by side, but the standardised panel is a monotone linear
#     rescaling of the raw one, so the boxplots were geometrically identical and
#     the pair carried no extra information. Its subtitles also collided.
#   * Boxplots only, on a single subset axis. The kernel-density outline was
#     dropped because the divergent predictions stretch it into near-vertical
#     lines that obscure the boxes.
#   * NO text on the plot: no title, subtitle, caption, or annotation. The
#     disclosure that used to live in the subtitle (how many subjects fall
#     outside the plotted range, and why the scale is what it is) now lives in
#     the LaTeX caption instead, where it belongs.
#
# NOTE ON THE DATA. 27.4% of subjects have |nTotal_pred| > 1000 while the
# interquartile range of every subset lies within [-40, 2]; the tail reaches
# ~8e5. Those divergent predictions are a known pipeline artifact (the same
# signature as the constant 2.03e97 `female` column that used to sit in
# anchors.csv/Dbar.csv), not signal. The y range is therefore chosen to cover
# every subset's IQR with a margin rather than the full data range.

suppressPackageStartupMessages({
  library(ggplot2)
  library(patchwork)
})

args <- commandArgs(trailingOnly = TRUE)
src <- if (length(args) >= 1) args[1] else "output_v3-dl-centered/openTotal/subjectProfilesK5-v2/postPred.csv"
out_dir <- if (length(args) >= 2) args[2] else "output_v3-dl-centered/openTotal/subjectProfilesK5-v2"
if (!file.exists(src)) stop("missing: ", src)
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
d <- read.csv(src, stringsAsFactors = FALSE)

stopifnot(all(c("IID", "salso", "nTotal_pred") %in% names(d)))

d$y <- as.numeric(d$nTotal_pred)
d$subset <- factor(sprintf("Subset %d", d$salso),
                   levels = sprintf("Subset %d", sort(unique(d$salso))))

cat(sprintf("n = %d subjects, %d subsets\n", nrow(d), nlevels(d$subset)))

# y range covering every subset's IQR plus 8% padding, so the boxes are readable
lims <- function(v) {
  z <- tapply(v, d$subset, function(g) {
    qq <- quantile(g, c(0.25, 0.75), na.rm = TRUE); c(qq[1], qq[2])
  })
  m <- matrix(unlist(z), ncol = 2, byrow = TRUE)
  pad <- 0.08 * diff(range(m))
  c(min(m[, 1]) - pad, max(m[, 2]) + pad)
}
lim_y <- lims(d$y)
n_out <- sum(d$y < lim_y[1] | d$y > lim_y[2], na.rm = TRUE)
cat(sprintf("y range [%.2f, %.2f]; %d subjects (%.1f%%) outside\n",
            lim_y[1], lim_y[2], n_out, 100 * n_out / nrow(d)))
cat(sprintf("n outside per subset: %s\n",
            paste(sprintf("S%d=%d", seq_len(nlevels(d$subset)),
                          tapply(n_out * 0 + as.integer(d$y < lim_y[1] | d$y > lim_y[2]),
                                 d$subset, sum)), collapse = " ")))

# Okabe-Ito, colour-blind safe
cols <- c("#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7")
names(cols) <- levels(d$subset)

th <- theme_minimal(base_size = 11) +
  theme(panel.grid.minor = element_blank(),
        panel.grid.major.x = element_blank(),
        panel.background = element_rect(fill = "white", colour = NA),
        axis.text.x = element_text(face = "bold"),
        plot.margin = margin(6, 8, 4, 6))

p <- ggplot(d, aes(x = subset, y = y)) +
  geom_boxplot(aes(fill = subset), width = 0.5, outlier.shape = NA,
               colour = "grey15", linewidth = 0.4, alpha = 0.85) +
  scale_fill_manual(values = cols, guide = "none") +
  scale_x_discrete(expand = expansion(add = 0.6)) +
  coord_cartesian(ylim = lim_y) +
  labs(x = NULL, y = expression(hat(y)[i])) +
  th

png_out <- file.path(out_dir, "subject_profiles.png")
pdf_out <- file.path(out_dir, "subject_profiles.pdf")
ggsave(png_out, p, width = 6.4, height = 4.6, dpi = 400, bg = "white")
ggsave(pdf_out, p, width = 6.4, height = 4.6, bg = "white")

cat("wrote", png_out, "and", pdf_out, "\n")

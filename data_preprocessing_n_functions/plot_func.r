library(deSolve)
library(patchwork)
library(ggplot2)
library(ggpubr)
library(ggpattern)
library(dplyr)
library(data.table)

tree_order <- c("XGBoost", "LightGBM", "Random Forest")
linear_order <- c("Ridge regression", "Lasso regression", "Elastic Net regression")
scenario_order1 <- c("S1", "S2", "S3")
scenario_order2 <- c("S3", "S4", "S5")
scenario_color1 <- c(
  "S1" = "#747474",
  "S2" = "#D86ECC",
  "S3" = "#348EC2"
)
scenario_color2 <- c(
  "S3" = "#348EC2",
  "S4" = "#0070C0",
  "S5" = "#002060"
)


subset_func <- function(dframe, metric, model, sce){
  if(model %in% tree_order){
    model_order <- tree_order
  } else {
    model_order <- linear_order
  }
  if(sce == 1) {
    scenario_order <- scenario_order1
  } else {
    scenario_order <- scenario_order2
  }
  
  temp <- subset(
    dframe,
    Metrics == metric &
      Model %in% model_order &
      scenario %in% scenario_order,
    select = c(Model, scenario, estimate, lower, upper)
    )
  
  temp$Model <- factor(temp$Model, levels = model_order)
  temp$scenario <- factor(temp$scenario, levels = scenario_order)
  temp_final <- subset(temp, Model == model)
  
  return(temp_final)
}

perf_plot_func <- function(df, model, sce){
  dframe <- subset_func(df, "AUPRC value", model, sce)
  text_AUROC <- subset_func(df, "AUROC_value", model, sce)
  names(text_AUROC) <- c("Model", "scenario", "AUROC_estimate", "AUROC_lower", "AUROC_upper")
  text_Sensitivity <- subset_func(df, "Sensitivity", model, sce)
  names(text_Sensitivity) <- c("Model", "scenario", "Sensitivity_estimate", "Sensitivity_lower", "Sensitivity_upper")
  text_Specificity <- subset_func(df, "Specificity", model, sce)
  names(text_Specificity) <- c("Model", "scenario", "Specificity_estimate", "Specificity_lower", "Specificity_upper")

  text_dframe <- dframe %>%
    select(Model, scenario, upper) %>%
    rename(AUPRC_upper = upper) %>%
    left_join(text_AUROC, by = c("Model", "scenario")) %>%
    left_join(text_Sensitivity, by = c("Model", "scenario")) %>%
    left_join(text_Specificity, by = c("Model", "scenario"))
  
  text_dframe$text <- paste0(
    "AUROC: ", sprintf("%.1f", text_dframe$AUROC_estimate), " [", sprintf("%.1f", text_dframe$AUROC_lower), "-", sprintf("%.1f", text_dframe$AUROC_upper), "]\n",
    "Sens.: ", sprintf("%.1f", text_dframe$Sensitivity_estimate), " [", sprintf("%.1f", text_dframe$Sensitivity_lower), "-", sprintf("%.1f", text_dframe$Sensitivity_upper), "]\n",
    "Spec.: ", sprintf("%.1f", text_dframe$Specificity_estimate), " [", sprintf("%.1f", text_dframe$Specificity_lower), "-", sprintf("%.1f", text_dframe$Specificity_upper), "]"
    )
  
  if (sce == 1){
    text_dframe$text_position <- ifelse(as.character(text_dframe$scenario) == "S2", 64.5, 72)
  } else if (sce == 2){
    text_dframe$text_position <- ifelse(as.character(text_dframe$scenario) == "S4", 70, 78)
  }
  
  if(sce == 1){
    scenario_color <- scenario_color1
  } else {
    scenario_color <- scenario_color2
  }
  
  xlabel <- ""
  ylabel <- "AUPRC (%)"
  pnl <- ggplot(dframe, aes(x = scenario, y = estimate, fill = scenario)) +
    geom_col(width = 0.68, color = "black", linewidth = 1.1) +
    geom_errorbar(aes(ymin = lower, ymax = upper), width = 0.20, color = "black", linewidth = 1.8) +
    geom_text(data = text_dframe, aes(x = scenario, y = text_position, label = text), inherit.aes = FALSE, size = 6, lineheight = 0.95, vjust = 0) +
    scale_fill_manual(values = scenario_color) +
    scale_y_continuous(limits = c(0, 100), breaks = seq(0, 100, by = 20), expand = expansion(mult = c(0, 0.02))) +
    ggtitle(model) +
    xlab(xlabel) +
    ylab(ylabel) +
    theme(
      plot.title = element_text(
        colour = "black",
        size = 42,
        face = "bold",
        hjust = 0.5
        ),
      axis.text.x = element_text(colour = "black"),
      axis.text.y = element_text(colour = "black"),
      axis.ticks.x = element_line(colour = "black"),
      axis.ticks.y = element_line(colour = "black"),
      axis.line = element_line(colour = "black", linewidth = 1.5),
      panel.grid.major.x = element_blank(),
      panel.grid.major.y = element_blank(),
      panel.grid.minor = element_blank(),
      panel.background = element_blank(),
      legend.position = "none",
      axis.title.x = element_blank(),
      axis.title.y = element_text(colour = "black")
      )
  return(pnl)
}

loli_plot_func <- function(dframe, model_type, sce){
  if (model_type == "tree"){
    model_color <- c("XGBoost" = "#D55E00", "LightGBM" = "#009E73", "Random Forest" = "#000000")
    model_order <- c("XGBoost", "LightGBM", "Random Forest")
  } else if (model_type == "linear"){
    model_color <- c("Ridge regression" = "#a6cee3", "Lasso regression" = "#1f78b4", "Elastic Net regression" = "#b2df8a")
    model_order <- c("Ridge regression", "Lasso regression", "Elastic Net regression")
  }
  
  if (sce == 1){
    dframe_temp <- subset(
      dframe,
      Metric == "AUPRC value" &
        Model %in% model_order &
        (
          (sce1 == "S2" & sce2 == "S1") |
            (sce1 == "S3" & sce2 == "S1") |
            (sce1 == "S3" & sce2 == "S2")
          )
      )
    dframe_temp$Model <- factor(dframe_temp$Model, levels = model_order)
    dframe_temp$comparison <- paste(dframe_temp$sce1, "vs", dframe_temp$sce2)
    dframe_temp$comparison <- factor(dframe_temp$comparison, levels = c("S3 vs S2", "S3 vs S1", "S2 vs S1"))
    annot_label <- c("S3 vs S2", "S3 vs S1", "S2 vs S1")
    annot_break <- c(1, 2, 3)
    annot_limit <- c(0.55, 3.55)
    y_adjusted <- 3.40
  } else if (sce == 2){
    dframe_temp <- subset(
      dframe,
      Metric == "AUPRC value" &
        Model %in% model_order &
        (
          (sce1 == "S4" & sce2 == "S3") |
            (sce1 == "S5" & sce2 == "S3")
          )
      )
    dframe_temp$Model <- factor(dframe_temp$Model, levels = model_order)
    dframe_temp$comparison <- paste(dframe_temp$sce1, "vs", dframe_temp$sce2)
    dframe_temp$comparison <- factor(dframe_temp$comparison, levels = c("S4 vs S3", "S5 vs S3"))
    annot_label <- c("S4 vs S3", "S5 vs S3")
    annot_break <- c(1, 2)
    annot_limit <- c(0.55, 2.55)
    y_adjusted <- 2.40
    }
  
  dframe_temp$comparison_position <- as.numeric(dframe_temp$comparison)
  if (model_type == "tree"){
    dframe_temp$model_position <- ifelse(dframe_temp$Model == "XGBoost", 0.22,
                                         ifelse(dframe_temp$Model == "LightGBM", 0, -0.22)
    )
  } else if(model_type == "linear"){
    dframe_temp$model_position <- ifelse(dframe_temp$Model == "Ridge regression", 0.22,
                                         ifelse(dframe_temp$Model == "Lasso regression", 0, -0.22)
    )
    }
  
  dframe_temp$y_position <- dframe_temp$comparison_position + dframe_temp$model_position
  dframe_plot <- copy(dframe_temp)
  
  xlabel <- "Tie-adjusted win probability for AUPRC (%)"
  ylabel <- ""
  
  panel <- ggplot(dframe_plot) +
    geom_vline(xintercept = 50, linetype = "dashed", color = "black", linewidth = 1.5) +
    annotate("text", x = 30, y = y_adjusted, label = "Second scenario favored", size = 15, fontface = "italic", color = "grey30") +
    annotate("text", x = 70, y = y_adjusted, label = "First scenario favored", size = 15, fontface = "italic", color = "grey30") +
    geom_segment(aes(x = 50, xend = diff_count, y = y_position, yend = y_position, color = Model), linewidth = 2.2, lineend = "round", show.legend = FALSE) +
    geom_point(aes(x = diff_count, y = y_position, color = Model), size = 8) +
    scale_color_manual(values = model_color) +
    scale_x_continuous(limits = c(25, 75), breaks = seq(0, 100, by = 10)) +
    scale_y_continuous(limits = annot_limit, breaks = annot_break, labels = annot_label, expand = expansion(mult = c(0, 0))) +
    xlab(xlabel) +
    ylab(ylabel) +
    theme(
      axis.text.x = element_text(colour = "black"),
      axis.text.y = element_text(colour = "black"),
      axis.ticks.x = element_line(colour = "black"),
      axis.ticks.y = element_line(colour = "black"),
      axis.line = element_line(colour = "black", linewidth = 1.5),
      panel.grid.major.x = element_blank(),
      panel.grid.major.y = element_blank(),
      panel.grid.minor = element_blank(),
      panel.background = element_blank(),
      legend.position = "none",
      axis.title.x = element_text(colour = "black"),
      axis.title.y = element_blank(),
      plot.margin = margin(t = 20, r = 45, b = 25, l = 35))
  
  return(panel)
}
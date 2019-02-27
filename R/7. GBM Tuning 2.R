library(h2o)
library(tidyquant)
library(data.table)
library(Metrics)

# Import
setwd("C:\\Users\\user\\Desktop\\cz4041\\R")
data <- fread("train.csv")
submission <- fread("test.csv")
new_prediction <- fread("predict.csv")

# Train test split
smp_size <- floor(0.8 * nrow(data))
set.seed(1)
train_ind <- sample(seq_len(nrow(data)), size = smp_size)
train <- data[train_ind, ]
test <- data[-train_ind, ]

#####################
# Start of h2o
#####################

# Initialize
# Change port number if it doesnt init
h2o.init(nthreads = -1, max_mem_size = '8g', port = 54321)

# Formation Energy
#####################

# Initialize trainset and testset
train.h2o <- as.h2o(train)
test.h2o <- as.h2o(test)

# Set independent and dependent variables
y <- "bandgap_energy_ev"
x <- setdiff(names(train.h2o), y)

# Baseline GBM model
#################################################

# Create Model
gbm.1 <- h2o.gbm(y = y, x = x, training_frame = train.h2o, seed = 1)

# Make Predictions
r1 <- h2o.predict(gbm.1, newdata = test.h2o)
rmsle(test$bandgap_energy_ev, as.vector(r1))
# RMSLE: 0.08187957

# New model to tune no. of trees
#################################################
m2 <- h2o.gbm(
  x = x,
  y = y,
  training_frame = train.h2o,
  validation_frame = test.h2o,
  sample_rate = 0.8,
  col_sample_rate = 0.8,
  seed = 1,
  learn_rate = 0.01,
  # Score the model for every 5 new trees
  score_tree_interval = 1,
  # More than enough trees to find the most optimal number
  ntrees = 10000,
  # Stop whem RMSLE does not improve least 0.001% for 10 consecutive scoring events
  stopping_rounds = 10,
  stopping_tolerance = 1e-5,
  stopping_metric = "RMSLE"
)

# Get RMSLE with validation set
h2o.rmsle(h2o.performance(m2, newdata = test.h2o))
# RMSLE: 0.0771346

# Get optimal number of trees
m2.p <- m2@model
m2.p$model_summary[1]$number_of_trees
# Optimal no. of tree = 788


# New model to tune dept of trees
#################################################
hyper_params <- list(max_depth = seq(1, 65, 8))

# Huge increment to find range where it's the most suitable
grid1 <- h2o.grid(
  hyper_params = list(max_depth = seq(1, 65, 8)),
  search_criteria = list(strategy = "Cartesian"),
  algorithm = "gbm",
  grid_id = "grid11",
  x = x,
  y = y,
  training_frame = train.h2o,
  validation_frame = test.h2o,
  ntrees = 788,
  learn_rate = 0.01,
  sample_rate = 0.8,
  col_sample_rate = 0.8,
  seed = 1,
  stopping_rounds = 10,
  stopping_tolerance = 1e-5,
  stopping_metric = "RMSLE",
  score_tree_interval = 1
)

# Find range of top 5 maxDepths
sortedGrid <- h2o.getGrid("grid11", sort_by="rmsle", decreasing = F)
sortedGrid@summary_table

# Score does not improve. Stick with default of 5


# New model to tune Sample rate
#################################################
hyper_params <- list(
  # Tuned
  max_depth = 5,
  ntrees = 788,
  # Tuning
  sample_rate = seq(0.2,1,0.01)
)

grid2 <- h2o.grid(
  hyper_params = hyper_params,
  algorithm = "gbm",
  search_criteria = list(strategy = "Cartesian"),
  grid_id = "grid12",
  x = x,
  y = y,
  training_frame = train.h2o,
  validation_frame = test.h2o,
  seed = 1,
  stopping_rounds = 10,
  stopping_tolerance = 1e-5,
  stopping_metric = "RMSLE",
  score_tree_interval = 1,
  
  # Parameters that matters
  learn_rate = 0.01,
  col_sample_rate = 0.8
)

sortedGrid <- h2o.getGrid("grid12", sort_by = "rmsle", decreasing = F)
sortedGrid@summary_table
# RMSLE: 0.07577071483197818
# Sample rate: 0.39 - 0.42

# New model to tune learning rate
#################################################
hyper_params <- list(
  # Tuned
  max_depth = 5,
  ntrees = 788,
  sample_rate = seq(0.39,0.42,0.01),
  
  # Tuning
  learn_rate = seq(0.001, 0.02, 0.001)
)

grid3 <- h2o.grid(
  hyper_params = hyper_params,
  algorithm = "gbm",
  search_criteria = list(strategy = "Cartesian"),
  grid_id = "grid13",
  x = x,
  y = y,
  training_frame = train.h2o,
  validation_frame = test.h2o,
  seed = 1,
  stopping_rounds = 10,
  stopping_tolerance = 1e-5,
  stopping_metric = "RMSLE",
  score_tree_interval = 1,
  
  # Parameters that matters
  col_sample_rate = 0.8
)

sortedGrid <- h2o.getGrid("grid13", sort_by = "rmsle", decreasing = F)
head(sortedGrid@summary_table, n = 10)
# RMSLE: 0.07565276932903842
# Learn rate: 0.01 - 0.018


# New model to col sample rate
#################################################

hyper_params <- list(
  # Tuned
  max_depth = 5,
  ntrees = 788,
  
  # KIV
  sample_rate = seq(0.39,0.42,0.01),
  learn_rate = seq(0.01, 0.18, 0.01),
  
  # Tuning
  col_sample_rate = seq(0.6, 1, 0.1),
  col_sample_rate_per_tree = seq(0.6, 1, 0.1),
  col_sample_rate_change_per_level = seq(0.8, 1.2, 0.1)
)

grid5 <- h2o.grid(
  hyper_params = hyper_params,
  algorithm = "gbm",
  search_criteria = list(strategy = "Cartesian"),
  grid_id = "grid15",
  x = x,
  y = y,
  training_frame = train.h2o,
  validation_frame = test.h2o,
  seed = 1,
  stopping_rounds = 10,
  stopping_tolerance = 1e-5,
  stopping_metric = "RMSLE",
  score_tree_interval = 1
)

sortedGrid <- h2o.getGrid("grid15", sort_by = "rmsle", decreasing = F)
head(sortedGrid@summary_table, n = 5)
# RMSLE: 0.03539765


# Save top 10 models
# Takes about 3h to train. Verifying results will be easier this way
for (i in 1:length(sortedGrid@model_ids)) {
  temp <- h2o.getModel(sortedGrid@model_ids[[i]])
  
  # Save top 10 models to file
  if (i <= 10) {
    p <- h2o.saveModel(object = temp, path = paste0(getwd(), "\\R Models"), force = TRUE)
    name <- file.path(paste0(getwd(), "\\R Models"), paste0("final", toString(i)))
    file.rename(file.path(paste0(getwd(), "\\R Models"), temp@model_id), name)
  }
  # Delete all other models
  else {
    h2o.rm(temp)
  }
}


# Finalize hyper-parameters
#################################################
hyper_params <- list(
  max_depth = 5,
  ntrees = 465,
  sample_rate = seq(0.34, 0.44, 0.1),
  learn_rate = seq(0.06, 0.1, 0.001),
  col_sample_rate = seq(0.6, 0.9, 0.3),
  col_sample_rate_per_tree = seq(0.6, 0.8, 0.01),
  col_sample_rate_change_per_level = seq(0.8, 1.2, 0.01)
)


#################################################
#################################################

# Demo on loading models
final1 <- h2o.loadModel(paste0(getwd(), "\\R Models\\final1"))
h2o.rmsle(h2o.performance(final1, newdata = test.h2o))


# Create Model
gbm.1 <- h2o.gbm(y = y,
                 x = x,
                 training_frame = train.h2o,
                 seed = 1,
                 ntrees = 465,
                 sample_rate = 0.44,
                 learn_rate = 0.08,
                 col_sample_rate = 0.9,
                 col_sample_rate_change_per_level = 0.8,
                 col_sample_rate_per_tree = 0.7
)

# Make Predictions
submission.h2o <- as.h2o(submission)
r1 <- h2o.predict(gbm.1, newdata = submission.h2o)
r1 <- as.data.table(r1)

fwrite(r1, "predict.csv")







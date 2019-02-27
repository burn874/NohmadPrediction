library(h2o)
library(tidyquant)
library(data.table)
library(Metrics)

# Import
setwd("C:\\Users\\user\\Desktop\\cz4041\\R")
data <- fread("train.csv")
submission <- fread("test.csv")

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
train <- train[, c(1:12, 13)]
train.h2o <- as.h2o(train)
test <- test[, c(1:12, 13)]
test.h2o <- as.h2o(test)

# Set independent and dependent variables
y <- "formation_energy_ev_natom"
x <- setdiff(names(train.h2o), y)

# Baseline GBM model
#################################################

# Create Model
gbm.1 <- h2o.gbm(y = y, x = x, training_frame = train.h2o, seed = 1)

# Make Predictions
r1 <- h2o.predict(gbm.1, newdata = test.h2o)
rmsle(test$formation_energy_ev_natom, as.vector(r1))
# RMSLE: 0.03659391

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
# RMSLE: 0.03613407

# Get optimal number of trees
m2.p <- m2@model
m2.p$model_summary[1]$number_of_trees
# Optimal no. of tree = 465


# New model to tune dept of trees
#################################################
hyper_params <- list(max_depth = seq(1, 65, 8))

# Huge increment to find range where it's the most suitable
grid1 <- h2o.grid(
  hyper_params = list(max_depth = seq(1, 65, 8)),
  search_criteria = list(strategy = "Cartesian"),
  algorithm = "gbm",
  grid_id = "grid1",
  x = x,
  y = y,
  training_frame = train.h2o,
  validation_frame = test.h2o,
  ntrees = 465,
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
sortedGrid <- h2o.getGrid("grid1", sort_by="rmsle", decreasing = F)
sortedGrid@summary_table

# Score does not improve. Stick with default of 5


# New model to tune Sample rate
#################################################
hyper_params <- list(
  # Tuned
  max_depth = 5,
  ntrees = 465,
  # Tuning
  sample_rate = seq(0.2,1,0.01)
  )

grid2 <- h2o.grid(
  hyper_params = hyper_params,
  algorithm = "gbm",
  search_criteria = list(strategy = "Cartesian"),
  grid_id = "grid2",
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

sortedGrid <- h2o.getGrid("grid2", sort_by = "rmsle", decreasing = F)
sortedGrid@summary_table
# RMSLE: 0.03598403
# Sample rate: 0.34 - 0.79
# Sample rate between this range seem to perform about the same.
# Will be able to find a better range when tweaking other parameters

# New model to tune learning rate
#################################################
hyper_params <- list(
  # Tuned
  max_depth = 5,
  ntrees = 465,
  sample_rate = seq(0.34,0.79,0.05),
  
  # Tuning
  learn_rate = seq(0.001, 0.01, 0.001)
)

grid3 <- h2o.grid(
  hyper_params = hyper_params,
  algorithm = "gbm",
  search_criteria = list(strategy = "Cartesian"),
  grid_id = "grid3",
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

sortedGrid <- h2o.getGrid("grid3", sort_by = "rmsle", decreasing = F)
sortedGrid@summary_table
# RMSLE: 0.03598403


# Higher learning rate seems to be better. Tweaking search range.
hyper_params <- list(
  # Tuned
  max_depth = 5,
  ntrees = 465,
  sample_rate = seq(0.34,0.79,0.05),
  
  # Tuning
  learn_rate = seq(0.01, 0.1, 0.005)
)

grid4 <- h2o.grid(
  hyper_params = hyper_params,
  algorithm = "gbm",
  search_criteria = list(strategy = "Cartesian"),
  grid_id = "grid4",
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

sortedGrid <- h2o.getGrid("grid4", sort_by = "rmsle", decreasing = F)
head(sortedGrid@summary_table, n = 10)
# RMSLE: 0.03592989
# Learn rate: 0.01 - 0.06
# Tweak sample rate range
# Sample rate: 0.34 - 0.54


# New model to col sample rate
#################################################

hyper_params <- list(
  # Tuned
  max_depth = 5,
  ntrees = 465,
  
  # KIV
  sample_rate = seq(0.34, 0.54, 0.1),
  learn_rate = seq(0.01, 0.1, 0.01),
  
  # Tuning
  col_sample_rate = seq(0.6, 1, 0.1),
  col_sample_rate_per_tree = seq(0.6, 1, 0.1),
  col_sample_rate_change_per_level = seq(0.8, 1.2, 0.1)
)

grid5 <- h2o.grid(
  hyper_params = hyper_params,
  algorithm = "gbm",
  search_criteria = list(strategy = "Cartesian"),
  grid_id = "grid5",
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

sortedGrid <- h2o.getGrid("grid5", sort_by = "rmsle", decreasing = F)
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









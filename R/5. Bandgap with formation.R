library(h2o)
library(tidyquant)
library(data.table)
library(Metrics)

# Import
setwd("C:\\Users\\user\\Desktop\\R")
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
#h2o.init(nthreads = -1, max_mem_size = '8g', port = 54321)

# Formation Energy
#####################

# Initialize trainset and testset
train.h2o <- as.h2o(train)
test <- test[, c(1:13, 14)]
test$formation_energy_ev_natom <- as.vector(h2o.predict(m3, newdata = test.h2o))
test.h2o <- as.h2o(test)

# Set independent and dependent variables
y <- "bandgap_energy_ev"
x <- setdiff(names(train.h2o), y)

#train.h2o <- as.h2o(data[, c(1:12, 13)])

# Create Model
m1 <- h2o.deeplearning(y = y, x = x, training_frame = train.h2o, seed = 1)
m2 <- h2o.glm(y = y, x = x, training_frame = train.h2o, seed = 1)
m3 <- h2o.gbm(y = y, x = x, training_frame = train.h2o, seed = 1)
m4 <- h2o.randomForest(y = y, x = x, training_frame = train.h2o, seed = 1)

# Predict on test set
r1 <- h2o.predict(m1, newdata = test.h2o)
rmsle(test$bandgap_energy_ev, as.vector(r1))
# 0.1004382

r1 <- h2o.predict(m2, newdata = test.h2o)
rmsle(test$bandgap_energy_ev, as.vector(r1))
# 0.1355821

r1 <- h2o.predict(m3, newdata = test.h2o)
rmsle(test$bandgap_energy_ev, as.vector(r1))
# 0.09378788

r1 <- h2o.predict(m4, newdata = test.h2o)
rmsle(test$bandgap_energy_ev, as.vector(r1))
# 0.09338316




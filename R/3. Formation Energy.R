library(h2o)
library(tidyquant)
library(data.table)

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

#train.h2o <- as.h2o(data[, c(1:12, 13)])

# Create Model
m1 <- h2o.deeplearning(y = y, x = x, training_frame = train.h2o, seed = 1)
m2 <- h2o.glm(y = y, x = x, training_frame = train.h2o, seed = 1)
m8 <- h2o.gbm(y = y, x = x, training_frame = train.h2o, seed = 1)

# Predict on test set
r1 <- h2o.predict(m1, newdata = test.h2o)
rmsle(test$formation_energy_ev_natom, as.vector(r1))

r1 <- h2o.predict(m2, newdata = test.h2o)
rmsle(test$formation_energy_ev_natom, as.vector(r1))

r1 <- h2o.predict(m8, newdata = test.h2o)
rmsle(test$formation_energy_ev_natom, as.vector(r1))








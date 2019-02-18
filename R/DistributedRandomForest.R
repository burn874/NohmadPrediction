library(h2o)
library(tidyquant)

# Import
setwd("C:\\Users\\user\\Desktop\\Updated R Codes")
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
m.drf1 <- h2o.randomForest(y = y, x = x, training_frame = train.h2o, ntrees = 1001, seed = 1)
m.gbm1 <- h2o.gbm(y = y, x = x, training_frame = train.h2o, ntrees = 1001, seed = 1)

# Predict on test set
r.drf1 <- h2o.predict(m.drf1, newdata = test.h2o)
r.gbm1 <- h2o.predict(m.gbm1, newdata = test.h2o)

# Error
rmsle(test$formation_energy_ev_natom, as.vector(r.drf1))
# 0.039418
rmsle(test$formation_energy_ev_natom, as.vector(r.gbm1))
# 0.04165609

# Formation Energy
#####################

# Import
data <- fread("train.csv")

# Train test split
smp_size <- floor(0.8 * nrow(data))
set.seed(1)
train_ind <- sample(seq_len(nrow(data)), size = smp_size)
train <- data[train_ind, ]
test <- data[-train_ind, ]

# Initialize trainset and testset
train <- train[, c(1:12, 14)]
train.h2o <- as.h2o(train)
test <- test[, c(1:12, 14)]
test.h2o <- as.h2o(test)

# Set independent and dependent variables
y <- "bandgap_energy_ev"
x <- setdiff(names(train.h2o), y)

#train.h2o <- as.h2o(data[, c(1:12, 14)])

# Create Model
m.drf2 <- h2o.randomForest(y = y, x = x, training_frame = train.h2o, ntrees = 1001, seed = 1)
m.gbm2 <- h2o.gbm(y = y, x = x, training_frame = train.h2o, ntrees = 1001, seed = 1)

# Predict on test set
r.drf2 <- h2o.predict(m.drf2, newdata = test.h2o)
r.gbm2 <- h2o.predict(m.gbm2, newdata = test.h2o)

# Error
rmsle(test$bandgap_energy_ev, as.vector(r.drf2))
# 0.09501631
rmsle(test$bandgap_energy_ev, as.vector(r.gbm2))
# 0.1055961


# Prediction
#####################

submission.h2o <- as.h2o(submission)

r.drf1 <- h2o.predict(m.drf1, newdata = submission.h2o)
r.gbm1 <- h2o.predict(m.gbm1, newdata = submission.h2o)
r.drf2 <- h2o.predict(m.drf2, newdata = submission.h2o)
r.gbm2 <- h2o.predict(m.gbm2, newdata = submission.h2o)


# Prepare Submission drf
#####################
result <- data.frame(as.vector(r.drf1), as.vector(r.drf2))
id <- as.data.frame(c(1:nrow(result)))
result <- cbind.data.frame(id, result)
names(result) <- c("id", "formation_energy_ev_natom", "bandgap_energy_ev")

fwrite(result, "submission.csv")


# Prepare Submission gbm
#####################
result <- data.frame(as.vector(r.gbm1), as.vector(r.gbm2))
id <- as.data.frame(c(1:nrow(result)))
result <- cbind.data.frame(id, result)
names(result) <- c("id", "formation_energy_ev_natom", "bandgap_energy_ev")

fwrite(result, "submission.csv")









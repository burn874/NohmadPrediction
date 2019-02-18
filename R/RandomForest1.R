library(randomForest)
library(data.table)
library(Metrics)

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


# Formation Energy
#####################

train1 <- train[,c(1:12, 13)]
test1 <- test[,c(1:12, 13)]

# Create Model
#train1 <- data[, c(1:12, 13)]
m.rf1 <- randomForest(formation_energy_ev_natom ~ ., data = train1, seed = 1, ntree = 1001)

# data$formation_energy_ev_natom
# data$bandgap_energy_ev

# Predict on test set
r.rf1 <- predict(m.rf1, type = "response", test1)

rmsle(test$formation_energy_ev_natom, r.rf1)
# 0.0389631

# Bandgap
#####################

train2 <- train[,c(1:12, 14)]
test2 <- test[,c(1:12, 14)]

# Create Model
#train2 <- data[, c(1:12, 14)]
m.rf2 <- randomForest(bandgap_energy_ev ~ ., data = train2, seed = 1, ntree = 1001)

# Predict on test set
r.rf2 <- predict(m.rf2, type = "response", test2)

rmsle(test$bandgap_energy_ev, r.rf2)
# 0.0962167

# Predict on Submission
#####################

r.rf1 <- predict(m.rf1, type = "response", submission)
r.rf2 <- predict(m.rf2, type = "response", submission)


# Prepare Submission
#####################
result <- data.frame(r.rf1, r.rf2)
id <- as.data.frame(c(1:nrow(result)))
result <- cbind.data.frame(id, result)
names(result) <- c("id", "formation_energy_ev_natom", "bandgap_energy_ev")

fwrite(result, "submission.csv")








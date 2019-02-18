library(caTools)
library(DMwR)
library(caret)
library(data.table)
library(corrplot)

setwd("C:\\Users\\user\\Desktop\\Updated R Codes")

data <- fread("train.csv")
data2 <- fread("test.csv")

#################################################
# PCA
#################################################

newdata <- rbind(data[, c(1:12)], data2)

# PCA
PCA <- prcomp(~ ., data = newdata, scale = T, center = T)

# Biplot
#biplot(PCA)

# Scree Plot
#plot(PCA$sdev/sum(PCA$sdev^2), type = "b")

#Remove unwanted PCA components which explain least variance
PC <- subset(PCA$x, select=-c(PC10,PC11, PC12))

#Final data with PCA
PCA.dt <- data.table(PC)
submission <- PCA.dt[c(2401:3000), ]
data <- cbind(PCA.dt[c(1:2400), ], data[, c(13:14)])





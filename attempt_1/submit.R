library(tidyverse)

load("./attempt_1/results/final_pred.rda")
train <- read_csv("data/train.csv")


write.csv(final_pred_svmr, file = "./attempt_1/attempt1.csv"
          , col.names = FALSE, row.names = FALSE)

write.csv(final_pred_svmp, file = "./attempt_1/attempt2.csv"
          , col.names = FALSE, row.names = FALSE)

write.csv(final_pred_mars, file = "./attempt_1/attempt3.csv"
          , col.names = FALSE, row.names = FALSE)

###### SVMR is best model, MARS 2nd

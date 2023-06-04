# get final model results

library(tidymodels)
library(tidyverse)

tidymodels_prefer()

result_files <- list.files("attempt_1/results/", "*.rda", full.names = TRUE)

for(i in result_files){
  load(i)
}

train <- read_csv("data/train.csv")
test <- read_csv("data/test.csv")

#### Set up parallel processing
parallel::detectCores()
cl <- makePSOCKcluster(8)
registerDoParallel(cl)

##############################################
# baseline/null model
null_model <- null_model(mode = "regression") %>%
  set_engine("parsnip")

null_workflow <- workflow() %>%
  add_model(null_model) %>%
  add_recipe(recipe_sink)

null_fit <- null_workflow %>%
  fit_resamples(resamples = a1_folds,
                control = control_resamples(save_pred = TRUE,
                                            save_workflow = TRUE))
null_metrics <- null_fit %>%
  collect_metrics()
# rmse = 0.9995754

###############################################
# organzie results to find best overall
# put all tune_grids together

model_set <- as_workflow_set(
  "elastic_net" = net_tune,
  "knn" = knn_tune,
  "neural_network" = mlp_tune,
  "svm_poly" = svmp_tune,
  "svm_rbf" = svmr_tune,
  "mars" = mars_tune
)

# plot of our results
model_results <- model_set %>%
  group_by(wflow_id) %>%
  mutate(best = map(result, show_best, metric = "rmse", n = 1)) %>%
  select(best) %>%
  unnest(cols = c(best)) 
# svmr, svmp, and mars are best 3. focus on those

model_table <- model_set %>%
  autoplot(metric = "rmse", select_best = TRUE) +
  theme_minimal() +
  geom_text(aes(y = mean + .03, label = wflow_id), angle = 90, hjust = 1) +
  theme(legend.position = "none")

# extract workflows
svmr_workflow <- extract_workflow(svmr_tune)
svmp_workflow <- extract_workflow(svmp_tune)
mars_workflow <- extract_workflow(mars_tune)

################################### finalize the best 3 workflows
best_svmr <- svmr_workflow %>%
  finalize_workflow(select_best(svmr_tune, metric = "rmse"))

best_svmp <- svmp_workflow %>%
  finalize_workflow(select_best(svmp_tune, metric = "rmse"))

best_mars <- mars_workflow %>%
  finalize_workflow(select_best(mars_tune, metric = "rmse"))

# fit training to final workflows

svmr_fit <- fit(best_svmr, train)

svmp_fit <- fit(best_svmp, train)

mars_fit <- fit(best_mars, train)



save(model_results, file = "attempt_1/results/model_results.rda")
save(model_set, svmr_fit, svmp_fit, mars_fit,
     file = "./attempt_1/results/results.rda")

################################################
load("./attempt_1/results/results.rda")
# test data

test <- test %>%
  select(id, get_var)

final_pred_svmr <- predict(svmr_fit, test) %>% 
  bind_cols(test %>% select(id))

final_pred_svmp <- predict(svmp_fit, test) %>% 
  bind_cols(test %>% select(id))

final_pred_mars <- predict(mars_fit, test) %>% 
  bind_cols(test %>% select(id))

# untransform response variable

final_pred_svmr <- final_pred_svmr %>%
  mutate(y = .pred) %>%
  select(-.pred)

final_pred_svmp <- final_pred_svmp %>%
  mutate(y = .pred) %>%
  select(-.pred)

final_pred_mars <- final_pred_mars %>%
  mutate(y = .pred) %>%
  select(-.pred)


save(final_pred_svmr, final_pred_svmp, final_pred_mars, 
     file = "./attempt_1/results/final_pred.rda")


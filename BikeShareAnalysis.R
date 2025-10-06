library(tidymodels)
library(tidyverse)
library(vroom)


bikeTrain <- vroom("train.csv")
bikeTest <- vroom("test.csv")


bikeTrain_clean <- bikeTrain |>
  select(-casual, -registered) |>
  mutate(count = log(count))


bike_recipe <- recipe(count ~ ., data = bikeTrain_clean) |>
  step_mutate(weather = ifelse(weather == 4, 3, weather)) |>
  step_mutate(weather = factor(weather)) |>
  step_time(datetime, features = "hour") |>
  step_mutate(season = factor(season)) |>
  step_zv(all_predictors()) |>
  step_dummy(all_nominal_predictors()) |>
  step_normalize(all_numeric_predictors()) |>
  step_rm(datetime)


bike_prep <- prep(bike_recipe)
bike_baked <- bake(bike_prep, new_data = bikeTrain_clean)


print(head(bike_baked, 5))


#Linear Regression
my_linear_model <- linear_reg() %>%
  set_engine("lm") %>%
  set_mode("regression") %>%
  fit(log(count) ~ season + holiday + workingday + weather +
        temp + atemp + humidity + windspeed,
      data = bike_train)


bike_predictions <- predict(my_linear_model, new_data = bike_test)


bike_predictions <- exp(bike_predictions)


kaggle_submission <- bike_predictions %>%
  bind_cols(bike_test) %>%
  select(datetime, .pred) %>%
  rename(count = .pred) %>%
  mutate(count = pmax(0, count)) %>%
  mutate(datetime = as.character(format(datetime)))


vroom_write(x = kaggle_submission, file = "./LinearPreds.csv", delim = ",")



#Penalized Regression
preg_model1 <- linear_reg(penalty = 0.2, mixture = 0.5) |>
  set_engine("glmnet")
preg_wf1 <- workflow() |>
  add_recipe(bike_recipe) |>
  add_model(preg_model1) |>
  fit(data = bikeTrain_clean)
pen_pred1 <- predict(preg_wf1, new_data = bikeTest)


penalized_submission5 <- pen_pred5 |>
  bind_cols(bikeTest |> select(datetime)) |>
  select(datetime, .pred) |>
  rename(count = .pred) |>
  mutate(count = pmax(0, count)) |>
  mutate(datetime = as.character(format(datetime)))


vroom_write(x = penalized_submission5, file = "./penalized5.csv", delim = ',')


#CV tuning parameters
preg_model <- linear_reg(penalty = tune(),
                         mixture = tune()) |>
  set_engine('glmnet')


preg_wf <- workflow() |>
  add_recipe(bike_recipe) |>
  add_model(preg_model)


grid_of_tuning_params <- grid_regular(penalty(),
                                      mixture(),
                                      levels = 25)


folds <- vfold_cv(bikeTrain_clean, v = 25, repeats = 1)


CV_results <- preg_wf |>
  tune_grid(resamples = folds,
            grid = grid_of_tuning_params,
            metrics = metric_set(rmse, mae))


collect_metrics(CV_results) |>
  filter(.metric == 'rmse') |>
  ggplot(data = bikeTrain_clean, aes(x = penalty, y = mean, color = factor(mixture))) +
  geom_line()


bestTune <- CV_results |>
  select_best(metric = 'rmse')


final_wf <- 
  preg_wf |>
  finalize_workflow(bestTune) |>
  fit(data = bikeTrain_clean)


tuning_preds <- predict(final_wf, new_data = bikeTest)


tuning_preds <- exp(tuning_preds)


tuning_submission <- tuning_preds |>
  bind_cols(bikeTest) |>
  select(datetime, .pred) |>
  rename(count = .pred) |>
  mutate(count = pmax(0, count)) |>
  mutate(datetime = as.character(format(datetime)))


vroom_write(x = tuning_submission, file = "./Tuning25.csv", delim = ",")


#Regression Trees
treereg_mod <- decision_tree(tree_depth = tune(),
                        cost_complexity = tune(),
                        min_n = tune()) |>
  set_engine("rpart") |>
  set_mode("regression")

treereg_wf <- workflow() |>
  add_recipe(bike_recipe) |>
  add_model(treereg_mod)

grid_of_tuning_params_tree <- grid_regular(tree_depth(),
                                      cost_complexity(),
                                      min_n(),
                                      levels = 5)

folds <- vfold_cv(bikeTrain_clean, v = 5, repeats = 1)

CV_results <- treereg_wf |>
  tune_grid(resamples = folds,
            grid = grid_of_tuning_params_tree,
            metrics = metric_set(rmse, mae))

bestTune_tree <- CV_results |>
  select_best(metric = 'rmse')

final_tree_wf <- 
  treereg_wf |>
  finalize_workflow(bestTune_tree) |>
  fit(data = bikeTrain_clean)

treereg_preds <- predict(final_tree_wf, new_data = bikeTest)

treereg_preds <- exp(treereg_preds)

treereg_submission <- treereg_preds |>
  bind_cols(bikeTest) |>
  select(datetime, .pred) |>
  rename(count = .pred) |>
  mutate(count = pmax(0, count)) |>
  mutate(datetime = as.character(format(datetime)))


vroom_write(x = treereg_submission, file = "./TreeReg.csv", delim = ",")


#Random Forests
install.packages("rpart")


randfor_mod <- rand_forest(mtry = tune(),
                           min_n = tune(),
                           trees = 300) |>
  set_engine('ranger') |>
  set_mode('regression')


randfor_wf <- workflow() |>
  add_recipe(bike_recipe) |>
  add_model(randfor_mod)



rand_for_grid <- grid_regular(mtry(range = c(1, 12)),
                       min_n(),
                       levels = 3)

folds_for <- vfold_cv(bikeTrain_clean, v = 3, repeats = 1)

CV_results_for <- randfor_wf |>
  tune_grid(resamples = folds_for,
            grid = rand_for_grid,
            metrics = metric_set(rmse, mae))


bestTune_randfor <- CV_results_for |>
  select_best(metric = 'rmse')


final_randfor_wf <- 
  randfor_wf |>
  finalize_workflow(bestTune_randfor) |>
  fit(data = bikeTrain_clean)


randfor_preds <- predict(final_randfor_wf, new_data = bikeTest)


randfor_preds <- exp(randfor_preds)

randfor_submission <- randfor_preds |>
  bind_cols(bikeTest) |>
  select(datetime, .pred) |>
  rename(count = .pred) |>
  mutate(count = pmax(0, count)) |>
  mutate(datetime = as.character(format(datetime)))


vroom_write(x = randfor_submission, file = "./RandomForests.csv", delim = ",")


#Boosted Trees and BART
library(bonsai)
library(lightgbm)
library(dbarts)

boost_model <- boost_tree(tree_depth = tune(),
                          trees = tune(),
                          learn_rate = tune()) |>
  set_engine("lightgbm") |>
  set_mode('regression')


boost_wf <- workflow() |>
  add_recipe(bike_recipe) |>
  add_model(boost_model)



boost_grid <- grid_regular(tree_depth(),
                           trees(),
                           learn_rate(),
                           levels = 3)

folds_boost <- vfold_cv(bikeTrain_clean, v = 3, repeats = 1)

CV_results_boost <- boost_wf |>
  tune_grid(resamples = folds_boost,
            grid = boost_grid,
            metrics = metric_set(rmse, mae))


bestTune_boost <- CV_results_boost |>
  select_best(metric = 'rmse')


final_boost_wf <- 
  boost_wf |>
  finalize_workflow(bestTune_boost) |>
  fit(data = bikeTrain_clean)


boost_preds <- predict(final_boost_wf, new_data = bikeTest)


boost_preds <- exp(boost_preds)

boost_submission <- boost_preds |>
  bind_cols(bikeTest) |>
  select(datetime, .pred) |>
  rename(count = .pred) |>
  mutate(count = pmax(0, count)) |>
  mutate(datetime = as.character(format(datetime)))


vroom_write(x = boost_submission, file = "./Boost2.csv", delim = ",")


bart_model <- bart(trees = tune()) |>
  set_engine('dbarts') |>
  set_mode('regression')


bart_wf <- workflow() |>
  add_recipe(bike_recipe) |>
  add_model(bart_model)



bart_grid <- grid_regular(trees(),
                           levels = 2)

folds_bart <- vfold_cv(bikeTrain_clean, v = 3, repeats = 1)

CV_results_bart <- bart_wf |>
  tune_grid(resamples = folds_bart,
            grid = bart_grid,
            metrics = metric_set(rmse, mae))


bestTune_bart <- CV_results_bart |>
  select_best(metric = 'rmse')


final_bart_wf <- 
  bart_wf |>
  finalize_workflow(bestTune_bart) |>
  fit(data = bikeTrain_clean)


bart_preds <- predict(final_bart_wf, new_data = bikeTest)


bart_preds <- exp(bart_preds)

bart_submission <- bart_preds |>
  bind_cols(bikeTest) |>
  select(datetime, .pred) |>
  rename(count = .pred) |>
  mutate(count = pmax(0, count)) |>
  mutate(datetime = as.character(format(datetime)))


vroom_write(x = bart_submission, file = "./BART.csv", delim = ",")


# Stacking with H20.ai
library(agua)


Sys.setenv(JAVA_HOME = "C:/Program Files/Eclipse Adoptium/jdk-25.0.0.36-hotspot")
Sys.setenv(PATH = paste(Sys.getenv("JAVA_HOME"), "bin", Sys.getenv("PATH"), sep=";"))


h2o::h2o.init()


auto_model <- auto_ml() |>
  set_engine('h2o', max_runtime_secs = 120) |>
  set_mode('regression')


automl_wf <- workflow() |>
  add_recipe(bike_recipe) |>
  add_model(auto_model) |>
  fit(data = bikeTrain_clean)


auto_preds <- predict(automl_wf, new_data = bikeTest)
auto_preds <- exp(auto_preds)


auto_submission <- auto_preds |>
  bind_cols(bikeTest) |>
  select(datetime, .pred) |>
  rename(count = .pred) |>
  mutate(count = pmax(0, count)) |>
  mutate(datetime = as.character(format(datetime)))


vroom_write(x = auto_submission, file = "./Stacking120.csv", delim = ",")


# Data Robot
dbot <- vroom("RoboPreds.csv")


kaggle_dbot <- dbot |>
  bind_cols(bikeTest) |> 
  select(datetime, count_PREDICTION) |> 
  rename(count = count_PREDICTION) |>
  mutate(count = pmax(0, exp(count))) |> 
  mutate(datetime = as.character(format(datetime)))
vroom_write(x=kaggle_dbot, file="./Data_Robot.csv", delim=",")

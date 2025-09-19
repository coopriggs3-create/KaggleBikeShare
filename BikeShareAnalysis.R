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

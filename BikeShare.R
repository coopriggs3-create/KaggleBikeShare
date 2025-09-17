library(tidyverse)
library(tidymodels)
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

#Linear Regression Model
lin_model <- linear_reg() |>
  set_engine('lm') |>
  set_mode('regression')


bike_workflow <- workflow() |>
  add_recipe(bike_recipe) |>
  add_model(lin_model) |>
  fit(data = bikeTrain_clean)


lin_preds <- predict(bike_workflow, new_data = bikeTest)


lin_preds <- lin_preds |>
  mutate(.pred = exp(.pred))


print(head(lin_preds, 5))


#Penalized Regression Model
preg_model <- linear_reg(penalty = 0.2, mixture = 0.5) |>
  set_engine("glmnet")
preg_wf <- workflow() |>
  add_recipe(bike_recipe) |>
  add_model(preg_model) |>
  fit(data = bikeTrain_clean)
pen_pred <- predict(preg_wf, new_data = bikeTest)


workflows_submission <- lin_preds |>
  bind_cols(bikeTest |> select(datetime)) |>
  select(datetime, .pred) |>
  rename(count = .pred) |>
  mutate(count = pmax(0, count)) |>
  mutate(datetime = as.character(format(datetime)))


vroom_write(x = workflows_submission, file = "./workflows.csv", delim = ',')



penalized_submission <- pen_pred |>
  bind_cols(bikeTest |> select(datetime)) |>
  select(datetime, .pred) |>
  rename(count = .pred) |>
  mutate(count = pmax(0, count)) |>
  mutate(datetime = as.character(format(datetime)))


vroom_write(x = workflows_submission, file = "./workflows.csv", delim = ',')
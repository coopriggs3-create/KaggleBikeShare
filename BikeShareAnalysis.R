library(tidymodels)
library(tidyverse)
library(vroom)


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

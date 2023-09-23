library(tidyverse)
library(summarytools)

result2 = read_csv("C:\\Users\\ahmad\\OneDrive\\Desktop\\PhD\\Codes\\Fair_Metric_Learning\\plots\\result_3.csv")
result2$run = "2"
result1 = read_csv("C:\\Users\\ahmad\\OneDrive\\Desktop\\PhD\\Codes\\Fair_Metric_Learning\\plots\\results.csv")
result1$run = "1"
result = rbind(result1, result2)

ggplot(subset(result, indicator == "mpe"), aes(x = value, color = decorrelation))+
  geom_density()+
  facet_wrap(data~type+output_type, scales = "free")+
  theme_bw()


stats = result %>% 
  mutate(output_type = ifelse(type == "triplet", "", output_type)) %>% 
  group_by(data, type, decorrelation, output_type, metric, margin, lambda, indicator, run) %>% 
  summarise(value = mean(value))
View(stats)

stats %>% 
  filter(indicator == "acc" & value > 0.85) %>%
  ungroup() %>%
  dfSummary() %>% view()
  
stats %>% 
  filter(indicator == "acc" & value > 0.85 & data != "imf") %>%
  ungroup() %>%
  dfSummary() %>% view()

stats %>% 
  filter(indicator == "acc"  & data != "imf") %>%
  group_by(output_type) %>% summarise(value = mean(value))




result %>% 
  group_by(data, type, indicator,output_type,margin) %>% # , decorrelation, output_type, metric, margin, lambda, indicator) %>% 
  summarise(value = mean(value)) %>%  
  filter(indicator == "rmse" & data!= "imf") %>%
  select(-indicator) %>% 
  View()

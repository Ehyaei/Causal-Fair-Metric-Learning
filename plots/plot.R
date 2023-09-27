library(tidyverse)
library(summarytools)

result2 = read_csv("C:\\Users\\ahmad\\OneDrive\\Desktop\\PhD\\Codes\\Fair_Metric_Learning\\plots\\result_3.csv")
result2$run = "2"
result1 = read_csv("C:\\Users\\ahmad\\OneDrive\\Desktop\\PhD\\Codes\\Fair_Metric_Learning\\plots\\results.csv")
result1$run = "1"
result = rbind(result1, result2)

result = read_csv("C:\\Users\\ahmad\\OneDrive\\Desktop\\PhD\\Codes\\Fair_Metric_Learning\\plots\\results.csv")
sort(unique(result$seed))
result$run = "1"
result$run[which(result$seed>40)] = "2"
ggplot(subset(result, indicator == "mpe"), aes(x = value, color = decorrelation))+
  geom_density()+
  facet_wrap(data~type+output_type, scales = "free")+
  theme_bw()


stats = result %>% 
  mutate(output_type = ifelse(type == "triplet", "", output_type)) %>% 
  group_by(data, type, decorrelation, output_type, metric, margin, lambda, indicator, run) %>% 
  summarise(value = mean(value))
sort(View(stats))

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
 result %>% group_by(decorrelation,indicator, type) %>% summarise(mean(value)) %>% View
 
  # n=20,000
 result1 = read_csv("C:\\Users\\ahmad\\OneDrive\\Desktop\\PhD\\Codes\\Fair_Metric_Learning\\plots\\results.csv")
 result1$run = "01"
  # n = 10,000
  result2 = read_csv("C:\\Users\\ahmad\\OneDrive\\Desktop\\PhD\\Codes\\Fair_Metric_Learning\\plots\\results.csv")
  result2$run = "02"
  
  # epoch = 40
  result3 = read_csv("C:\\Users\\ahmad\\OneDrive\\Desktop\\PhD\\Codes\\Fair_Metric_Learning\\plots\\results.csv")
  result3$run = "03"
  
  # new design
  result4 = read_csv("C:\\Users\\ahmad\\OneDrive\\Desktop\\PhD\\Codes\\Fair_Metric_Learning\\plots\\results.csv")
  result4$run = "04"
  View(result4)
  
  result = rbind(result1, result2, result3, result4)

View(result) 
result %>% group_by(margin, decorrelation, indicator, run) %>% summarise(value = mean(value)) %>% 
  filter(indicator == "acc") %>% View





result = read_csv("C:\\Users\\ahmad\\OneDrive\\Desktop\\PhD\\Codes\\Fair_Metric_Learning\\plots\\results.csv")
result = read_csv("C:\\Users\\ahmad\\Downloads/results.csv")

stats = result %>% 
  group_by(data, type, decorrelation, output_type, metric, radii, lambda, indicator) %>% 
  summarise(value = mean(value))
sort(View(stats))

View(result)

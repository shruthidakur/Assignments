library(tidyverse)  #Loads tidyverse library
library(haven)      #Loads haven library- reads SPSS, Stata, and SAS files.
setwd("C:/MY_BOOKS/WINTER-22/OMSBA 5112")
Alumni<-read_csv("Alumni.csv")      #reads Alumni.csv and generates Alumni Dataframe
summary(Alumni)         #Computes summary statistics on alumni Dataframe(minimum,maximum,Mean,Median,etc)
summary(subset(Alumni,select=c(sfratio,alumnigivingrate))) 
summary(subset(Alumni,subset=sfratio<10,select=c(sfratio,alumnigivingrate))) 
nfhs<-read_dta("IAHR52FL.dta")      #reads IAHR52FL.dta and generates nfhs Dataframe
new_df <- nfhs%>%                   #read new_df Dataframe                 
  select(hhid:hv208)%>%
  rename(survey_month=hv006)%>%
  filter(survey_month==1)%>%
  mutate(rural=hv025==2)
ggplot(data=nfhs,                 #Plot graph for count of Number of household members
       mapping=aes(x=hv009),binwidth=1)+
  geom_histogram()+
  xlab("Number of household members")
new_df %>%                       #Get the count of households rural and urban
  group_by(rural)%>%
  count()


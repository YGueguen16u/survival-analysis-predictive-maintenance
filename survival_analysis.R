install.packages("survminer")

# 1) Load packages
library(survival)   
library(survminer)  

# 2) Import CSV
data <- read.csv("survival_industry_extended.csv")
str(data)         
head(data)

# 3) Create Surv object
data$SurvObj <- with(data, Surv(time, status))

# 4) Kaplan-Meier (overall)
fit_km <- survfit(SurvObj ~ 1, data = data)
summary(fit_km)
ggsurvplot(fit_km, data = data,
           title      = "Kaplan-Meier: Overall",
           xlab       = "Hours",
           ylab       = "Survival Probability",
           risk.table = TRUE)

# 5) Kaplan-Meier by brand
fit_km_brand <- survfit(SurvObj ~ brand, data = data)
ggsurvplot(fit_km_brand, data = data,
           title      = "Kaplan-Meier by Brand",
           pval       = TRUE,      
           risk.table = TRUE)

# 6) Log-rank test
res_logrank <- survdiff(SurvObj ~ brand, data = data)
res_logrank

# 7) Cox Proportional Hazards
cox_model <- coxph(SurvObj ~ brand + usage_rate + temp_avg +
                     age_machine + env_humidity + maintenance_freq +
                     operator_experience + shock_events,
                   data = data)
summary(cox_model)

# 8) Proportional Hazards check (optional)
cox_zph <- cox.zph(cox_model)
cox_zph
plot(cox_zph) 


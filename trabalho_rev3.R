## Lending Club Loan Data  =======================================

# https://www.kaggle.com/wendykan/lending-club-loan-data

# questão: quais clientes terão atrasos no pagamento de seus empréstimos bancários  >>> LOAN STATUS

# Y = loan_status

# classificar todas as entradas "late", "charged off" e "default" como sendo 1, e as outras como 0.
# O problema economico que voces querem resolver é melhorar a politica de crédito do banco, 
# de modo que ele não conceda mais empréstimos para mal pagadores, visando dessa forma economizar possíveis perdas.

## PACOTES ========================================================
library(vroom) # leitura de dados
library(plyr)
library(dplyr)
library(tidyr)
library(esquisse)
library(ggplot2)
library(caret)
library(corrplot) # matriz de correlação
library(Hmisc) # imput missing
library(psych) # correlação 
# devtools::install_github("laresbernardo/lares")
library(lares) # correlação - bastante específico
library(imputeTS) #imputacao de missing
library(e1071)


## CARREGANDO DADOS ===============================================

setwd("C:\\Users\\Daniel\\Documents\\BI - MASTER (MBA)\\DM\\TRABALHO")

df <- vroom("loan.csv") 

#sumary(df)

summary(as.factor(df$loan_status))

prop.table(table(df$loan_status)) #verificando a proporcao de 

#estatus dos pagamentos

ggplot(dff) +
  aes(x = loan_status) +
  geom_bar() +
  scale_fill_hue() +
  theme_minimal()

# Gera aleatoriamente os índices para base final (50% da base original, por exemplo)
indexes = createDataPartition(df$loan_status, p=0.01, list = FALSE)   # amostragem estratificada
dff = df[indexes,]

prop.table(table(dff$loan_status))

## AVALIANDO MISSING VALUE N.1 ====================================

sapply(dff, function(x) sum(is.na(x)))

## Removendo col > 25% NA ==========================================

# função para remover colunas de X com mais de sig% NA
remov_col <- function(data=x, sig=y){
  miss = c()
  for(i in 1:ncol(data)) {
    if(length(which(is.na(data[,i]))) > sig*nrow(data)) miss <- append(miss,i) 
  }
  data <- data[,-miss]
}

df2 <- remov_col(data = dff, sig = 0.25)

str(df2)

sapply(df2, function(x) sum(is.na(x)))

## ALTERANDO VARIAVEIS ============================================

# alterando de character para factor
df2 <- df2 %>%  mutate_each(funs(as.factor(.)), names(.[,sapply(., is.character)]))

# alterando de inteiro para numerico
df2 <- df2 %>%  mutate_each(funs(as.numeric(.)), names(.[,sapply(., is.integer)]))

summary(df2)

## SEPARANDO EM TEEINO E TESTE =====================================

# Divisão em base de treino e teste balanceado
# install.packages('caTools')
library(caTools)
set.seed(123) #semente para reprodução de resultados
split <-  sample.split(df2$loan_status, SplitRatio = 0.7)
train <-  subset(df2, split == TRUE)
test <-  subset(df2, split == FALSE)

str(train)
head(train)

## PREPARACAO BASE DE TREINO ==========================

#avaliando missing
sapply(train, function(x) sum(is.na(x)))

## MISSING n.2  =======================================

# imputando a media em missing  numericos ##
train <- as.data.frame(train)
test <- as.data.frame(test)

# funcao de imput missing values (mean/median/min)

train[train == ""] <- NA 

impute <- function(data, type) {
  for (i in which(sapply(data, is.numeric))) {
    data[is.na(data[, i]), i] <- type(data[, i],  na.rm = TRUE)
  }
  return(data)}

train <- impute(train,median)

Mode <- function(x) { 
  ux <- sort(unique(x))
  ux[which.max(tabulate(match(x, ux)))] 
}

i1 <- !sapply(train, is.numeric)

train[i1] <- lapply(train[i1], function(x)
  replace(x, is.na(x), Mode(x[!is.na(x)])))

head(train)

# verificação das colunas "NA"
train %>%
  select_if(~ !any(is.na(.))) %>% names() # ver colunas que não tem NA

train %>% 
  select_if(~ any(is.na(.))) %>% names()  # ver colunas que tem NA

str(train)

# numero muito grande de fatores
excluir <- c("emp_title", "title", "zip_code", "earliest_cr_line")

train <- train[,!(names(train) %in% excluir)]   
test <-  test[,!(names(test) %in% excluir)]             

#Y = loan_status
summary(a$loan_status)

#deletando os itens com poucos 
train <- dplyr::filter(train, train$loan_status != "Default")

test <- dplyr::filter(test, test$loan_status != "Default")

# classificar todas as entradas "late", "charged off" e "charged off" como sendo 1, e as outras como 0.

#treino
levels(train$loan_status)[levels(train$loan_status) %in% c("Current", "Fully Paid", "In Grace Period", "Default", 
                                                           "Does not meet the credit policy. Status:Fully Paid")] <- "0"
levels(train$loan_status)[levels(train$loan_status) %in% c("Charged Off", "Late (31-120 days)", "Late (16-30 days)",
                                                           "Does not meet the credit policy. Status:Charged Off")] <- "1"
#teste
levels(test$loan_status)[levels(test$loan_status) %in% c("Current", "Fully Paid", "In Grace Period", "Default", 
                                                           "Does not meet the credit policy. Status:Fully Paid")] <- "0"
levels(test$loan_status)[levels(test$loan_status) %in% c("Charged Off", "Late (31-120 days)", "Late (16-30 days)",
                                                           "Does not meet the credit policy. Status:Charged Off")] <- "1"

##### ZERO VARIANCIA =============================

#procura por atributos (colunas) com variância 0
nearZeroVarianceIndexes = nearZeroVar(train)
train = train[,-nearZeroVarianceIndexes]

#remover atributos da base de teste tb
test = test[,-nearZeroVarianceIndexes]


## NORMALIZANDO ======================================

# normalizando os numericos ####  AINDA NÃO

normalize <- function(x){
  return( (x-min(x)) / (max(x)-min(x)))
}

train_n <-  rose_train %>%
  mutate_if(is.numeric, normalize)

str(train_n)

sapply(train_n, function(x) sum(is.na(x)))


# reduzir dimensionalidade #####

## Reduzindo dimensionalidade com Random Forest
library(Boruta)

boruta_output <- Boruta(loan_status ~ ., data=train, doTrace=0) 

# Get significant variables including tentatives
boruta_signif <- getSelectedAttributes(boruta_output, withTentative = TRUE)
print(boruta_signif)  

# Do a tentative rough fix
roughFixMod <- TentativeRoughFix(boruta_output)
boruta_signif <- getSelectedAttributes(roughFixMod)
print(boruta_signif)

# Variable Importance Scores
imps <- attStats(roughFixMod)
imps2 = imps[imps$decision != 'Rejected', c('meanImp', 'decision')]
(imps2[order(-imps2$meanImp), ])  # descending sort

# Plot variable importance
plot(boruta_output, cex.axis=.7, las=2, xlab="", main="Variable Importance") 

# Remoção das rejeitadas importancia de 15 para cima.

excluir2 <- c("term", 
              "sub_grade", 
              "grade", 
              "revol_bal", 
              "bc_open_to_buy", 
              "total_rev_hi_lim",            
              "total_bc_limit",              
              "acc_open_past_24mths",        
              "revol_util",                  
              "total_acc",                   
              "tot_hi_cred_lim",             
              "num_op_rev_tl",               
              "num_rev_accts",              
              "num_bc_tl",                   
              "annual_inc",                  
              "num_bc_sats",                 
              "bc_util",                     
              "open_acc",                  
              "issue_d",                     
              "num_actv_rev_tl",            
              "num_sats",                    
              "dti",                         
              "num_actv_bc_tl",               
              "tot_cur_bal",                
              "avg_cur_bal",                 
              "total_bal_ex_mort",           
              "percent_bc_gt_75",            
              "num_rev_tl_bal_gt_0",         
              "total_il_high_credit_limit",  
              "initial_list_status",         
              "num_tl_op_past_12m",          
              "mo_sin_rcnt_rev_tl_op",       
              "mo_sin_old_rev_tl_op",        
              "mo_sin_rcnt_tl",              
              "mths_since_recent_bc",        
              "application_type",            
              "num_il_tl",                 
              "mort_acc",                    
              "verification_status",         
              "inq_last_6mths",             
              "home_ownership")


train <- train[,!(names(train) %in% excluir2)]   
test <-  test[,!(names(test) %in% excluir2)]


str(train)
## BALANCEMANTO DA BASE DE TREINO ================

# verificando o desbalanceamento
ggplot(train) +
  aes(x = loan_status) +
  geom_bar() +
  scale_fill_hue() +
  theme_minimal()

# metodo down
set.seed(123)
down_train <- downSample(x = train[, -ncol(train)],
                         y = train$loan_status)
table(down_train$loan_status)   

#metodo up
set.seed(123)
up_train <- upSample(x = train[, -ncol(train)],
                     y = train$loan_status)                         
table(up_train$loan_status) 

#metodo Rose
library(ROSE)
set.seed(123)
rose_train <- ROSE(loan_status ~ ., data  = train)$data                         
table(rose_train$loan_status)

# metodo smote  # super lento

#install.packages(DMwR)
#library(DMwR)
balance = table(train[,length(train)])
majorityClass = balance[1]
minorityclass = balance[2]
#calculo para definir o aumento no perc_over sem alterar a classe majoritaria
percent_over = 400 # 200 ou 1306
percent_under = majorityClass*100 / ((percent_over/100)*minorityclass)
smote_train <- SMOTE(loan_status~., train, perc.over = percent_over, perc.under = percent_under)

table(train_smote$loan_status)

# 100 é pouco, ver desbalanceamento

#### Aplica PCA para seleção de atributos ####
pcatrain <-  rose_train[,-c(7)]
pcatest <-  test[,-c(7)]

#normalizacao + pca
preprocessParams  = preProcess(pcatrain, method=c("range", "pca"), thresh=0.99)
print(preprocessParams)
pcatrain = predict(preprocessParams, pcatrain)
pcatest = predict(preprocessParams, pcatest)

#adicionar a coluna de classe
#pcatrain = cbind(pcatrain, rose_train[,c(7)])
#pcatest = cbind(pcatest, test[,c(7)])


##### One-Hot Encoding  ===================================
# treino
#loan_status <- as.numeric(pcatrain$loan_status)

#b <-  pcatrain[-loan_status]

# Creating dummy variables is converting a categorical variable to as many binary variables as here are categories.
dummies_model <- dummyVars( ~ ., data=pcatrain, fullRank = T)

# Create the dummy variables using predict. The Y variable (Purchase) will not be present in trainData_mat.
train_mat <- data.frame(predict(dummies_model, newdata = pcatrain))

# # See the structure of the new dataset
str(train_mat)

loan_status <- as.data.frame(rose_train[, c(7)])

pcatrain_d <- cbind(loan_status, train_mat)

names(pcatrain_d)[names(pcatrain_d) == "rose_train[, c(7)]"] <- "loan_status"

str(pcatrain_d)

# teste
#loan_status <- as.numeric(pcatest$loan_status)

#c <- pcatest %>% select(-loan_status)


# Creating dummy variables is converting a categorical variable to as many binary variables as here are categories.
dummies_model <- dummyVars( ~ ., data=pcatest, fullRank = T)

# Create the dummy variables using predict. The Y variable (Purchase) will not be present in trainData_mat.
test_mat <- data.frame(predict(dummies_model, newdata = pcatest))

# # See the structure of the new dataset
str(train_mat)

loan_status <- as.data.frame(test[, c(7)])

pcatest_d <- cbind(loan_status, test_mat)

names(pcatest_d)[names(pcatest_d) == "test[, c(7)]"] <- "loan_status"

str(pcatest_d)

#herificando nome das colunas
colnames(pcatest_d)

sapply(pcatest_d, function(x) sum(is.na(x)))

t = na.omit(pcatest_d)

#### MODELOS =======================================

#### SVM 

system.time(svm_model <- svm(loan_status ~., pcatrain_d, probability =T))
predictionsSVM <- predict(svm_model, t, probability =T)
cm = table(predictionsSVM, t$loan_status); cm
acuracy = 1 - mean(predictionsSVM != t$loan_status)
acuracy

# kappa
confusionMatrix(cm)$overall[2]

# olhar confianca nas inferencias!
probabilidades = attr(predictionsSVM, "probabilities")
concat = cbind(t$loan_status, predictionsSVM, probabilidades)
View(concat) #confianca e previsões

#https://blog.revolutionanalytics.com/2016/08/roc-curves-in-two-lines-of-code.html  CURVA ROC


#### Regressao Logistica 

classifier = glm(formula = loan_status ~ .,
                 family = binomial,
                 data = pcatrain_d)

# Inferência ####
prob_pred = predict(classifier, type = 'response', newdata = t)
y_pred = ifelse(prob_pred > 0.5, 1, 0)

cm = table(t$loan_status, y_pred > 0.5); cm
accuracy = 1 - mean(y_pred != t$loan_status)
accuracy


summary(classifier)

plot(classifier)

##### KNN 
library(class)

system.time(knn_predict <- knn(pcatrain_d, 
                               t, 
                               cl=pcatrain_d$loan_status, k = 5))

#o teste já foi passado no treinamento. 
#a resposta do modelo já são as previsões
table(knn_predict, t$loan_status)
accuracy = 1 - mean(knn_predict != t$loan_status)
accuracy

###### Decision Tree 

tree_model <- tree(CLASSE ~., train)
predictionsDtree <- predict(tree_model, test, type = "class")
table(predictionsDtree, test$CLASSE)
acuracy = 1 - mean(predictionsDtree != test$CLASSE)
acuracy
summary(tree_model)
plot(tree_model)
text(tree_model)

#####  Random Forest 

forest_model <- randomForest(CLASSE ~., data = train, 
                             importance = TRUE, 
                             do.trace = 100)
predictionsForest = predict(forest_model, test)
table(predictionsForest, test$CLASSE)
acuracy = 1 - mean(predictionsForest != test$CLASSE)
acuracy
forest_model

#Plot dos erros
plot(forest_model)
legend("topright", legend=c("OOB", "0", "1"),
       col=c("black", "red", "green"), lty=1:1, cex=0.8)
#lty = line type, cex = character expansion factor

#Duas medidas de importância para rankear os atributos
varImpPlot(forest_model)


############## RECALL ##################################
#Salvar modelo para utilizá-lo quando chegarem novos dados
save(svm_model, file = 'svm_model')


# MODELO EM PRODUCAO ==========================================

# 1. Carregar base de dados a ser inferida
setwd("C://Users//Daniel//Documents//BI - MASTER (MBA)//SISTEMAS DE APOIO À DECISÃO//DM//AULA_03//EXERCICIO_MUSH")
Mushroom = read.table("mushrooms_sem_rotulo.txt", header = TRUE)

# 2. Carregar modelo treinado
load("forest_model") #carregando modelo já treinado

# 3. Inferir novos dados
predictions <- predict(forest_model, Mushroom)

# 4. Concatenacao das inferencias com os dados
base_inferida <- cbind(Mushroom, predictions)





## Analise exploratoria dos daddos 1a. parte ======================

esquisser(df2)

#estatus dos pagamentos

ggplot(dff) +
  aes(x = loan_status) +
  geom_bar() +
  scale_fill_hue() +
  theme_minimal()

#proposito dos emprestimos
ggplot(df2) +
  aes(x = purpose) +
  geom_bar() +
  scale_fill_hue() +
  theme_minimal()

#taxa de juros
ggplot(df2) +
  aes(x = int_rate) +
  geom_density(adjust = 0.8, fill = "#0c4c8a") +
  theme_minimal()

#meses de inicio do emprestimo
ggplot(df2) +
  aes(x = issue_d, fill = term) +
  geom_bar(position = "dodge") +
  scale_fill_brewer(palette = "Paired") +
  theme_minimal()


df2 %>%
  filter(!(issue_d %in% "%Y")) %>%
  ggplot() +
  aes(x = issue_d, fill = term) +
  geom_bar(position = "dodge") +
  scale_fill_brewer(palette = "Paired") +
  theme_minimal()

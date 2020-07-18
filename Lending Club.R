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
library(ggplot2) #grafico
library(cowplot) #grafico
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

# funcao de imput missing values (mean/median/min)
df2[df2 == ""] <- NA #caso haja valor vazio, fica NA

df2 <- as.data.frame(df2)

impute <- function(data, type) {
  for (i in which(sapply(data, is.numeric))) {
    data[is.na(data[, i]), i] <- type(data[, i],  na.rm = TRUE)
  }
  return(data)}

df2 <- impute(df2, median)

Mode <- function(x) { 
  ux <- sort(unique(x))
  ux[which.max(tabulate(match(x, ux)))] 
}

i1 <- !sapply(df2, is.numeric)

df2[i1] <- lapply(df2[i1], function(x)
  replace(x, is.na(x), Mode(x[!is.na(x)])))


head(df2)

sapply(df2, function(x) sum(is.na(x)))


# verificação das colunas "NA"
df2 %>%
  select_if(~ !any(is.na(.))) %>% names() # ver colunas que não tem NA

df2 %>% 
  select_if(~ any(is.na(.))) %>% names()  # ver colunas que tem NA


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

#Y = loan_status
summary(train$loan_status)

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

##### REDUCAO DE DIMENSIONALIDADE =============================

# numero muito grande de fatores
excluir <- c("emp_title", "title", "zip_code", "earliest_cr_line")

train <- train[,!(names(train) %in% excluir)]   
test <-  test[,!(names(test) %in% excluir)]  

## zERO vARIANCIA ####

#procura por atributos (colunas) com variância 0
nearZeroVarianceIndexes = nearZeroVar(train)
train = train[,-nearZeroVarianceIndexes]

#remover atributos da base de teste tb
test = test[,-nearZeroVarianceIndexes]

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

#metodo Rose
library(ROSE)
set.seed(123)
rose_train <- ROSE(loan_status ~ ., data  = train)$data                         
table(rose_train$loan_status)

#### Sem aplicar PCA =========================

nortrain <-  rose_train[,-c(7)]
nortest <-  test[,-c(7)]

#normalizacao + pca
preprocessParams  = preProcess(nortrain, method=c("range"), thresh=0.99)
print(preprocessParams)
nortrain = predict(preprocessParams, nortrain)
nortest = predict(preprocessParams, nortest)

##### One-Hot Encoding  ===================================
# treino
# Criando as variaveis ficticias
dummies_model <- dummyVars( ~ ., data=nortrain, fullRank = T)

# criando as dummies.
train_mat <- data.frame(predict(dummies_model, newdata = nortrain))

# # See the structure of the new dataset
str(train_mat)

loan_status <- as.data.frame(rose_train[, c(7)])

nortrain_d <- cbind(loan_status, train_mat)

names(nortrain_d)[names(nortrain_d) == "rose_train[, c(7)]"] <- "loan_status"

str(nortrain_d)

# teste
# Creating dummy variables is converting a categorical variable to as many binary variables as here are categories.
dummies_model <- dummyVars( ~ ., data=nortest, fullRank = T)

# Create the dummy variables using predict. The Y variable (Purchase) will not be present in trainData_mat.
test_mat <- data.frame(predict(dummies_model, newdata = nortest))

# # See the structure of the new dataset
str(train_mat)

loan_status <- as.data.frame(test[, c(7)])

nortest_d <- cbind(loan_status, test_mat)

names(nortest_d)[names(nortest_d) == "test[, c(7)]"] <- "loan_status"

str(nortest_d)


#### Aplica PCA para seleção de atributos ====
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

#verificando nome das colunas
colnames(pcatest_d)

sapply(pcatest_d, function(x) sum(is.na(x)))


#### MODELOS =======================================

#### SVM ########

system.time(svm_model <- svm(loan_status ~., nortrain_d, probability=T, metric="kappa"))
predictionsSVM <- predict(svm_model, nortest_d, probability =T)


# inferencia
cm_svm = confusionMatrix(predictionsSVM, nortest_d$loan_status, positive = NULL, 
                         dnn = c("Prediction", "Reference"))
cm_svm

#### Regressao Logistica ########

glm_model <-  glm(formula = loan_status ~ .,
                 family = binomial,
                 data = nortrain_d)

# Inferência
prob_pred <-  predict(glm_model, type = 'response', newdata = nortest_d)
y_pred <-  as.factor(ifelse(prob_pred > 0.5, 1, 0))

cm_glm = confusionMatrix(y_pred, nortest_d$loan_status, positive = NULL, 
                         dnn = c("Prediction", "Reference"))
cm_glm


#### KNN ################
library(class)
                        
# sem PCA
system.time(knn_model <- knn(nortrain_d, 
                               nortest_d, 
                               cl=nortrain_d$loan_status, k = 40))

# inferencia
cm_knn = confusionMatrix(knn_model, nortest_d$loan_status, positive = NULL, 
                         dnn = c("Prediction", "Reference"))
cm_knn


# com PCA
system.time(knn_predict <- knn(pcatrain_d, 
                               pcatest_d, 
                               cl=pcatrain_d$loan_status, k = 40))

# inferencia
cm_knnp = confusionMatrix(knn_predict, nortest_d$loan_status, positive = NULL, 
                         dnn = c("Prediction", "Reference"))
cm_knnp

###### Decision Tree  ########
library(tree)

tree_model <- tree(loan_status ~., nortrain_d)
predictionsDtree <- predict(tree_model, nortest_d, type = "class")

# inferencia
cm_tree = confusionMatrix(predictionsDtree, nortest_d$loan_status, positive = NULL, 
                         dnn = c("Prediction", "Reference"))
cm_tree

# plot
summary(tree_model)
plot(tree_model)
text(tree_model)

#####  Random Forest  ########
library(randomForest)

forest_model <- randomForest(loan_status ~., data = nortrain_d, 
                             importance = TRUE, 
                             do.trace = 100,
                             ntree = 300)

predictionsForest = predict(forest_model, nortest_d)

# inferencia
cm_rf = confusionMatrix(predictionsForest, nortest_d$loan_status, positive = NULL, 
                          dnn = c("Prediction", "Reference"))
cm_rf


#Plot dos erros
plot(forest_model)
legend("topright", legend=c("OOB", "0", "1"),
       col=c("black", "red", "green"), lty=1:1, cex=0.8)
#lty = line type, cex = character expansion factor

#Duas medidas de importância para rankear os atributos
varImpPlot(forest_model)


############## RECALL ##################################
#Salvar modelo para utilizá-lo quando chegarem novos dados
#save(svm_model, file = 'svm_model')


# MODELO EM PRODUCAO ==========================================

# 1. Carregar base de dados a ser inferida
#SET DADOS
#NOVOS DADOS = XXXX

# 2. Carregar modelo treinado
#load("svm_model") #carregando modelo já treinado

# 3. Inferir novos dados
#predictions_new <- predict(svm_model, NOVOS DADOS)

# 4. Concatenacao das inferencias com os dados
#base_inferida <- cbind(NOVOS DADOS, predictions_new)


## Analise exploratoria dos daddos 1a. parte ======================

#estatus dos pagamentos
ggplot(dff)+
  aes(x=loan_status) + 
  geom_bar(width=.7, fill="tomato3") + 
  theme(axis.text.x = element_text(angle=65, vjust=0.7))

#estatus da divida x valor (box-plot)
ggplot(dff) +
  aes(x = loan_status, y = loan_amnt, fill = loan_status) +
  geom_boxplot() +
  scale_fill_brewer(palette = "Accent") +
  labs(x = "Estatus da dívida", y = "Valor", fill = "Estatus da dívida") +
  theme_minimal()


#proposito dos emprestimos
ggplot(dff)+
  aes(x=purpose) + 
  geom_bar(width=.7, fill="tomato3") + 
  theme(axis.text.x = element_text(angle=65, vjust=0.7))

#quantidade por tempo e 
a <- ggplot(dff)+
  aes(term, loan_amnt)+
  geom_boxplot(varwidth=T) + 
  labs(x="meses",
       y="valor do empréstimo")

#taxa de juros por tenpo
b <- ggplot(dff)+
  aes(term, int_rate)+
  geom_boxplot(varwidth=T) + 
  labs(x="meses",
       y="taxa de juros")

plot_grid(a, b, labels = "AUTO")

#taxa de juros
ggplot(dff) +
  aes(x = int_rate) +
  geom_density(adjust = 0.8, fill = "#0c4c8a") +
  theme_minimal()

#meses de inicio do emprestimo
ggplot(dff) +
  aes(x = issue_d, fill = term) +
  geom_bar(position = "dodge") +
  scale_fill_brewer(palette = "Paired") +
  theme_minimal()

#taxa de juros por grade
ggplot(dff) +
  aes(x = grade, y = int_rate, fill = application_type) +
  geom_boxplot() +
  scale_fill_hue() +
  theme_minimal()
  labs(x = "Grade", y = "taxa de juros", fill = "Tipo de aplicação") +

#box plot por propósitos
ggplot(dff) +
  aes(x = purpose, y = loan_amnt, fill = purpose) +
  geom_boxplot() +
  labs(x = "propósito", y = "valor") +
  theme_minimal()
  
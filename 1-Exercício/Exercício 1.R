# Carregando os pacotes
library(tidyverse)
library(ggplot2)
library(caret)

# Leitura do dataset (exemplo: Pima Indians Diabetes dataset)
# Pode ser substituído por outro dataset relacionado à saúde.
url <- "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
data <- read.csv(url)
# Definindo os nomes das colunas
colnames(data) <- c('num_preg', 'glucose', 'blood_pressure', 'skin_thickness', 'insulin', 'bmi', 'diabetes_pedigree', 'age', 'outcome')

# Examinando as primeiras linhas do dataset
head(data)

# Análise Descritiva
summary(data)

# Visualização: Histograma do nível de glicose
ggplot(data, aes(x=glucose)) + 
  geom_histogram(binwidth=10, fill="blue", color="black") +
  labs(title="Distribuição do Nível de Glicose", x="Glicose", y="Frequência")

# Visualização: Gráfico de dispersão entre Idade e IMC
ggplot(data, aes(x=age, y=bmi)) + 
  geom_point() +
  labs(title="Relação entre Idade e IMC", x="Idade", y="IMC")

# Modelo Explicativo: Regressão logística para prever a ocorrência de diabetes
model <- glm(outcome ~ glucose + bmi + age, data=data, family=binomial)

# Sumário do modelo
summary(model)

# Teste de correlação entre glicose e idade
cor.test(data$glucose, data$age)

# Teste de associação entre glicose e presença de diabetes (qui-quadrado)
table_glucose_diabetes <- table(data$glucose > 120, data$outcome)
chisq.test(table_glucose_diabetes)

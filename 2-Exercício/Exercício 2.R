data <- read.csv('./2-Exercício/data/heart.csv')
head(data)

summary(data$age)
summary(data$chol)
summary(data$trestbps)
summary(data$thalach)

hist(data$age, main="Distribuição da Idade", xlab="Idade")
boxplot(chol ~ target, data=data, main="Colesterol por Doença Cardíaca", xlab="Doença Cardíaca", ylab="Colesterol")

cor.test(data$age, data$chol)
cor.test(data$trestbps, data$thalach)

model <- glm(target ~ age + sex + cp + chol + thalach + ca, data=data, family=binomial)
summary(model)

AIC(model)

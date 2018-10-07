
# load libraries
library(tidyverse)
library(tm)

# read in the training data set

textData <- read.csv("train.csv")


# split training data into train set and validation set
set.seed(123)
train_ind <- sample(seq_len(nrow(textData)), size = floor(0.6*nrow(textData)))
train <- textData[train_ind, ]
test <-  textData[-train_ind, ]


#-------------- Linear Model--------------------------

#===============================
# Build Models
#===============================

# aggregate all text for each user
data.aggre <- aggregate(text~user.id+gender+topic+sign+age, data=train, paste, collapse = ",")

# create corpus
new.data <- VCorpus(VectorSource(data.aggre$text))

# reduce term sparcity
mywords <- tm_map(new.data, stripWhitespace)
mywords <- tm_map(mywords, removeNumbers)
mywords <- tm_map(mywords, content_transformer(tolower))
mywords <- tm_map(mywords, removeWords, stopwords("english"))
mywords <- tm_map(mywords, stemDocument)
mywords <- tm_map(mywords, removeWords, c("urllink","nbsp"))
mywords <- tm_map(mywords, removePunctuation)

# create document term matrix
blogs.clean.tfidf = DocumentTermMatrix(mywords, control = list(weighting = weightTfIdf))
blogs.clean.tfidf

# remove sparse terms at various thresholds.
tfidf.99 = removeSparseTerms(blogs.clean.tfidf, 0.99)  
tfidf.99

tfidf.95 = removeSparseTerms(blogs.clean.tfidf, 0.95)
tfidf.95

tfidf.90 = removeSparseTerms(blogs.clean.tfidf, 0.90)
tfidf.90

tfidf.85 = removeSparseTerms(blogs.clean.tfidf, 0.85)
tfidf.85



# Model 1: Use tfidf.85

# bind term matrix to aggregated train data
new.training <-cbind(data.aggre[,-6], as.matrix(tfidf.85))
colnames(new.training)[c(2,3,4,5)] <- c(c("Usergender","Usertopic","Usersign","Userage")) 

cols <- c("Usergender","Usertopic","Usersign")
new.training[cols] <- lapply(new.training[cols], factor)
#colnames(new.training)



lm <- lm(Userage ~ .-(user.id),data = new.training)
summary(lm)
# R2=0.5458



# Model 2: remove usertopic and Usersign in train data

lm.2 <- lm(Userage ~. -(user.id)-(Usertopic)-(Usersign),data = new.training)
summary(lm.2)
# R2= 0.4955



# Model 3: use tfidf.90

# bind term matrix to aggregated train data
new.training.2 <-cbind(data.aggre[,-6], as.matrix(tfidf.90))

colnames(new.training.2)[c(2,3,4,5)] <- c(c("Usergender","Usertopic","Usersign","Userage")) 

cols <- c("Usergender","Usertopic","Usersign")
new.training.2[cols] <- lapply(new.training.2[cols], factor)

                                                 
lm.3 <- lm(Userage ~ .-(user.id),data = new.training.2)
summary(lm.3) # R.2 increased to 0.5988


# Model 4: 
# try other weighting methods

blogs.clean.smart = DocumentTermMatrix(mywords, control = list(weighting = function (x) weightSMART(x,spec="ltc")))
blogs.clean.smart


# Remove sparse terms at various thresholds.
#smart.99 = removeSparseTerms(blogs.clean.smart, 0.99)  # remove terms that are absent from at least 99% of documents (keep most terms)
#smart.99

smart.95 = removeSparseTerms(blogs.clean.smart, 0.95)
smart.95

smart.90 = removeSparseTerms(blogs.clean.smart, 0.90)
smart.90

smart.85 = removeSparseTerms(blogs.clean.smart, 0.85)
smart.85

# try 0.85 removeSParseTerms
# bind term matrix to aggregated train data
new.training.4 <-cbind(data.aggre[,-6], as.matrix(smart.85))

colnames(new.training.4)[c(2,3,4,5)] <- c(c("Usergender","Usertopic","Usersign","Userage")) 

cols <- c("Usergender","Usertopic","Usersign")
new.training.4[cols] <- lapply(new.training.4[cols], factor)

lm.4 <- lm(Userage ~ .-(user.id),data = new.training.4)
summary(lm.4)
# Residual standard error: 5.411 on 11487 degrees of freedom
# Multiple R-squared:  0.5749,	Adjusted R-squared:  0.538 
# F-statistic: 15.58 on 997 and 11487 DF,  p-value: < 2.2e-16



smart.80 = removeSparseTerms(blogs.clean.smart, 0.80)
smart.80

new.training.5 <-cbind(data.aggre[,-6], as.matrix(smart.80))

colnames(new.training.5)[c(2,3,4,5)] <- c(c("Usergender","Usertopic","Usersign","Userage")) 

cols <- c("Usergender","Usertopic","Usersign")
new.training.5[cols] <- lapply(new.training.5[cols], factor)

lm.5 <- lm(Userage ~ .-(user.id),data = new.training.5)
summary(lm.5)



# =============================================
# cross validate on test (40% of original train)
# =============================================

# aggregate all text for each user
valid.aggre <- aggregate(text~user.id+gender+topic+sign+age, data=test, paste, collapse = ",")

# create corpus
new.valid <- VCorpus(VectorSource(valid.aggre$text))

# reduce term sparcity
mywords.valid <- tm_map(new.valid, stripWhitespace)
mywords.valid <- tm_map(mywords.valid, removeNumbers)
mywords.valid <- tm_map(mywords.valid, content_transformer(tolower))
mywords.valid <- tm_map(mywords.valid, removeWords, stopwords("english"))
mywords.valid <- tm_map(mywords.valid, stemDocument)
mywords.valid <- tm_map(mywords.valid, removeWords, c("urllink","nbsp"))
mywords.valid <- tm_map(mywords.valid, removePunctuation)

# create document term matrix using tfidf weighting
valid.clean.tfidf = DocumentTermMatrix(mywords.valid, control = list(weighting = weightTfIdf))
valid.clean.tfidf

#Remove sparse terms at various thresholds.
tfidf.99.valid= removeSparseTerms(valid.clean.tfidf, 0.99)  # remove terms that are absent from at least 99% of documents (keep most terms)
tfidf.99.valid

tfidf.95.valid = removeSparseTerms(valid.clean.tfidf, 0.95)
tfidf.95.valid

tfidf.90.valid = removeSparseTerms(valid.clean.tfidf, 0.90)
tfidf.90.valid

tfidf.85.valid = removeSparseTerms(valid.clean.tfidf, 0.85)
tfidf.85.valid


# To use lm and lm.2 on validation set

# bind term matrix to aggregated train data
new.valid <-cbind(valid.aggre[,-6], as.matrix(tfidf.85.valid))

colnames(new.valid)[c(2,3,4,5)] <- c(c("Usergender","Usertopic","Usersign","Userage")) 

cols <- c("Usergender","Usertopic","Usersign")
new.valid[cols] <- lapply(new.valid[cols], factor)
#colnames(new.valid)

# find all terms in train and validation sets
train.terms <- colnames(new.training)[-1:-5]
valid.terms <- colnames(new.valid)[-1:-5]

# find terms in train but not in validation
terms.to.add.valid <- train.terms[!(train.terms %in% valid.terms)]
# add terms to validation set
new.valid[terms.to.add.valid] <- 0 

# find terms in validation set but not in train
terms.to.del.valid <- valid.terms[!(valid.terms %in% train.terms)]
terms.to.del.valid
# character(0) 
# no terms in valid set but not in train - no further action
# new.valid.2 <- select(new.valid,-terms.to.del.valid)

# predict on validation set using lm

pred.age.valid <- predict(lm, newdata=new.valid)

# calculate MAE
MAE <- mean(abs(pred.age.valid - new.valid$Userage))
MAE 
#[1] 4.737965    when using tfidf 
#[1] 24.36753    when using weightSMART

# predict on validation set using lm.2

pred.age.valid.2 <- predict(lm.2, newdata=new.valid)

# calculate MAE
MAE <- mean(abs(pred.age.valid.2 - new.valid$Userage))
MAE 
# [1] 5.038436
# MAE increased using lm.2


# try use lm.3 on validation set

# bind term matrix to aggregated train data
new.valid.3 <-cbind(valid.aggre[,-6], as.matrix(tfidf.90.valid))

colnames(new.valid.3)[c(2,3,4,5)] <- c(c("Usergender","Usertopic","Usersign","Userage")) 

cols <- c("Usergender","Usertopic","Usersign")
new.valid.3[cols] <- lapply(new.valid.3[cols], factor)


train.terms.3 <- colnames(new.training.2)[-1:-5]
valid.terms.3 <- colnames(new.valid.3)[-1:-5]

# find terms in train but not in validation
terms.to.add.valid.3 <- train.terms.3[!(train.terms.3 %in% valid.terms.3)]
terms.to.add.valid.3

# add terms to validation set
new.valid.3[terms.to.add.valid.3] <- 0 

# find terms in validation set but not in train
terms.to.del.valid.3 <- valid.terms.3[!(valid.terms.3 %in% train.terms.3)]
terms.to.del.valid.3
# character(0) - no further action
#new.valid.3 <- select(new.valid.3,-terms.to.del.valid.3)

# predict on validation set

pred.age.valid.3 <- predict(lm.3, newdata=new.valid.3)

# calculate MAE
MAE <- mean(abs(pred.age.valid.3 - new.valid.3$Userage))
MAE 
# [1] 4.745015



# try lm.4 (weightsmart)


valid.clean.smart = DocumentTermMatrix(mywords.valid, control = list(weighting = function (x) weightSMART(x,spec="ltc")))
valid.clean.smart


# Remove sparse terms at various thresholds.
#smart.99 = removeSparseTerms(blogs.clean.smart, 0.99)  # remove terms that are absent from at least 99% of documents (keep most terms)
#smart.99

smart.95.valid = removeSparseTerms(valid.clean.smart, 0.95)
smart.95.valid

smart.90.valid = removeSparseTerms(valid.clean.smart, 0.90)
smart.90.valid

smart.85.valid = removeSparseTerms(valid.clean.smart, 0.85)
smart.85.valid

# bind term matrix to aggregated train data
new.valid.4 <-cbind(valid.aggre[,-6], as.matrix(smart.85.valid))

colnames(new.valid.4)[c(2,3,4,5)] <- c(c("Usergender","Usertopic","Usersign","Userage")) 

cols <- c("Usergender","Usertopic","Usersign")
new.valid.4[cols] <- lapply(new.valid.4[cols], factor)


train.terms.4 <- colnames(new.training.4)[-1:-5]
valid.terms.4 <- colnames(new.valid.4)[-1:-5]

# find terms in train but not in validation
terms.to.add.valid.4 <- train.terms.4[!(train.terms.4 %in% valid.terms.4)]
terms.to.add.valid.4

# add terms to validation set
new.valid.4[terms.to.add.valid.4] <- 0 

# find terms in validation set but not in train
terms.to.del.valid.4 <- valid.terms.4[!(valid.terms.4 %in% train.terms.4)]
terms.to.del.valid.4
# character(0) - no further action
#new.valid.4 <- select(new.valid.4,-terms.to.del.valid.4)

# predict on validation set

pred.age.valid.4 <- predict(lm.4, newdata=new.valid.4)

# calculate MAE
MAE <- mean(abs(pred.age.valid.4 - new.valid.4$Userage))
MAE 
# [1] 4.428047


# try lm.5

smart.80.valid = removeSparseTerms(valid.clean.smart, 0.80)

# bind term matrix to aggregated train data
new.valid.5 <-cbind(valid.aggre[,-6], as.matrix(smart.80.valid))

colnames(new.valid.5)[c(2,3,4,5)] <- c(c("Usergender","Usertopic","Usersign","Userage")) 

cols <- c("Usergender","Usertopic","Usersign")
new.valid.5[cols] <- lapply(new.valid.5[cols], factor)


train.terms.5 <- colnames(new.training.5)[-1:-5]
valid.terms.5 <- colnames(new.valid.5)[-1:-5]

# find terms in train but not in validation
terms.to.add.valid.5 <- train.terms.5[!(train.terms.5 %in% valid.terms.5)]
terms.to.add.valid.5

# add terms to validation set
new.valid.5[terms.to.add.valid.5] <- 0 

# find terms in validation set but not in train
terms.to.del.valid.5 <- valid.terms.5[!(valid.terms.5 %in% train.terms.5)]
terms.to.del.valid.5
# character(0) - no further action
#new.valid.4 <- select(new.valid.4,-terms.to.del.valid.4)

# predict on validation set

pred.age.valid.5 <- predict(lm.5, newdata=new.valid.5)

# calculate MAE
MAE <- mean(abs(pred.age.valid.5 - new.valid.5$Userage))
MAE 
# [1] 4.433363

# It seems lm.4 is the best.

# ========================
# Apply model on test data
# ========================

testData <- read.csv("test.csv")

# Create corpus
test.aggre <- aggregate(text~user.id+gender+topic+sign, data=testData, paste, collapse = ",")
new.test <- VCorpus(VectorSource(test.aggre$text))

# reduce term sparcity
mywords.test <- tm_map(new.test, stripWhitespace)
mywords.test <- tm_map(mywords.test, removeNumbers)
mywords.test <- tm_map(mywords.test, content_transformer(tolower))
mywords.test <- tm_map(mywords.test, removeWords, stopwords("english"))
mywords.test <- tm_map(mywords.test, stemDocument)
mywords.test <- tm_map(mywords.test, removeWords, c("urllink","nbsp"))
mywords.test <- tm_map(mywords.test, removePunctuation)

test.clean.tfidf = DocumentTermMatrix(mywords.test, control = list(weighting = weightTfIdf))
test.clean.tfidf

# Remove sparse terms at various thresholds.
#tfidf.99.test = removeSparseTerms(test.clean.tfidf, 0.99)  
#tfidf.99.test

tfidf.90.test = removeSparseTerms(test.clean.tfidf, 0.90)
tfidf.90.test

tfidf.85.test = removeSparseTerms(test.clean.tfidf, 0.85)
tfidf.85.test

# use lm on test data (tidf.85)

# bind term matrix to aggregated train data
new.test <-cbind(test.aggre[,-5], as.matrix(tfidf.85.test))
colnames(new.test)[c(2,3,4)] <- c(c("Usergender","Usertopic","Usersign")) 

cols <- c("Usergender","Usertopic","Usersign")
new.test[cols] <- lapply(new.test[cols], factor)
#colnames(new.test)

train.terms <- colnames(new.training)[-1:-5]
test.terms <- colnames(new.test)[-1:-5]


# find terms in train but not in test
terms.to.add.test <- train.terms[!(train.terms %in% test.terms)]
terms.to.add.test
# character(0)
# add terms to validation set
# new.test[terms.to.add.test] <- 0 

# find terms in test but not in train
terms.to.del.test <- test.terms[!(test.terms %in% train.terms)]
new.test <- select(new.test,-terms.to.del.test)

# predict on new.test

pred.age <- predict(lm, newdata=new.test)
pred.table <- cbind(test.aggre$user.id, pred.age)
write.table(pred.table, file="submission13.csv", row.names=F, col.names = c("user.id", "age"), sep=',')



# use lm.3 (tfidf.90)

# bind term matrix to aggregated train data
new.test.3 <-cbind(test.aggre[,-5], as.matrix(tfidf.90.test))

colnames(new.test.3)[c(2,3,4)] <- c(c("Usergender","Usertopic","Usersign")) 

cols <- c("Usergender","Usertopic","Usersign")
new.test.3[cols] <- lapply(new.test.3[cols], factor)
#colnames(new.test)

train.terms.3 <- colnames(new.training.2)[-1:-5]
test.terms.3 <- colnames(new.test.3)[-1:-5]


# find terms in train but not in test
terms.to.add.test.3 <- train.terms.3[!(train.terms.3 %in% test.terms.3)]

# add terms to validation set
new.test.3[terms.to.add.test.3] <- 0 

# find terms in test but not in train
terms.to.del.test.3 <- test.terms.3[!(test.terms.3 %in% train.terms.3)]
new.test.3 <- select(new.test.3,-terms.to.del.test.3)

# predict on new.test.2

pred.age.3 <- predict(lm.3, newdata=new.test.3)
pred.table.3 <- cbind(test.aggre$user.id, pred.age.3)
write.table(pred.table.3, file="submission14.csv", row.names=F, col.names = c("user.id", "age"), sep=',')




# use lm.4 (weightSMART)

test.clean.smart = DocumentTermMatrix(mywords.test, control = list(weighting = function (x) weightSMART(x,spec="ltc")))

smart.85.test = removeSparseTerms(test.clean.smart, 0.85)
smart.85.test

# bind term matrix to aggregated train data
new.test.4 <-cbind(test.aggre[,-5], as.matrix(smart.85.test))

colnames(new.test.4)[c(2,3,4)] <- c(c("Usergender","Usertopic","Usersign")) 

cols <- c("Usergender","Usertopic","Usersign")
new.test.4[cols] <- lapply(new.test.4[cols], factor)
#colnames(new.test)

train.terms.4 <- colnames(new.training.4)[-1:-5]
test.terms.4 <- colnames(new.test.4)[-1:-5]


# find terms in train but not in test
terms.to.add.test.4 <- train.terms.4[!(train.terms.4 %in% test.terms.4)]
terms.to.add.test.4
# character(0)
# add terms to validation set
# new.test.4[terms.to.add.test.4] <- 0 

# find terms in test but not in train
terms.to.del.test.4 <- test.terms.4[!(test.terms.4 %in% train.terms.4)]
new.test.4 <- select(new.test.4,-terms.to.del.test.4)

# predict on new.test

pred.age.4 <- predict(lm.4, newdata=new.test.4)
pred.table.4 <- cbind(test.aggre$user.id, pred.age.4)
write.table(pred.table.4, file="submission15.csv", row.names=F, col.names = c("user.id", "age"), sep=',')



#-------------------------------------------------------------




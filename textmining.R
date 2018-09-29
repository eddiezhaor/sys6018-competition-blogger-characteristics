library(tidyverse)
library(tm)
library(wordcloud)
library(quanteda)
library(glmnet)
library(caret)
#import the training data set
textData <- read.csv("../eddie/Desktop/DATA MINING/sys6018-competition-blogger-characteristics/all/train.csv")
set.seed(123)
train_ind <- sample(seq_len(nrow(textData)), size = floor(0.6*nrow(textData)))
train <- textData[train_ind, ]

test <-  textData[-train_ind, ]
corpus <- Corpus(VectorSource(train$text))
text_corpus <- corpus(corpus)
#tokenlize the set of texts 
mytokens <- tokens(text_corpus, what= "word", remove_punct =TRUE, remove_numbers = TRUE, remove_symbols=TRUE, remove_hyphens=TRUE)
mytokens <- tokens_tolower(mytokens)
#remove stopwords
mytokens <- tokens_remove(mytokens, stopwords("en"))
#apply a stemmer to words
mytokens<-tokens_wordstem(mytokens, language="english")
text_dfm <- dfm(mytokens)
docfreq_irish_dfm <- dfm_trim(text_dfm, max_docfreq = 0.1, docfreq_type = "prop",min_termfreq = 100)
top10 <- data.frame(topfeatures(text_dfm, 30))
#plot the top10 words with the highest frequency
colnames(top10) <- c("frequency")
wordcloud(rownames(top10), freq = top10$frequency,color=brewer.pal(8,"Dark2"), max.words = 50, min.freq = 1,random.order = FALSE,scale=c(2,1))
g <- ggplot(data = top10, aes(reorder(rownames(top10),top10$frequency), top10$frequency))
g+geom_bar(stat = "identity",aes(fill=rownames(top10))) + coord_flip() +labs(title="The top 30 words with the highest frequency", x="Frequency", y="Words", fill ="Words")

text_tfidf <- dfm_tfidf(text_dfm)
top10_tfidf <- topfeatures(text_tfidf, 20)
topfeatures(text_tfidf[1,],20)

data.aggre <- aggregate(text~user.id+gender+topic+sign+age, data=train, paste, collapse = ",")
new.data <- VCorpus(VectorSource(data.aggre$text))
mywords <- tm_map(new.data, stripWhitespace)
mywords <- tm_map(mywords, removeNumbers)
mywords <- tm_map(mywords, content_transformer(tolower))
mywords <- tm_map(mywords, removeWords, stopwords("english"))
mywords <- tm_map(mywords, stemDocument)
mywords <- tm_map(mywords, removeWords, c("urllink","nbsp"))
mywords <- tm_map(mywords, removePunctuation)

dtm <- DocumentTermMatrix(mywords)
dtm <- removeSparseTerms(dtm, 0.9)
dtm <- weightTfIdf(dtm, normalize = TRUE)
m <-as.matrix(dtm)

v <- sort(colSums(m), decreasing = TRUE)
df <- data.frame(v)
top10 <- data.frame(head(df, 30))
wordcloud(rownames(top10), freq = top10$v,color=brewer.pal(8,"Dark2"), max.words = 50, min.freq = 1,random.order = FALSE,scale=c(2,1))
g <- ggplot(data = top10, aes(reorder(rownames(top10),top10$v), top10$v))
g+geom_bar(stat = "identity",aes(fill=rownames(top10))) + coord_flip() +labs(title="The top 30 words with the highest frequency", x="Frequency", y="Words", fill ="Words")

#new.training <- train[,-7]

new.training <-cbind(data.aggre[,-6], data.frame(m))
age <-new.training[,5]
new.training <- new.training[,-5]
colnames(new.training)[c(2,3,4)] <- c(c("Usergender","Usertopic","Usersign")) 

cols <- c("Usergender","Usertopic","Usersign")
new.training[cols] <- lapply(new.training[cols], factor)
colnames(new.training)
dummie <- dummyVars("~.",new.training)
dummie <- data.frame(predict(dummie, newdata = new.training))

new.training <- dummie[,-1]
new.training <- new.training[,order(names(new.training))]
t <- as.matrix(new.training)

fit <- glmnet(t, as.matrix(age), alpha = 1, family = "gaussian")
cvfit = cv.glmnet(t, as.matrix(age), type.measure = "mae", nfolds = 12)
cvfit$cvm
cvfit$cvup
cvfit$cvlo
cvfit$lambda.min
cvfit$lambda.1se
plot(fit,label = TRUE)
print(fit)






testData <- read.csv("../eddie/Desktop/DATA MINING/sys6018-competition-blogger-characteristics/all/test.csv")
testData2 <- testData
test.data.aggre <- aggregate(text~user.id+gender+topic+sign, data=testData2, paste, collapse = ",")
neworder <- data.frame(user.id = unique(testData2$user.id))
test.data.aggre <- test.data.aggre[match(neworder$user.id,test.data.aggre$user.id),]
test.new.data <- VCorpus(VectorSource(test.data.aggre$text))
test.mywords <- tm_map(test.new.data, stripWhitespace)
test.mywords <- tm_map(test.mywords, removeNumbers)
test.mywords <- tm_map(test.mywords, content_transformer(tolower))
#test.mywords <- tm_map(test.mywords, removeWords, stopwords("english"))
test.mywords <- tm_map(test.mywords, stemDocument)
test.mywords <- tm_map(test.mywords, removeWords, c("urllink","nbsp"))
test.mywords <- tm_map(test.mywords, removePunctuation)
test.dtm <- DocumentTermMatrix(test.mywords)
test.dtm <- removeSparseTerms(test.dtm, 0.6)
test.dtm <- weightTfIdf(test.dtm, normalize = TRUE)
test.m <-as.matrix(test.dtm)
test.new.training <-cbind(test.data.aggre[,-5], test.m)
rownames(test.new.training) <-seq(1,nrow(test.new.training),1)
cols <- c(2, 3, 4)
test.new.training[cols] <- lapply(test.new.training[cols], factor)
colnames(test.new.training)[cols] <- c("Usergender","Usertopic","Usersign")
test.dummie <- dummyVars("~.",test.new.training)
test.dummie <- data.frame(predict(test.dummie, newdata = test.new.training))

new.test.data <- test.dummie[colnames(test.dummie) %in% colnames(new.training)]
not.exist <-new.training[!colnames(new.training) %in% colnames(test.dummie)]
etrCol <- data.frame(matrix(0, nrow =nrow(new.test.data), ncol = length(colnames(not.exist))))
colnames(etrCol) <-as.vector(colnames(not.exist))
new.test.data2 <-cbind(new.test.data, etrCol)
new.test.data2 <- new.test.data2[,order(names(new.test.data2))]
prediction <- data.frame(predict(fit, as.matrix(new.test.data2),s = 0.1077512))#0.06165922
output <- cbind(test.data.aggre$user.id, prediction)
colnames(output) <- c("user.id", "age")
#setwd("./Desktop/DATA MINING/sys6018-competition-blogger-characteristics/Eddie/")
write.csv(output, "submission6.csv",row.names=FALSE)

colnames(new.training)

new.training <- dummie[,-1]




'
test
testData2 <- test
test.data.aggre <- aggregate(text~user.id+gender+topic+sign, data=testData2, paste, collapse = ",")
neworder <- data.frame(user.id = unique(testData2$user.id))
test.data.aggre <- test.data.aggre[match(neworder$user.id,test.data.aggre$user.id),]
test.new.data <- VCorpus(VectorSource(test.data.aggre$text))
test.mywords <- tm_map(test.new.data, stripWhitespace)
test.mywords <- tm_map(test.mywords, removeNumbers)
test.mywords <- tm_map(test.mywords, content_transformer(tolower))
#test.mywords <- tm_map(test.mywords, removeWords, stopwords("english"))
test.mywords <- tm_map(test.mywords, stemDocument)
test.mywords <- tm_map(test.mywords, removeWords, c("urllink","nbsp"))
test.mywords <- tm_map(test.mywords, removePunctuation)
test.dtm <- DocumentTermMatrix(test.mywords)
test.dtm <- removeSparseTerms(test.dtm, 0.6)
test.dtm <- weightTfIdf(test.dtm, normalize = TRUE)
test.m <-as.matrix(test.dtm)
test.new.training <-cbind(test.data.aggre[,-5], test.m)
rownames(test.new.training) <-seq(1,nrow(test.new.training),1)
cols <- c(2, 3, 4)
test.new.training[cols] <- lapply(test.new.training[cols], factor)
colnames(test.new.training)[cols] <- c("Usergender","Usertopic","Usersign")
test.dummie <- dummyVars("~.",test.new.training)
test.dummie <- data.frame(predict(test.dummie, newdata = test.new.training))

new.test.data <- test.dummie[colnames(test.dummie) %in% colnames(new.training)]
not.exist <-new.training[!colnames(new.training) %in% colnames(test.dummie)]
etrCol <- data.frame(matrix(0, nrow =nrow(new.test.data), ncol = length(colnames(not.exist))))
colnames(etrCol) <-as.vector(colnames(not.exist))
new.test.data2 <-cbind(new.test.data, etrCol)
new.test.data2 <- new.test.data2[,order(names(new.test.data2))]
prediction <- data.frame(predict(fit, as.matrix(new.test.data2),s = 0.05))
'

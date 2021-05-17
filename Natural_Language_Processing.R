library(data.table)
library(tidyverse)
library(text2vec)
library(caTools)
library(glmnet)
library(dplyr)


data <- read.csv("nlpdata.csv")
data %>% class()
inspectdf::inspect_na(data)


# Split data
set.seed(123)
split <- data$Liked %>% sample.split(SplitRatio = 0.8)
train <- data %>% subset(split == T)
test <- data %>% subset(split == F)


#  Preprocessing

it_train <- train$Review %>% 
  itoken(preprocessor = tolower, 
         tokenizer = word_tokenizer,
         ids = train$X,
         progressbar = F)


vocab <- it_train %>% create_vocabulary()
vocab %>% 
  arrange(desc(term_count)) %>% 
  head(110) %>% 
  tail(10)

vectorizer <- vocab %>% vocab_vectorizer()
dtm_train <- it_train %>% create_dtm(vectorizer)

dtm_train %>% dim()
identical(rownames(dtm_train), train$X)


# Modeling ----
glmnet_classifier <- dtm_train %>%
  cv.glmnet(y = train[['Liked']],
            family = 'binomial', 
            type.measure = "auc",
            nfolds = 10,
            thresh = 0.001,# high value is less accurate, but has faster training
            maxit = 1000)# again lower number of iterations for faster training

glmnet_classifier$cvm %>% max() %>% round(3) %>% paste("-> Max AUC")


it_test <- test$Review %>% tolower() %>% word_tokenizer()

it_test <- it_test %>% 
  itoken(ids = test$X,
         progressbar = F)

dtm_test <- it_test %>% create_dtm(vectorizer)

#create_dtm(it_test, vectorizer,
           #type = c("dgCMatrix", "dgTMatrix", "RsparseMatrix"))
           
preds <- predict(glmnet_classifier, dtm_test, type = 'response')[,1]
glmnet:::auc(test$Liked, preds) %>% round(2)


# Prune some words ----
stop_words <- c("i", "you", "he", "she", "it", "we", "they",
                "me", "him", "her", "them",
                "my", "your", "yours", "his", "our", "ours",
                "myself", "yourself", "himself", "herself", "ourselves",
                "the", "a", "an", "and", "or", "on", "by", "so",
                "from", "about", "to", "for", "of", 
                "that", "this", "is", "are")

vocab <- it_train %>% create_vocabulary(stopwords = stop_words)


pruned_vocab <- vocab %>% 
  prune_vocabulary(term_count_min = 10, 
                   doc_proportion_max = 0.5,
                   doc_proportion_min = 0.001)

pruned_vocab %>% 
  arrange(desc(term_count)) %>% 
  head(10) 

vectorizer <- pruned_vocab %>% vocab_vectorizer()

dtm_train <- it_train %>% create_dtm(vectorizer)
dtm_train %>% dim()

glmnet_classifier <- dtm_train %>% 
  cv.glmnet(y = train[['Liked']], 
            family = 'binomial',
            type.measure = "auc",
            nfolds = 4,
            thresh = 0.001,
            maxit = 1000)

glmnet_classifier$cvm %>% max() %>% round(3) %>% paste("-> Max AUC")

dtm_test <- it_test %>% create_dtm(vectorizer)
preds <- predict(glmnet_classifier, dtm_test, type = 'response')[,1]
glmnet:::auc(test$Liked, preds) %>% round(2)


# N Grams ----
vocab <- it_train %>% create_vocabulary(ngram = c(1L, 2L))

vocab <- vocab %>% 
  prune_vocabulary(term_count_min = 10, 
                   doc_proportion_max = 0.5)

bigram_vectorizer <- vocab %>% vocab_vectorizer()

dtm_train <- it_train %>% create_dtm(bigram_vectorizer)
dtm_train %>% dim()

glmnet_classifier <- dtm_train %>% 
  cv.glmnet(y = train[['Liked']], 
            family = 'binomial',
            type.measure = "auc",
            nfolds = 4,
            thresh = 0.001,
            maxit = 1000)

glmnet_classifier$cvm %>% max() %>% round(3) %>% paste("-> Max AUC")

dtm_test <- it_test %>% create_dtm(bigram_vectorizer)
preds <- predict(glmnet_classifier, dtm_test, type = 'response')[,1]
glmnet:::auc(test$Liked, preds) %>% round(2)

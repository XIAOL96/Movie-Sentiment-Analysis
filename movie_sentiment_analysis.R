rm(list = ls())
#setwd("~/downloads")

all = read.table("data.tsv",stringsAsFactors = F,header = T)
splits = read.table("splits.csv", header = TRUE) #test id
Vocabulary = read.table("myVocab.txt")
s = 3

# Load libraries
mypackages = c("dplyr", "text2vec", "tidytext", "glmnet")
tmp = setdiff(mypackages, rownames(installed.packages()))
if(length(tmp) > 0) install.packages(tmp)
library(dplyr)
library(text2vec)
library(tidytext)
library(glmnet)

# Remove HTML tags
all$review <- gsub('<.*?>', ' ', all$review)
# Remove grammar/punctuation
all$review <- tolower(gsub('[[:punct:]]', '', all$review))
# Remove numbers
all$review <- gsub('[[:digit:]]+', '', all$review)

stop_words = c("i", "me", "my", "myself",
               "we", "our", "ours", "ourselves",
               "you", "your", "yours",
               "their", "they", "his", "her",
               "she", "he", "a", "an", "and",
               "is", "was", "are", "were",
               "him", "himself", "has", "have",
               "it", "its", "of", "one", "for",
               "the", "us", "this")

# Train-test split
test = all[which(all$new_id %in% splits[,s]),]
train = all[-which(all$new_id %in% splits[,s]),]

# Create a vocabulary-based DTM
prep_fun = tolower
tok_fun = word_tokenizer
train_tokens = train$review %>% prep_fun %>% tok_fun
it_train = itoken(train_tokens, ids = train$id, progressbar = FALSE)

vocab = create_vocabulary(it_train, ngram = c(1L, 2L),
                          stopwords = stop_words)

vectorizer = vocab_vectorizer(vocab)
dtm_train = create_dtm(it_train, vectorizer)               

test_tokens = test$review %>% prep_fun %>% tok_fun
it_test = itoken(test_tokens, ids = test$id, progressbar = FALSE)
dtm_test = create_dtm(it_test, vectorizer)

# Fit model
set.seed(500)
NFOLDS = 15
train_x = dtm_train[,which(colnames(dtm_train)%in%Vocabulary[,1])]
test_x = dtm_test[,which(colnames(dtm_test)%in%Vocabulary[,1])]

mycv = cv.glmnet(x=train_x, y=train$sentiment, 
                 family='binomial',type.measure = "auc", 
                 nfolds = NFOLDS, alpha=0)   #ridge
myfit = glmnet(x=train_x, y=train$sentiment, 
               lambda = mycv$lambda.min, family='binomial', alpha=0) 

logit_pred = predict(myfit, test_x, type = "response")

# Evaluation
glmnet:::auc(test$sentiment, logit_pred) 

# Results
results = data.frame(new_id = test$new_id, prob = logit_pred)
colnames(results) = c("new_id", "prob")


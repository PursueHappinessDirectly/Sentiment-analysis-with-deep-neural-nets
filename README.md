# Sentiment-analysis-with-deep-neural-nets
The data folder contains negative and positive reviews from ecommerce platform.     
The process_data folder contains train data set, test set and word dictionary. They are the results from data pretreatment.      
The model folder contains deep neural network, which is consisted of Bi-directional GRU layer and attention mechanism.       
The result folder contains prediction results from trained model.      

The command to run the code is       
python main.py -positive_reviews Data/pos.txt -negative_reviews Data/neg.txt

import argparse
from utils.process import *
from model.model import *
import json


if __name__ == '__main__':
    #set for commander line input
    parser = argparse.ArgumentParser(description='Sentiment Analysis')
    parser.add_argument("-positive_reviews", help="need pos.txt file", type=str, required=True)
    parser.add_argument("-negative_reviews", help="need neg.txt file", type=str, required=True)
    args = parser.parse_args()

    #read file and preprocess the data
    print("Start reading pos.txt file.")
    review_pos = read_file(args.positive_reviews)
    print("Start reading neg.txt file.")
    review_neg = read_file(args.negative_reviews)

    #train/test split
    print("Start train/test data split")
    X_train_set, y_train, X_test_set, y_test = test_train_generator(review_pos, review_neg)
    word_dict = word_to_index(X_train_set, 90)
    X_train = sentences_to_indices(np.array(X_train_set), word_dict, 80) 
    X_test = sentences_to_indices(np.array(X_test_set), word_dict, 80) 

    #saving engineered data
    print("Saving engineered features into files.")
    np.save('process_data/X_train.npy', X_train)
    np.save('process_data/X_test.npy', X_test)
    np.save('process_data/y_train.npy', np.array(y_train))
    np.save('process_data/y_test.npy', np.array(y_test))
    with open('process_data/word_dict.json', 'w') as fp:
        json.dump(word_dict, fp)

    #train model and make predictions
    mod = attention_model(n_words = int(max(X_train.flatten())))
    mod.build_model()
    mod.fit(X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test)
    mod.predict(X_test = X_test, y_test = y_test, word_dict = word_dict)
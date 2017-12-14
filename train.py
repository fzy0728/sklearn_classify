from feature import *
from sklearn import linear_model
from sklearn import svm
from sklearn.model_selection import cross_val_score

from collections import defaultdict
import pandas as pd


def construct_dict(df):
    """
    Construct slot and char dict from df.
    :param df: data df, with columns: [matchedStr, regex, tag]
    :return: dict with slot name, and dict with char
    """
    slot_dict = defaultdict(lambda: 0) # mapping from slot_keys to supports
    char_dict = defaultdict(lambda: 0) # mapping from characters to their supports
    # read each row
    for regex in df.regex:
        # store and count all slot occurrences
        for slot in re.findall(SLOT_PATTERN, regex):
            slot_dict[slot] += 1
        regex = re.sub(SLOT_PATTERN, '', regex) # get rid of slots
        # store and count all char occurrences
        for char in regex:
            if char in slot_regex_expr_list: # ignore regex expression
                continue
            char_dict[char] += 1

    return slot_dict, char_dict


if __name__ == '__main__':

    # setup
    data_path = './pattern_data.csv'
    data_df = pd.read_csv(data_path, error_bad_lines=False)
    n_samples = len(data_df)
    train_df = data_df[: int(n_samples * 0.8)]
    test_df = data_df[int(n_samples * 0.8):]

    # get dict
    train_slot_dict, train_char_dict = construct_dict(data_df)

    # get X and y
    X = None
    for sample_tuple in data_df.iterrows():
        sample = sample_tuple[1]
        feature = extract_feature(sample['matchedStr'], sample['regex'], train_slot_dict, train_char_dict)
        X = feature if X is None else np.vstack((X, feature))

    y = np.array(data_df.tag.values)


    # # Logistic Regression
    # lr = linear_model.LogisticRegression(C=10)
    # # lr.fit(X_train, y_train)
    # # print('accuracy: {}'.format(sum(lr.predict(X_test) == y_test) / len(y_test)))
    # scores = cross_val_score(lr, X, y, cv=5)
    # print(scores)
    # print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    # SVM
    clf = svm.SVC(C=1.0, kernel='linear', decision_function_shape='ovo', probability=True)
    # clf.fit(X_train, y_train)
    # print('accuracy: {}'.format(sum(clf.predict(X_test) == y_test) / len(y_test)))
    # cross validation
    scores = cross_val_score(clf, X, y, cv=5)
    print(scores)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))




    # # get X and y for training
    # X_train = None
    # for sample_tuple in train_df.iterrows():
    #     sample = sample_tuple[1]
    #     feature = extract_feature(sample['matchedStr'], sample['regex'], train_slot_dict, train_char_dict)
    #     X_train = feature if X_train is None else np.vstack((X_train, feature))
    #
    # y_train = np.array(train_df.tag.values)
    #
    # # get X and y for testing
    # X_test = None
    # for sample_tuple in test_df.iterrows():
    #     sample = sample_tuple[1]
    #     feature = extract_feature(sample['matchedStr'], sample['regex'], train_slot_dict, train_char_dict)
    #     X_test = feature if X_test is None else np.vstack((X_test, feature))
    # y_test = np.array(test_df.tag.values)

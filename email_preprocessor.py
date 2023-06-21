"""email_preprocessor.py
Preprocess Enron email dataset into features for use in supervised learning algorithms
"""
import re
import os
import numpy as np


def tokenize_words(text):
    """Transforms an email into a list of words.

    Parameters:
    -----------
    text: str. Sentence of text.

    Returns:
    -----------
    Python list of str. Words in the sentence `text`.

    """
    # Define words as lowercase text with at least one alphabetic letter
    pattern = re.compile(r"[A-Za-z]+[\w^\']*|[\w^\']*[A-Za-z]+[\w^\']*")
    return pattern.findall(text.lower())


def count_words(email_path="data/enron"):
    """Determine the count of each word in the entire dataset (across all emails)

    Parameters:
    -----------
    email_path: str. Relative path to the email dataset base folder.

    Returns:
    -----------
    word_freq: Python dictionary. Maps words (keys) to their counts (values) across the dataset.
    num_emails: int. Total number of emails in the dataset.
    """
    count = 0
    word_freq = {}

    for dirpath, dirnames, filenames in os.walk(email_path):
        for i in filenames:
            if ".txt" in i:
                filepath = os.path.join(dirpath, i)
                file = open(filepath, encoding="latin-1").read()
                words = tokenize_words(file)
                for word in words:
                    if word in word_freq:
                        word_freq[word] += 1
                    else:
                        word_freq[word] = 1
                count += 1

    return word_freq, count


def find_top_words(word_freq, num_features=200):
    """Given the dictionary of the words that appear in the dataset and their respective counts,
    compile a list of the top `num_features` words and their respective counts.

    Parameters:
    -----------
    word_freq: Python dictionary. Maps words (keys) to their counts (values) across the dataset.
    num_features: int. Number of top words to select.

    Returns:
    -----------
    top_words: Python list. Top `num_features` words in high-to-low count order.
    counts: Python list. Counts of the `num_features` words in high-to-low count order.
    """

    words = []
    counts = []

    frequencies = sorted(word_freq, key=word_freq.get, reverse=True)

    for i in range(num_features):
        if i < len(frequencies):
            curr = frequencies[i]
            words.append(curr)
            counts.append(word_freq[curr])

    return words, counts


def make_feature_vectors(top_words, num_emails, email_path="data/enron"):
    """Count the occurance of the top W (`num_features`) words in each individual email, turn into
    a feature vector of counts.

    Parameters:
    -----------
    top_words: Python list. Top `num_features` words in high-to-low count order.
    num_emails: int. Total number of emails in the dataset.
    email_path: str. Relative path to the email dataset base folder.

    Returns:
    -----------
    feats. ndarray. shape=(num_emails, num_features).
        Vector of word counts from the `top_words` list for each email.
    y. ndarray of nonnegative ints. shape=(num_emails,).
        Class index for each email (spam/ham)
    """

    labels = np.zeros(num_emails)
    word_freq = np.zeros((num_emails, len(top_words)))

    index = 0

    for dirpath, dirnames, filenames in os.walk(email_path):
        for i in filenames:
            if ".txt" in i:
                filepath = os.path.join(dirpath, i)
                file = open(filepath, encoding="latin-1").read()
                words = tokenize_words(file)

                if "spam" in filepath:
                    labels[index] = 0
                else:
                    labels[index] = 1

                for j in range(len(top_words)):
                    word_freq[index, j] = words.count(top_words[j])

                index += 1

    return word_freq, labels


def make_train_test_sets(features, y, test_prop=0.2, shuffle=True):
    """Divide up the dataset `features` into subsets ("splits") for training and testing. The size
    of each split is determined by `test_prop`.

    Parameters:
    -----------
    features. ndarray. shape=(num_emails, num_features).
        Vector of word counts from the `top_words` list for each email.
    y. ndarray of nonnegative ints. shape=(num_emails,).
        Class index for each email (spam/ham)
    test_prop: float. Value between 0 and 1.  The proportion of the dataset to use for testing.
    shuffle: boolean. Whether or not to shuffle the dataset before splitting.

    Returns:
    -----------
    x_train: ndarray. shape=(num_train_samps, num_features).
        Training dataset
    y_train: ndarray. shape=(num_train_samps,).
        Class values for the training set
    inds_train: ndarray. shape=(num_train_samps,).
        The index of each training set email in the original unshuffled dataset.
    x_test: ndarray. shape=(num_test_samps, num_features).
        Test dataset
    y_test:ndarray. shape=(num_test_samps,).
        Class values for the test set
    inds_test: ndarray. shape=(num_test_samps,).
        The index of each test set email in the original unshuffled dataset.
    """
    inds = np.arange(y.size)
    if shuffle:
        features = features.copy()
        y = y.copy()

        inds = np.arange(y.size)
        np.random.shuffle(inds)
        features = features[inds]
        y = y[inds]

    train = int(features.shape[0] * (1 - test_prop))

    x_train = features[:train, :]
    y_train = y[:train]
    inds_train = inds[:train]

    x_test = features[train:, :]
    y_test = y[train:]
    inds_test = inds[train:]

    return x_train, y_train, inds_train, x_test, y_test, inds_test


def retrieve_emails(inds, email_path="data/enron"):
    """Obtain the text of emails at the indices `inds` in the dataset.

    Parameters:
    -----------
    inds: ndarray of nonnegative ints. shape=(num_inds,).
        The number of ints is user-selected and indices are counted from 0 to num_emails-1
        (counting does NOT reset when switching to emails of another class).
    email_path: str. Relative path to the email dataset base folder.

    Returns:
    -----------
    Python list of str. len = num_inds = len(inds).
        Strings of entire raw emails at the indices in `inds`
    """
    pass

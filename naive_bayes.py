'''naive_bayes_multinomial.py
Naive Bayes classifier with Multinomial likelihood for discrete features
'''
import numpy as np


class NaiveBayes:
    '''Naive Bayes classifier using Multinomial likeilihoods'''
    def __init__(self, num_classes):
        '''Naive Bayes constructor
        '''
        self.num_classes = num_classes
        self.class_priors = None
        self.class_likelihoods = None

    def get_priors(self):
        '''Returns the class priors'''
        return self.class_priors

    def get_likelihoods(self):
        '''Returns the class likelihoods'''
        return self.class_likelihoods

    def train(self, data, y):
        '''Train the Naive Bayes classifier so that it records the "statistics" of the training set

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features).
        y: ndarray. shape=(num_samps,). 
        '''
        num_features = data.shape[1]
        self.class_likelihoods = np.zeros((self.num_classes, num_features))
        self.class_priors = np.zeros(self.num_classes)

        for i in range(self.num_classes):
            matches = np.where(y == i, 1, 0)
            self.class_priors[i] = np.sum(matches)/y.shape[0]
            for j in range(data.shape[1]):
                idx = np.where(y == i)
                class_words = np.sum(data[:, j][idx])
                total_words = np.sum(data[idx, :])
                likelihoods = (class_words + 1)/(total_words + data.shape[1])
                self.class_likelihoods[i, j] = likelihoods


    def predict(self, data):
        '''Combine the class likelihoods and priors to compute the posterior distribution.

        Parameters:
        -----------
        data: ndarray. shape=(num_test_samps, num_features).

        Returns:
        -----------
        ndarray of nonnegative ints. shape=(num_samps,). Predicted class of each test data sample.
        '''
        log_prior = np.log(self.class_priors)
        log_posterior = log_prior + data @ np.log(self.class_likelihoods).T

        classes = np.argmax(log_posterior, axis = 1)

        return classes



    def accuracy(self, y, y_pred):
        '''Computes accuracy based on percent correct: Proportion of predicted class labels `y_pred`
        that match the true values `y`.

        Parameters:
        -----------
        y: ndarray. shape=(num_data_sams,)
            Ground-truth, known class labels for each data sample
        y_pred: ndarray. shape=(num_data_sams,)
            Predicted class labels by the model for each data sample

        Returns:
        -----------
        float. Between 0 and 1. Proportion correct classification.

        '''
        
        pc = np.sum(np.where(y == y_pred, 1, 0))/y.shape[0] * 100

        return pc

    def confusion_matrix(self, y, y_pred):
        '''Create a confusion matrix based on the ground truth class labels (`y`) and those predicted
        by the classifier (`y_pred`).

        Parameters:
        -----------
        y: ndarray. shape=(num_data_samps,)
            Ground-truth, known class labels for each data sample
        y_pred: ndarray. shape=(num_data_samps,)
            Predicted class labels by the model for each data sample

        Returns:
        -----------
        ndarray. shape=(num_classes, num_classes).
            Confusion matrix
        '''
        
        matrix = np.zeros((self.num_classes, self.num_classes))

        for i in range(self.num_classes):
            for j in range(self.num_classes):
                act_match = np.where(y == i, 1, 0)
                pred_match = np.where(y_pred == j, 1, 0)
                matches = np.logical_and(act_match, pred_match)
                matrix[i, j] = np.sum(np.where(matches == True, 1, 0))

        return matrix
    
    def kfold(self, data, labels, k):
        '''Perform k-fold cross validation on the data and labels. Returns an array of accuracies
        
        Parameters:
        -----------
        data: ndarray. shape=(num_data_samps, num_features).
        labels: ndarray. shape=(num_data_samps,).
        k: int. Number of folds to use in cross validation
        
        Returns:
        -----------
        accuracies: ndarray. shape=(k,). Array of accuracies for each fold
        '''
        inds = np.arange(labels.size)

        # shuffle data
        features = data.copy()
        y = labels.copy()
        inds = np.arange(y.size)
        np.random.shuffle(inds)
        features = features[inds]
        y = y[inds]

        accuracies = np.zeros(k)

        # start folds
        start = 0
        fold = y.size//k

        for i in range(k):
            end = start + fold

            # test fold
            x_test = features[start:end, :]
            y_test = y[start:end]

    
            # before kth fold
            x_before = features[0:start, :]
            y_before = y[0:start]

            # after kth fold
            x_after = features[end:, :]
            y_after = y[end:]

            # combine into training
            x_train = np.vstack((x_before, x_after))
            y_train = np.hstack((y_before, y_after))

            # print("train", x_train.shape)

            # train and eval
            self.train(x_train, y_train)
            y_pred = self.predict(x_test)
            acc = self.accuracy(y_test, y_pred)
            accuracies[i] = acc

            start += fold
        
        return accuracies

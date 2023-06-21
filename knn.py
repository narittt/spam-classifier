'''knn.py
K-Nearest Neighbors algorithm for classification
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from palettable import cartocolors


class KNN:
    '''K-Nearest Neighbors supervised learning algorithm'''
    def __init__(self, num_classes):
        '''KNN constructor
        '''
        self.exemplars = None
        self.classes = None

        self.num_classes = num_classes

    def train(self, data, y):
        '''Train the KNN classifier on the data `data`, where training samples have corresponding
        class labels in `y`.

        Parameters:
        -----------
        data: ndarray. shape=(num_train_samps, num_features). Data to learn / train on.
        y: ndarray. shape=(num_train_samps,). Corresponding class of each data sample.
        '''
        self.exemplars = data
        self.classes = y

    def predict(self, data, k):
        '''Use the trained KNN classifier to predict the class label of each test sample in `data`.

        Parameters:
        -----------
        data: ndarray. shape=(num_test_samps, num_features). Data to predict the class of
        k: int. Determines the neighborhood size of training points around each test sample used to
            make class predictions.

        Returns:
        -----------
        ndarray of nonnegative ints. shape=(num_test_samps,). Predicted class of each test data
        sample.

        '''

        predicted_classes = []
        
        for sample in range(data.shape[0]):
            dist = np.sqrt(np.sum(np.square(self.exemplars - data[sample]), axis = 1))
            # print(dist.shape)
            closest = np.argpartition(dist, k)[:k]
            classes, counts = np.unique(self.classes[closest], return_counts = True)
            predicted_classes.append(classes[np.argmax(counts)])
        
        return np.array(predicted_classes)

    def accuracy(self, y, y_pred):
        '''Computes accuracy based on percent correct: Proportion of predicted class labels `y_pred`
        that match the true values `y`.

        Parameters:
        -----------
        y: ndarray. shape=(num_data_sams. Ground-truth
        y_pred: ndarray. shape=(num_data_sams,)
            Predicted class labels by the model for each data sample

        Returns:
        -----------
        float. Proportion correct classification.

        '''
        N = y.shape[0]

        correct = np.sum(np.where(y == y_pred, 1, 0))

        return correct/N 

    def plot_predictions(self, k, n_sample_pts):
        '''Paints the data space in colors corresponding to which class the classifier would
         hypothetically assign to data samples appearing in each region.

        Parameters:
        -----------
        k: int. Determines the neighborhood size of training points around each test sample used to
            make class predictions.
        n_sample_pts: int.
        '''
        
        color = ListedColormap(cartocolors.qualitative.Safe_4.mpl_colors)

        vector = np.linspace(-40, 40, n_sample_pts)
        x, y = np.meshgrid(vector, vector)
        
        data = np.column_stack((x.flatten(), y.flatten()))
        data = np.reshape(data, (n_sample_pts * n_sample_pts, self.exemplars.shape[1]))

        y_pred = self.predict(data, k)
        
        y_pred = np.reshape(y_pred, (n_sample_pts, n_sample_pts))

        colors = plt.pcolormesh(x, y, y_pred, cmap=color)

        plt.colorbar(colors)
        plt.title("Plot of Predictions")
        plt.xlabel("X")
        plt.ylabel("Y")


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

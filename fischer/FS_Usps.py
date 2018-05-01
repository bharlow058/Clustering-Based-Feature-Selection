import scipy.io
from sklearn import cross_validation
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
# from FS_package.function.similarity_based import fisher_score
# from FS_package.utility import construct_W
import fisher_score as FS
import construct_W as CW


def main():
    # load matlab data
    print '-----------------------------------------'
    print 'Loading \'pixraw10P\' Data !'
    mat = scipy.io.loadmat('../data/USPS.mat')
    print 'Data Loaded !'
    print '-----------------------------------------'
    X = mat['X']    # data
    y = mat['Y']    # label
    y = y[:, 0]
    X = X.astype(float)
    n_samples, n_features = X.shape

    # split data
    print 'Splitting data into 10 folds !'
    ss = cross_validation.KFold(n_samples, n_folds=10, shuffle=True)
    print 'Data Splitted !'
    print '-----------------------------------------'

    # evaluation
    num_fea = 100
    print 'Initializing KNN !'
    neigh = KNeighborsClassifier(n_neighbors=10)
    print 'KNN Initialized !'
    print '-----------------------------------------'
    correct = 0

    fold_no = 0
    for train, test in ss:
        print '\tFold No.',fold_no
        kwargs = {"neighbor_mode": "supervised", "fisher_score": True, 'y': y[train]}
        print 'Constructing Affinity Matrix !'
        # W = construct_W.construct_W(X[train], **kwargs)
        W = CW.construct_W(X[train], **kwargs)
        print 'Affinity Matrix Constructed !'

        print 'Calculating Fischer Score and ranking...'
        # score = fisher_score.fisher_score(X[train], y[train])
        score = FS.fisher_score(X[train], y[train])
        # idx = fisher_score.feature_ranking(score)
        idx = FS.feature_ranking(score)
        print 'Fischer Score and ranking calculated !'

        selected_features = X[:, idx[0:num_fea]]
        neigh.fit(selected_features[train], y[train])
        y_predict = neigh.predict(selected_features[test])
        acc = accuracy_score(y[test], y_predict)
        print acc
        correct = correct + acc
        fold_no += 1
        print '-----------------------------------------'

    print '10 fold Cross - Validation Accuracy:', round((float(correct)/10)*100.0,2),'%'


if __name__ == '__main__':
    main()


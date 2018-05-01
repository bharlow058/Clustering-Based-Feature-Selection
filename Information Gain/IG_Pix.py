import numpy as np
import scipy.io
from sklearn.metrics import accuracy_score
from sklearn import cross_validation
from sklearn.svm import SVC
import MIFS


def main():
    print 'Information Gain !'
    filename = '../data/pixraw10P.mat'#['../data/arcene.mat', '../data/gisette.mat', '../data/madelon.mat']
    # for f_num in range(len(filename)):
    print filename
    mat = scipy.io.loadmat(filename)
    X = mat['X']    # data
    y = mat['Y']    # label
    y = y[:, 0]
    X = X.astype(float)
    n_sample, n_features = X.shape
    # split data
    ss = cross_validation.KFold(n_sample, n_folds=10, shuffle=True)
    # choose SVM as the classifier
    clf = SVC()
    num_fea = np.linspace(5, 300, 60)
    correct = np.zeros(len(num_fea))
    f=0
    for train, test in ss:
        # select features
        print 'Fold No.',f+1
        f+=1
        print 'Training Model !'
        F = MIFS.mifs(X[train], y[train], n_selected_features=300)
        print 'Model Trained !'
        for n in range(len(num_fea)):
            fea_idx = F[0:num_fea[n]]
            features = X[:, fea_idx]
            clf.fit(features[train], y[train])
            y_predict = clf.predict(features[test])
            acc = accuracy_score(y[test], y_predict)
            correct[n] += acc
    correct.astype(float)
    correct /= 10
    for i in range(len(num_fea)):
        print num_fea[i], correct[i]
    print 'Accuracy:',round(correct*100.0,2),'%'


if __name__ == '__main__':
    main()
import scipy.io
import MCFS
from construct_W import construct_W
import unsupervised_evaluation


def main():
    # load matlab data
    print 'Loading Data !'
    mat = scipy.io.loadmat('../data/pixraw10P.mat')
    print 'Data Loaded !'
    X = mat['X']
    X = X.astype(float)
    y = mat['Y']
    y = y[:, 0]

    # construct W
    kwargs = {"metric": "euclidean", "neighborMode": "knn", "weightMode": "heatKernel", "k": 5, 't': 0.1}
    W = construct_W(X, **kwargs)

    # mcfs feature selection
    n_selected_features = 100
    print 'Training Model !'
    S = MCFS.mcfs(X, n_selected_features, W=W, n_clusters=20)
    print 'Model Trained !'
    idx = MCFS.feature_ranking(S)

    # evaluation
    X_selected = X[:, idx[0:n_selected_features]]
    ari, nmi, acc = unsupervised_evaluation.evaluation(X_selected=X_selected, n_clusters=20, y=y)
    # print 'ARI:', ari
    # print 'NMI:', nmi
    print 'Accuracy:',round(acc*100.0,2),'%'

if __name__ == '__main__':
    main()
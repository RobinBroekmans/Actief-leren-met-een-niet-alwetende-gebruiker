import numpy as np
import pandas as pd
from anomatools.models import SSkNNO
from matplotlib import pyplot as plt
from scipy.io import arff
from sklearn import datasets, mixture
from sklearn import svm
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle

import Strategies
from Parameters import *
from data import Data
import math
from sklearn import datasets
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics.pairwise import check_pairwise_arrays
from scipy.linalg import cholesky
from sklearn.linear_model import LogisticRegression
import json
from sklearn.linear_model import SGDClassifier
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def set_up_uncertainty(X, y):
    w = []
    for i in range(len(X)):
        curr_w = 1
        if y[i] == 1:
            curr_w = 1 * 20
            if d_name[-1] == 'Ionosphere':
                curr_w = 20 
        w.append(curr_w) 
    if m_gmm:
        clf = mixture.GaussianMixture(n_components=1, covariance_type='full')
        lenght = np.count_nonzero(y == 1)
        normals = np.zeros(np.shape(X))
        normals = normals[lenght:]
        ctr = 0
        for w in range(len(X)):
            if y[w] == -1:
                normals[ctr] = X[w]
                ctr = ctr + 1
        clf.fit(normals)
    else:
        clf = svm.SVC(random_state=11, probability=True)  
        clf.fit(X, y) 

    calibrated = CalibratedClassifierCV(base_estimator=clf, method='isotonic') 
    calibrated.fit(X, y, sample_weight=w)

    predict_proba = calibrated.predict_proba(X)

    certainties = np.zeros(len(X))

    if m_gmm:
        for i in range(len(predict_proba)):
            curr = predict_proba[i][0]
            certainties[i] = curr
    else:
        for i in range(len(predict_proba)):
            if y[i] == -1:
                ano = 0
            else:
                ano = 1
            curr = predict_proba[i][ano]
            db = predict_proba[i][abs(1 - ano)]
            if db > curr:  
                pass 
            certainties[i] = pow(curr, 2)       


    if user_unc_plot:
        a = np.sort(certainties)
        plt.plot(a)
        plt.title("User certainties of the points")
        plt.show()

    return certainties 
def set_up_data_dami(name):
    loc = ''
    if name == "Annthyroid_withoutdupl_02_v09":
        loc = loc + name
    elif name == 'Annthyroid_withoutdupl_05_v09':
        loc = loc + name
    elif name == 'KDDCup99_withoutdupl_1ofn':
        loc = loc + name
    elif name == 'KDDCup99_withoutdupl_catremoved':
        loc = loc + name
    elif name == 'KDDCup99_withoutdupl_idf':
        loc = loc + name
    elif name == 'PageBlocks_withoutdupl_02_v09':
        loc = loc + name
    elif name == 'PageBlocks_withoutdupl_05_v09':
        loc = loc + name
    elif name == 'Spambase (2%)':
        loc = loc + 'SpamBase_withoutdupl_02_v10'
    elif name == 'SpamBase':
        loc = loc + 'SpamBase_withoutdupl_05_v10'
    elif name == 'SpamBase_withoutdupl_10_v10':
        loc = loc + name
    elif name == 'SpamBase_withoutdupl_20_v10':
        loc = loc + name
    elif name == 'SpamBase_withoutdupl_40':
        loc = loc + name
    elif name == 'Waveform':
        loc = loc + 'Waveform_withoutdupl_v10'
    elif name == 'Wilt_withoutdupl_02_v10':
        loc = loc + name
    elif name == 'Wilt_withoutdupl_05':
        loc = loc + name
    elif name == 'Stamps_withoutdupl_norm_02_v04':
        loc = loc + name
    elif name == "Pima_withoutdupl_02_v10":
        loc = loc + name
    elif name == "Waveform_withoutdupl_norm_v10":
        loc = loc + name
    elif name == 'ALOI_withoutdupl':
        loc = loc + name
    elif name == 'Cardiotocography':
        loc = loc + 'Cardiotocography_withoutdupl_05_v01'
    elif name == 'Ionosphere':
        loc = loc + 'ionosphere_arff'

    else:
        print("Not sure if this name exists...")
        loc = loc + name
    loc = loc + '.arff'

    if normelize:
        data = arff.loadarff(loc)
    else:
        print("ONLY NORM DATASETS CAN BE USED")
    df = pd.DataFrame(data[0])

    if not name == 'Ionosphere':
        X = df.drop(['outlier', 'id'], axis=1).values
        tmp = df['outlier'].values
    else:
        X = df.drop(['class'], axis=1).values
        tmp = df['class'].values

    y = []
    if not name == 'Ionosphere':
        for i in tmp:
            if i == b'yes':
                y.append(1)
            elif i == b'no':
                y.append(-1)
            else:
                raise RuntimeError("Problem with data set-up")
    else:
        for i in tmp:
            if i == b'b':
                y.append(1)
            elif i == b'g':
                y.append(-1)
            else:
                raise RuntimeError("Problem with data set-up")


    uncertainty = set_up_uncertainty(X, y) 
    contamination = y.count(1) / len(y)
    print(name + " dataset")
    print("Contamination: " + str(contamination))
    X_train, X_test, y_train, y_test, train_uncertainty, test_uncertainty = train_test_split(X, y, uncertainty, test_size=0.3,random_state=r_rs[-1],stratify=y)

    if False:
        pca = PCA().fit(X_train) 
        plt.xlabel('number of components')
        plt.ylabel('cumulative explained variance')
        plt.show()  

        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)   

    skf = StratifiedKFold(n_splits=n_splits)
    fold_train = []
    fold_test = []
    for trains, tests in skf.split(X, y):
        fold_train.append(trains)
        fold_test.append(tests)

    return Data(X_train, y_train, train_uncertainty, contamination, X_test, y_test, test_uncertainty, name=name, xx=X, yy=np.array(y), uncunc=uncertainty, foldtrain=fold_train, foldtest=fold_test, it=0, k_fold=True)


def set_up_data_cardio():
    if normelize:
        data = arff.loadarff('
        ')
    else:
        data = arff.loadarff('
        ')
    df = pd.DataFrame(data[0])

    X = df.drop('outlier', axis=1).values
    tmp = df['outlier'].values
    y = []
    for i in tmp:
        if i == b'yes':
            y.append(1)
        elif i == b'no':
            y.append(-1)
        else:
            raise RuntimeError("Problem with data set-up")

    uncertainty = set_up_uncertainty(X, y) 
    contamination = y.count(1) / len(y)
    print("Cardio dataset")
    print("Contamination: " + str(contamination))
    X_train, X_test, y_train, y_test, train_uncertainty, test_uncertainty = train_test_split(X, y, uncertainty, test_size=0.3,random_state=r_rs[-1],stratify=y)
    if normelize: 
        pipe = make_pipeline(StandardScaler(), LogisticRegression())
        pipe.fit(X_train, y_train)
        pipe.score(X_test, y_test)
    return Data(X_train, y_train, train_uncertainty, contamination, X_test, y_test, test_uncertainty, name="Cardio_05_v1")


def set_up_data_shuttle():
    data = arff.loadarff('')
    df = pd.DataFrame(data[0])

    X = df.drop('outlier', axis=1).values
    tmp = df['outlier'].values
    y = []
    for i in tmp:
        if i == b'yes':
            y.append(1)
        elif i == b'no':
            y.append(-1)
        else:
            raise RuntimeError("Problem with data set-up")

    uncertainty = set_up_uncertainty(X, y) 
    contamination = y.count(1) / len(y)
    print("Contamination: " + str(contamination))
    X_train, X_test, y_train, y_test, train_uncertainty, test_uncertainty = train_test_split(X, y, uncertainty, test_size=0.2,random_state=r_rs[-1],stratify=y)
    return Data(X_train, y_train, train_uncertainty, contamination, X_test, y_test, test_uncertainty, name="Shuttle")


def set_up_data_ion():
    data = arff.loadarff('')
    df = pd.DataFrame(data[0])

    X = df.drop('class', axis=1).values
    tmp = df['class'].values
    y = []
    for i in tmp:
        if i == b'b':
            y.append(1)
        elif i == b'g':
            y.append(-1)
        else:
            raise RuntimeError("Problem with data set-up")

    uncertainty = set_up_uncertainty(X, y) 
    contamination = y.count(1) / len(y)
    print("Contamination: " + str(contamination))
    X_train, X_test, y_train, y_test, train_uncertainty, test_uncertainty = train_test_split(X, y, uncertainty,
                                                                                             test_size=0.2,
                                                                                             random_state=r_rs[-1],
                                                                                             stratify=y)
    if normelize:
        pca = PCA().fit(X_train)
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.xlabel('number of components')
        plt.ylabel('cumulative explained variance')
        plt.show()  
        pipe = make_pipeline(StandardScaler(), PCA(n_components=0.95, svd_solver='full', random_state=42))
        pipe.fit(X_train, y_train)
        pipe.score(X_test, y_test)
    return Data(X_train, y_train, train_uncertainty, contamination, X_test, y_test, test_uncertainty, name="Ionosphere")


def set_up_data_ALOI():
    if normelize:
        data = arff.loadarff('')
    else:
        data = arff.loadarff('')
    df = pd.DataFrame(data[0])
    X = df.drop('outlier', axis=1).values
    tmp = df['outlier'].values
    y = []
    for i in tmp:
        if i == b'yes':
            y.append(1)
        elif i == b'no':
            y.append(-1)
        else:
            raise RuntimeError("Problem with data set-up")

    uncertainty = set_up_uncertainty(X, y)
    contamination = y.count(1) / len(y)
    print("Contamination: " + str(contamination))

    X_train, X_test, y_train, y_test, train_uncertainty, test_uncertainty = train_test_split(X, y, uncertainty,test_size=0.33, random_state=r_rs[-1], stratify=y)
    if normelize: 
        pipe = make_pipeline(StandardScaler(), LogisticRegression())
        pipe.fit(X_train, y_train)
        pipe.score(X_test, y_test)

    return Data(X_train, y_train, train_uncertainty, contamination, X_test, y_test, name="ALOI")


def set_up_data_waveform():
    data = arff.loadarff('')
    df = pd.DataFrame(data[0])
    X = df.drop('outlier', axis=1).values
    tmp = df['outlier'].values
    y = []
    for i in tmp:
        if i == b'yes':
            y.append(1)
        elif i == b'no':
            y.append(-1)
        else:
            raise RuntimeError("Problem with data set-up")

    uncertainty = set_up_uncertainty(X, y)
    contamination = y.count(1) / len(y)
    print("Contamination: " + str(contamination))
    X_train, X_test, y_train, y_test, train_uncertainty, test_uncertainty = train_test_split(X, y, uncertainty, train_size=0.8,test_size=0.2, random_state=r_rs[-1], stratify=y)
    return Data(X_train, y_train, train_uncertainty, contamination, X_test, y_test, test_uncertainty, name="Waveform")


def set_up_data_wilt():
    data = arff.loadarff('')
    df = pd.DataFrame(data[0])

    X = df.drop('outlier', axis=1).values
    tmp = df['outlier'].values
    y = []
    for i in tmp:
        if i == b'yes':
            y.append(1)
        elif i == b'no':
            y.append(-1)
        else:
            raise RuntimeError("Problem with data set-up")

    uncertainty = set_up_uncertainty(X, y) 
    contamination = y.count(1) / len(y)
    print("Wilt dataset")
    print("Contamination: " + str(contamination))
    X_train, X_test, y_train, y_test, train_uncertainty, test_uncertainty = train_test_split(X, y, uncertainty, test_size=0.3,random_state=r_rs[-1],stratify=y)
    return Data(X_train, y_train, train_uncertainty, contamination, X_test, y_test, test_uncertainty, name="Wilt")


def set_up_data_pageblock():
    if normelize:
        data = arff.loadarff('')
    else:
        data = arff.loadarff('')
    df = pd.DataFrame(data[0])

    X = df.drop('outlier', axis=1).values
    tmp = df['outlier'].values
    y = []
    for i in tmp:
        if i == b'yes':
            y.append(1)
        elif i == b'no':
            y.append(-1)
        else:
            raise RuntimeError("Problem with data set-up")

    uncertainty = set_up_uncertainty(X, y)
    contamination = y.count(1) / len(y)
    print("Pageblock dataset")
    print("Contamination: " + str(contamination))

    X_train, X_test, y_train, y_test, train_uncertainty, test_uncertainty = train_test_split(X, y, uncertainty, test_size=0.3,random_state=r_rs[-1],stratify=y)
    if normelize: 
        pipe = make_pipeline(StandardScaler(), LogisticRegression())
        pipe.fit(X_train, y_train)
        pipe.score(X_test, y_test)
    return Data(X_train, y_train, train_uncertainty, contamination, X_test, y_test, test_uncertainty, name="Pageblock")


def set_up_data_iris():
    iris = datasets.fetch_olivetti_faces()
    X = np.array(iris.data)
    y = np.array(iris.target)
    contamination = 0.05
    d1 = subsample(X,y, contamination)

    X_train, X_test, y_train, y_test, train_uncertainty, test_uncertainty = train_test_split(d1.X, d1.true_labels, d1.true_user_certainty,
                                                                                             test_size=0.3,
                                                                                             random_state=r_rs[-1],
                                                                                             stratify=d1.true_labels)


    return Data(X_train, y_train, train_uncertainty, contamination, X_test, y_test, test_uncertainty, name="Iris")


def set_up_data_digits():
    cap = 1797

    digits = datasets.load_digits()
    n_sample = len(digits.images)
    data = digits.images.reshape((n_sample, -1))
    X = data
    y = digits.target
    X_shuffled, y_shuffled = shuffle(X, y)
    D = X_shuffled[:cap]
    y = y_shuffled[:cap]
    for i in range(len(y)):
        if y[i] == 8 or y[i] == 7 or y[i] == 5:
            y[i] = 1
        else:
            y[i] = -1

    model = svm.SVC(probability=True)
    model.fit(D, y)
    probs = model.predict_proba(D)
    proba = np.zeros(len(probs))
    for i in range(len(probs)):
        proba[i] = (max(probs[i][0], probs[i][1]) *0.98) **7

    order = np.argsort(proba) 
    percentage_anomaly = 0.05
    sampl = n_sample - 500
    data_X = []
    data_y = []
    data_probs = []
    found = False
    k = 0
    nbr_anomaly_points = int(sampl * percentage_anomaly)
    nbr_normal_points = int(sampl * (1-percentage_anomaly))
    print("Digits dataset")
    print("Anomalies: " + str(nbr_anomaly_points))
    print("Normal: " + str(nbr_normal_points))
    for i in range(nbr_anomaly_points):
        while not found:
            r = order[k] 
            k = k + 1
            if y[r] == 1:
                found = True
        rnd_X = D[r]
        rnd_y = y[r]
        rnd_prob = proba[r]
        data_X.append(rnd_X)
        data_y.append(rnd_y)
        data_probs.append(rnd_prob)
        found = False

    found = False
    k = 0
    for i in range(int(sampl * (1 - percentage_anomaly))):
        while not found:
            r = order[k] 
            k = k + 1
            if y[r] == -1:
                found = True
        rnd_X = D[r]
        rnd_y = y[r]
        rnd_prob = proba[r]
        data_X.append(rnd_X)
        data_y.append(rnd_y)
        data_probs.append(rnd_prob)
        found = False

    p = []
    for i in range(len(data_probs)):
        p.append(i)

    if user_unc_plot:
        a = np.sort(data_probs)
        plt.plot(p, a)
        plt.title("User uncertainties of the points")
        plt.show()

    X_train, X_test, y_train, y_test, train_uncertainty, test_uncertainty = train_test_split(data_X, data_y, data_probs,
                                                                                             train_size=0.7,
                                                                                             test_size=0.3,
                                                                                             random_state=r_rs[-1],
                                                                                             stratify=data_y)

    return Data(np.array(X_train), np.array(y_train), train_uncertainty, percentage_anomaly, np.array(X_test),
                np.array(y_test), test_uncertainty, name="Digits")


def set_up_2D_dataset():
    contamination = 0.05
    n = math.ceil(samples)*1.5
    n = int(n)
    X, y = make_blobs(n_samples=n, cluster_std=4, centers=3, n_features=2, random_state=7) 

    if q_data_set_up:
        plt.scatter(X[:, 0], X[:, 1], c=y)
        plt.title("Genereerde clusters")
        plt.savefig('clusters')
        plt.show()
    d1 = subsample(X, y, contamination)

    X_train, X_test, y_train, y_test, train_uncertainty, test_uncertainty = train_test_split(d1.X, d1.true_labels,
                                                                                             d1.true_user_certainty,
                                                                                             test_size=0.3,
                                                                                             random_state=r_rs[-1],
                                                                                             stratify=d1.true_labels)

    skf = StratifiedKFold(n_splits=n_splits)
    fold_train = []
    fold_test = []
    for trains, tests in skf.split(d1.X, d1.true_labels):
        fold_train.append(trains)
        fold_test.append(tests)

    c = d1.contamination
    detector = SSkNNO(contamination=c, weighted=True)
    detector.fit(X)
    predict_proba = detector.predict_proba(X, method='squash')
    predict_proba_m = np.array(Strategies.get_model_uncertainty(7, predict_proba))
    print(predict_proba_m)
    a = np.full_like(predict_proba_m, 0.75)    
    return Data(X_train, y_train, train_uncertainty, contamination, X_test, y_test, test_uncertainty, name="2D", xx=d1.X,
                yy = d1.true_labels, uncunc=d1.true_user_certainty, foldtrain=fold_train, foldtest=fold_test, k_fold=True, it=0)


def subsample(X, y, cont):  
    clf = svm.SVC(random_state=11) 
    clf.fit(X, y)
    calibrated = CalibratedClassifierCV(base_estimator=clf, method='isotonic') 
    calibrated.fit(X, y)
    probs = calibrated.predict_proba(X)
    proba = np.zeros(len(probs))
    for i in range(len(probs)):
        proba[i] = probs[i][y[i]] 
    contamination = cont
    total_points = samples
    anom_max_count = math.ceil(total_points * contamination)
    anom_count = 0
    normal_max_count = total_points - anom_max_count
    normal_count = 0
    d_X = np.zeros((anom_max_count + normal_max_count, len(X[1])))
    d_y = np.zeros(anom_max_count + normal_max_count)
    d_unc = np.zeros(anom_max_count + normal_max_count)
    ctr = 0
    for i in range(len(y)):
        if y[i] == 0:
            if anom_count < anom_max_count:
                anom_count = anom_count + 1
                d_X[ctr] = X[i]
                d_y[ctr] = 1 
                d_unc[ctr] = proba[i]
                d_unc[ctr] = d_unc[ctr] ** 2
                ctr = ctr + 1
        else:
            if normal_count < normal_max_count:
                normal_count = normal_count + 1
                d_X[ctr] = X[i]
                d_y[ctr] = -1 
                d_unc[ctr] = proba[i]
                d_unc[ctr] = d_unc[ctr] ** 2
                ctr = ctr + 1
        if ctr > total_points:
            break

    if True:
        tmp_distances = np.zeros(anom_max_count + normal_max_count)
        for qq in range(len(d_unc)):
            coor = [0,0] 
            dis = (d_X[qq][0] - coor[0])**2 + (d_X[qq][1] - coor[1])**2
            tmp_distances[qq] = dis
        max_dis = max(tmp_distances)
        for qqq in range(len(d_unc)):
            d_unc[qqq] = ((1 - (tmp_distances[qqq] / max_dis))**20) * 5/10 + (1-d_unc[qqq]) * (5/10)
            d_unc[qqq] = 1 - d_unc[qqq]

    if q_data_set_up:
        a = np.sort(proba)
        plt.plot(a)
        plt.title("Totaal Expert zekerheden") 
        plt.show()
    if q_data_set_up:
        a = np.sort(d_unc)
        plt.plot(a)
        plt.title("Expert zekerheden") 
        plt.show()

    if q_data_set_up:
        plt.scatter(d_X[:, 0], d_X[:, 1], c=d_y)
        plt.title("Bemonsterde dataset") 
        plt.show()

    if q_data_set_up:
        plt.scatter(d_X[:, 0], d_X[:, 1], c=d_unc)
        plt.title("Expert zekerheden")
        plt.colorbar()
        plt.savefig('expertzekerheden')
        plt.show()
    d1 = Data(d_X, d_y, d_unc, contamination, [], [], [], xx=[], yy=[], uncunc=[], foldtrain = [], foldtest=[], it=0, two_d_flag=1, name="Blobs2D", k_fold=False)
    return d1


def set_up_sketch_data(nbr):
    if nbr == 1:
        f = open('')
    elif nbr == 2:
        f = open('')
    elif nbr == 3:
        f = open('')
    elif nbr == 4:
        f = open('')
    elif nbr == 5:
        f = open('')
    elif nbr == 6:
        f = open('')
    elif nbr == 7:
        f = open('')
    elif nbr == 8:
        f = open('')
    elif nbr == 9:
        f = open('')
    data = json.loads(f.read())
    samp = len(data['samples']['no-sketch'][:])

    X = np.zeros((samp,2))
    y = np.zeros(samp)
    aa = data['samples']['no-sketch'][:]
    anom = 0
    normal = 0
    for i in range(samp):
        a = aa[i]
        X[i][0] = a['x']
        X[i][1] = a['y']
        if a['labelName'] == "neutral":
            y[i] = -1
            normal = normal + 1
        elif a['labelName'] == 'plus':
            y[i] = 1
            anom = anom + 1
        else:
            raise RuntimeError("Only classes neutral and plus can be used")

    unc = set_up_uncertainty(X, y)

    if q_data_set_up:
        plt.scatter(X[:, 0], X[:, 1], c=y)
        plt.title("Sketch: " + str(nbr))
        plt.show()

    if user_unc_plot:
        plt.scatter(X[:, 0], X[:, 1], c=unc)
        plt.title("Sketch " + str(nbr) + ": true user uncertainties")
        plt.colorbar()
        plt.show()

    return Data(X, y, unc, (anom / (anom + normal)), [], [], [], two_d_flag=1, name="Sketch2D")


def set_up_Lorenzo_data():
    np.random.seed(3)
    n = 500
    class_prior = 0.9
    contamination = 1 - class_prior 
    a1_ = np.random.randn(2, np.int(n * (1 - class_prior) / 6)) * 1.4
    a2_ = np.random.randn(2, np.int(n * (1 - class_prior) / 6)) * 1.4
    a3_ = np.random.randn(2, np.int(n * (1 - class_prior) / 6)) * 1.4
    a4_ = np.random.randn(2, np.int(n * (1 - class_prior) / 6)) * 1.4
    a5_ = np.random.randn(2, np.int(n * (1 - class_prior) / 6)) * 1.4
    a6_ = np.random.randn(2, np.int(n * (1 - class_prior) / 6)) * 1.4
    num_anom = a1_.shape[1] + a2_.shape[1] + a3_.shape[1] + a4_.shape[1] + a5_.shape[1] + a6_.shape[1]

    d1_ = np.random.randn(2, np.int(n * 0.6)) * 1.3
    n_ = n - d1_.shape[1] - num_anom
    d2_ = np.random.randn(2, n_) * 1.2

    d2_[0, :] += 5

    a1_[0, :] += 6.5
    a1_[1, :] -= 6.5
    a2_[0, :] -= 4.
    a2_[1, :] -= 6.5
    a3_[0, :] += 1.
    a3_[1, :] -= 6.5
    a4_[0, :] += 6.5
    a4_[1, :] += 6.5
    a5_[0, :] -= 4.
    a5_[1, :] += 6.5
    a6_[0, :] += 1.5
    a6_[1, :] += 6.5

    data_set = np.concatenate((a1_, a2_, a3_, a4_, a5_, a6_, d1_, d2_), axis=1)
    X_train = data_set.T
    y = np.zeros(n, dtype=np.int)
    y[:num_anom] = +1
    for i in range(len(y)):
        if y[i] == 0:
            y[i] = -1 
    fig = plt.figure(figsize=(6, 5), facecolor='w', edgecolor='k')
    colors = np.array(['b' if y[i] == 0 else 'r' for i in range(n)]) 
    plt.scatter(X_train.T[0], X_train.T[1], 40, colors, alpha=0.8)
    plt.show()

    from sklearn.svm import SVC
    from sklearn.calibration import CalibratedClassifierCV

    clf = CalibratedClassifierCV(SVC(C=0.01, kernel='poly', degree=2), method='sigmoid', cv=10).fit(X_train, y)
    pos_prob = clf.predict_proba(X_train)

    predict_probs = [max(q[0], q[1]) for q in pos_prob]
    plt.scatter(X_train.T[0], X_train.T[1], c = predict_probs)
    plt.show()

    X_train, X_test, y_train, y_test, train_uncertainty, test_uncertainty = train_test_split(X_train, y, predict_probs,
                                                                                             train_size=0.7,
                                                                                             test_size=0.3,
                                                                                             random_state=r_rs[-1],
                                                                                             stratify=y)


    return Data(X_train, y_train, train_uncertainty, contamination, X_test, y_test, test_uncertainty, two_d_flag=1, name="LorenzoData")



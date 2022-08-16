import math
import random

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, mean_squared_error, f1_score, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

import Parameters
import Strategies
from Parameters import *
import deprecation
from scipy.stats import beta

class Data:
    def __init__(self, X, true_labels, true_user_uncertainty, contamination, test_X, test_true_labels, test_usr_unc, xx, yy, uncunc, foldtrain, foldtest, it,
                 rs=1, two_d_flag=1, name="not_specified", k_fold=True, rep=1):

        self.xx = xx
        self.yy = yy
        self.uncunc = uncunc
        self.it = it
        self.foldtrain = foldtrain
        self.foldtest = foldtest
        self.k_fold = k_fold

        if k_fold:
            foldtrain_curr = foldtrain[it]
            self.X = xx[foldtrain_curr] 
            self.true_labels = yy[foldtrain_curr] 
            self.true_user_certainty = uncunc[foldtrain_curr]  

            self.test_true_labels = yy[foldtest[it]] 
            self.test_X = xx[foldtest[it]] 
            self.test_usr_unc = uncunc[foldtest[it]]        

            sc = StandardScaler()
            self.X = sc.fit_transform(xx[foldtrain_curr])
            self.test_X = sc.transform(xx[foldtest[it]])
  
        else:
            print("! not k-fold")
            self.X = X 
            self.test_true_labels = test_true_labels
            self.test_X = test_X
            self.true_labels = true_labels
            self.true_user_certainty = true_user_uncertainty
            self.test_usr_unc = test_usr_unc

        self.name = name
        self.est_labels = np.zeros(len(self.X))
        self.est_user_uncertainty = np.zeros(len(self.X))
        self.known_labels = np.zeros(len(self.X))
        self.times_labeled = np.zeros(len(self.X))
        self.times_idk = np.zeros(len(self.X))
        self._rs = 4
        self.queried = np.zeros(len(self.X)) 
        self.query_results = np.zeros(len(self.X)) 
        self.query_chance = np.zeros(len(self.X)) 
        self.query_expected_chance = np.zeros(len(self.X)) 
        self.query_counter = 0  
        self.model_uncertainty = np.zeros(len(self.X))
        self.query_order = []

        self.deterministic_idk = np.zeros(len(self.X))
        random.seed(r_rs[rep])
        for i in range(len(self.deterministic_idk)):
            r = random.random()
            self.deterministic_idk[i] = (r > self.true_user_certainty[i])

        self.auc = []
        self.f1 = []
        self.rmse = []
        self.mined_uncertainty = []
        self.score_of_queries = []
        self.contamination = contamination

        self.wasted_query = [] 
        self.two_d_flag = two_d_flag
        self.last_queried = -1

        self.homogenity = []
        self.lokalisatie = []
        self.ongekendheid = []
        self.gemiddelde_ongekendheid = []

    def get_name(self):
        return str(self.name)

    def get_trained_points_and_labels(self):
        x_train = []
        y_train = []
        for i in range(len(self.X)):
            if self.times_labeled[i] != 0:
                x_train.append(self.X[i])
                y_train.append(self.known_labels[i])
        return [x_train, y_train]

    def getX(self):
        return self.X

    def get_test_X(self):
        return self.test_X

    def get_est_uses_uncertainty(self):
        return self.est_user_uncertainty

    def get_est_labels(self):
        return self.est_labels

    def set_est_labels(self, predict):
        self.est_labels = predict

    def get_history_auc(self):
        return self.auc.copy()

    def get_history_f1(self):
        return self.f1.copy()

    def get_auc(self, prediction):
        if not self.k_fold:
            print('AUC without k-fold!!')
        if split:
            comp = self.test_true_labels
        else:
            comp = self.true_labels
        return roc_auc_score(comp, prediction) 

    def get_f1(self, prediction):
        if split:
            comp = self.test_true_labels
        else:
            comp = self.true_labels
        return f1_score(comp, prediction)

    def add_auc(self, prediction):
        self.auc.append(self.get_auc(prediction)) 

    def get_history_rmse(self):
        return self.rmse.copy()

    def get_rmse(self):
        mse = mean_squared_error(self.true_user_certainty, self.est_user_uncertainty)
        return math.sqrt(mse)

    def add_rmse(self):
        self.rmse.append(self.get_rmse())

    def query(self, index):
        self.last_queried = index
        known = False 
        self.queried[index] = True

        if deterministic:
            criteria = self.deterministic_idk[index]
        else:
            r = random.random()
            criteria = r > self.true_user_certainty[index]

        if criteria: 
            self.times_idk[index] = self.times_idk[index] + 1
            self.wasted_query.append(1)
            known = False
        else: 
            self.times_labeled[index] = self.times_labeled[index] + 1
            self.known_labels[index] = self.true_labels[index]
            self.wasted_query.append(0)
            known = True

        if self.true_labels[index] == -1: 
            qr = 1
        else: 
            qr = 2
        if not known:
            qr = qr + 2
        self.query_results[self.query_counter] = qr
        self.query_chance[self.query_counter] = self.true_user_certainty[index]
        self.query_expected_chance[self.query_counter] = self.est_user_uncertainty[index]

        self.query_counter = self.query_counter + 1
        self.query_order.append(index)
        return known

    def is_idk(self, index):
        if self.times_idk[index] != 0:
            if self.times_labeled[index] == 0:
                return True
        return False

    def is_known(self, index):
        return self.times_labeled[index] != 0

    def is_not_queried(self, index):
        return not self.queried[index]

    def get_known_labels(self):
        return self.known_labels  
    def fresh_copy(self, shuffled=False, iteration_nbr=0, repeat=False, repe=1):
        X = self.X.copy()
        y = self.true_labels.copy()
        test_X = self.test_X.copy()
        test_y = self.test_true_labels.copy()
        unc = self.true_user_certainty.copy()
        test_unc = self.test_usr_unc
        xx = self.xx.copy()
        yy = self.yy.copy()
        uncunc = self.uncunc.copy()

        if shuffled:
            print('Iteration: '+ str(iteration_nbr))
            d = Data(X, y, unc, self.contamination, test_X, test_y, test_unc, two_d_flag=self.two_d_flag, name=self.get_name(),
            xx=self.xx, yy=self.yy, uncunc=self.uncunc, it=iteration_nbr, foldtrain=self.foldtrain, foldtest=self.foldtest, k_fold=True)
            return d 
        if not repeat:
            new_data = Data(X, y, unc, self.contamination, test_X, test_y, test_unc, two_d_flag=self.two_d_flag, name=self.get_name(), xx=xx, yy=yy, uncunc=uncunc, k_fold=True, foldtrain=self.foldtrain, foldtest=self.foldtest, it=self.it)
        else:
            new_data = Data(X, y, unc, self.contamination, test_X, test_y, test_unc, two_d_flag=self.two_d_flag,
                            name=self.get_name(), xx=xx, yy=yy, uncunc=uncunc, k_fold=True, foldtrain=self.foldtrain,
                            foldtest=self.foldtest, it=iteration_nbr, rep=repe)
        return new_data

    @deprecation.deprecated
    def _shuffle(self): 
        self.X, self.true_labels, self.true_user_certainty, self.contamination, self.test_X, self.test_true_labels, self.test_usr_unc = \
            shuffle(self.X, self.true_labels, self.true_user_certainty, self.contamination, self.test_X, self.test_true_labels,
                    self.test_usr_unc, random_state=self._rs)

    def shuffle_copy(self):
        data_new = self.fresh_copy()
        data_new._shuffle()
        return data_new

    def get_user_uncertainty_labels(self):
        a = np.zeros(len(self.X))
        for i in range(len(self.X)):
            if self.is_idk(i):
                a[i] = 1
            elif self.is_known(i):
                a[i] = -1
        return a

    def most_uncertain_not_queried(self, uncertainty, model_uncertainty=-1):
        srt = np.argsort(uncertainty)[::-1] 
        for i in range(len(uncertainty)):
            idx = srt[i]
            if not self.queried[idx]:
                if model_uncertainty == -1:
                    self.model_uncertainty[self.query_counter] = -1
                else:
                    self.model_uncertainty[self.query_counter] = model_uncertainty[idx]
                return idx
        raise RuntimeError("No not queried elements found")


    def probabilistic_most_uncertain_not_queried(self, uncertainty, model_uncertainty=-1):
        probs = np.copy(uncertainty)
        nbr = list(range(0, len(uncertainty)))
        for i in range(len(uncertainty)):
            if self.queried[i]:
                probs[i] = 0

        srt = np.argsort(uncertainty)[::-1]
        str = srt[:]

        draw = random.choices(nbr, weights=probs, k=1) 
        idx = draw[0]
        if model_uncertainty == -1:
            self.model_uncertainty[self.query_counter] = -1
        else:
            self.model_uncertainty[self.query_counter] = model_uncertainty[idx]
        return idx   
    def most_uncertain_not_queried_above_treshold(self, uncertainty, treshold):
        srt = np.argsort(uncertainty)
        for i in range(len(uncertainty)):
            idx = srt[i]
            if not self.queried[idx]:
                if self.est_user_uncertainty[idx] > treshold:
                    return idx, True
        print("No suitable point to query found, query most uncertain (model) instead")
        return self.most_uncertain_not_queried(uncertainty), False

    def print_avg_uncertainty(self):
        print("Average uncertainty: " + str(np.mean(self.true_user_certainty)))
        anomaly_uncertainty = []
        for i in range(0, len(self.X)):
            if self.true_labels[i] == -1:
                anomaly_uncertainty.append(self.true_user_certainty[i])
        print("Average anomaly uncertainty: " + str(np.mean(anomaly_uncertainty)))


def query_lowest_predict_proba_return_reward(data, predict_proba, rewards, model_uncertainty):
    idx = query_highest_predict_proba(data, predict_proba, model_uncertainty)
    return rewards[idx]


def query_highest_predict_proba(data, predict_proba, model_uncertainty):
    if not probabilistic_sampling:
        idx = data.most_uncertain_not_queried(predict_proba, model_uncertainty)
    else:
        idx = data.probabilistic_most_uncertain_not_queried(predict_proba, model_uncertainty)
    data.query(idx)

    score = Strategies.combine_user_model_unc(data.true_user_certainty[idx],model_uncertainty[idx], Parameters.gain_comb[0]) 
    data.score_of_queries.append(score)
    if data.deterministic_idk[idx]:
        data.mined_uncertainty.append(0) 
    else:
        model_uncertainty = model_uncertainty[idx] 
        data.mined_uncertainty.append(model_uncertainty)
    return idx


def query_lucky(data, predict_proba, model_uncertainty):
    lucky = False
    while not lucky:
        idx = data.most_uncertain_not_queried(predict_proba, model_uncertainty)
        if data.query(idx):
            lucky = True
            data.mined_uncertainty.append(predict_proba[idx])


def query_most_true_uncertain(data):
    idx = data.most_uncertain_not_queried(uncertainty=data.true_user_certainty)
    data.query(idx)


def update_r_rs():
    global r_rs
    r_rs = r_rs + 1


def query_most_uncertain_above_treshold(data, predict_proba, treshold):
    most_uncertain_not_queried = data.most_uncertain_not_queried(predict_proba)
    idx, found = data.most_uncertain_not_queried_above_treshold(predict_proba, treshold)
    diff = predict_proba[idx] - predict_proba[most_uncertain_not_queried]
    return data.query(idx), diff, found


class KnnUserUncertainty(Data):
    def __init__(self, d, nbr_neighbours): 
        super().__init__(d.X, d.true_labels, d.true_user_certainty, d.contamination, d.test_X, d.test_true_labels,
                         d.test_usr_unc, two_d_flag=d.two_d_flag, xx=d.xx, yy=d.yy, uncunc=d.uncunc, foldtrain=d.foldtrain, foldtest=d.foldtest, it=d.it)

        nbrs = NearestNeighbors(n_neighbors=nbr_neighbours, algorithm='ball_tree').fit(d.X)
        self.distances, self.indices = nbrs.kneighbors(d.X)

    def shuffle_copy(self): 
        raise NotImplementedError("It's not possible to get a shuffle copy with knn user uncertainty")

    def fresh_copy(self, shuffled=False, iteration_nbr=0, repeat=False): 
        raise NotImplementedError("It's not possible to get a copy with knn user uncertainty")

    def update_est_user_uncertainty(self):
        est_user_uncertainty = np.zeros(len(self.indices))
        for i in range(len(self.indices)):
            idk_count = 0
            labels_count = 0 
            for j in range(len(self.indices[0])):
                ind = self.indices[i][j] 
                if self.is_idk(ind):
                    idk_count = idk_count + 1
                    labels_count = labels_count + 1 
                elif self.is_not_queried(ind):
                    pass 
                elif self.is_known(ind):
                    labels_count = labels_count + 1
                else:
                    raise RuntimeError("User uncertainty has unknown label")
            if labels_count == 0:
                est_user_uncertainty[i] = 0
            else:
                est_user_uncertainty[i] = 1 - (idk_count / labels_count)
        self.est_user_uncertainty = est_user_uncertainty

    def query(self, index):
        a = super().query(index)
        self.update_est_user_uncertainty()
        self.add_rmse()
        return a

    def nbr_count(self):
        counts = np.zeros(len(self.X))
        for i in range(len(self.X)):
            curr_count = 0
            for j in self.indices[i]:
                if self.is_known(j):
                    curr_count = curr_count + 1
            counts[i] = curr_count

        return counts


class KnnPriorDistanceAndLabelInformation2(KnnUserUncertainty):
    def __init__(self, d, nbr_neighbours):
        super().__init__(d, nbr_neighbours)

    def update_est_user_uncertainty(self):
        est_user_uncertainty = np.zeros(len(self.indices))

        known = 0
        idk = 0
        for i in range(len(self.get_known_labels())):
            if self.is_idk(i):
                idk = idk + 1 
            if self.is_known(i):
                known = known + 1 
        avg_user_uncertainty = 1 - (known / (idk + known))
        if avg_user_uncertainty == 0:
            avg_user_uncertainty = 0.5
        self.gemiddelde_ongekendheid.append(avg_user_uncertainty) 
        for i in range(len(self.indices)):
            dist = self.distances[i]
            indices = self.indices[i]
            est_user_uncertainty[i] = self.est_user_uncertainty_of_point(avg_user_uncertainty, dist, indices)
        self.est_user_uncertainty = est_user_uncertainty

    def est_user_uncertainty_of_point(self, avg_user_uncertainty, dist, indices): 
        if avg_user_uncertainty == 0:
            avg_user_uncertainty = 0.5
        at_least_one = False
        dist_idk_count = 0
        dist_queried_count = 0
        nq_dist_to_not_queried_nbrs_count = []
        nq_dist_to_all_neighbours_count = []
        known_nbs = 0
        labels_in_group = []
        anom_dist = 0
        normal_dist = 0
        label_dist = 0

        debug_counter = 0 
        nbr_neighbrs = len(self.indices[0])
        for j in range(0, nbr_neighbrs):
            ind = indices[j]
            curr_dist_w = math.pow(dist[j], 2)
            if curr_dist_w == 0:
                curr_dist_w = 0.0001 
            nq_dist_to_all_neighbours_count.append(1/curr_dist_w) 
            if self.is_idk(ind):
                known_nbs = known_nbs + 1
                dist_idk_count = dist_idk_count + (1/curr_dist_w)
                dist_queried_count = dist_queried_count + (1/curr_dist_w) 
            elif self.is_not_queried(ind):
                debug_counter = debug_counter + 1
                nq_dist_to_not_queried_nbrs_count.append(1/curr_dist_w) 
            elif self.is_known(ind):
                known_nbs = known_nbs + 1
                dist_queried_count = dist_queried_count + (1/curr_dist_w)

                curr_label = self.get_known_labels()[ind]
                labels_in_group.append(curr_label)
                if curr_label == -1: 
                    normal_dist = normal_dist + 1/curr_dist_w
                    label_dist = label_dist + 1/curr_dist_w
                elif curr_label == 1: 
                    anom_dist = anom_dist + 1/curr_dist_w
                    label_dist = label_dist + 1/curr_dist_w
                else:
                    raise RuntimeError("Known label is not normal and not anomaly")
            else:
                raise RuntimeError("User uncertainty has unknown label")
        if dist_queried_count == 0 and nq_dist_to_all_neighbours_count == []:
            self.homogenity.append(avg_user_uncertainty)
            self.lokalisatie.append(avg_user_uncertainty)
            self.ongekendheid.append(avg_user_uncertainty)
            return avg_user_uncertainty

        else:
            fraction_labeled = known_nbs / nbr_neighbrs
            nq_dist_to_not_queried_nbrs_count_tot = sum(nq_dist_to_not_queried_nbrs_count)
            nq_dist_to_all_neighbours_count_tot = sum(nq_dist_to_all_neighbours_count)
            if dist_queried_count == 0:
                self.homogenity.append(avg_user_uncertainty)
                self.lokalisatie.append(avg_user_uncertainty)
                self.ongekendheid.append(avg_user_uncertainty)
                return avg_user_uncertainty

            lokalisatie = label_dist/(dist_idk_count+label_dist) 
            if dist_idk_count == 0:
                lokalisatie = 1
                if label_dist == 0:
                    lokalisatie = 0.5
            elif label_dist == 0:
                lokalisatie = 0
            ongekendheid = (nq_dist_to_not_queried_nbrs_count_tot / nq_dist_to_all_neighbours_count_tot)

            if len(labels_in_group) == 0:
                homogeniteit = lokalisatie
            else:
                homogenity = max(normal_dist / label_dist, anom_dist / label_dist)
                homogenity = abs((normal_dist/label_dist) - (anom_dist/label_dist))
                homogeniteit = homogenity

            weight = beta.cdf(ongekendheid, 1, 4, loc=0, scale=1)  
            unc = (homogeniteit/2 + lokalisatie/2) * (weight) + (1-weight) * avg_user_uncertainty

            self.homogenity.append(homogeniteit)
            self.lokalisatie.append(lokalisatie)
            self.ongekendheid.append(ongekendheid)


            return unc


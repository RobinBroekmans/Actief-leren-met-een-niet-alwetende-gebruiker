from anomatools.models import SSkNNO, SSDO
from sklearn.ensemble import IsolationForest
from data import *
from data_set_up import *
from Parameters import *
import math
from plots import user_vs_model_uncertainty, heatmap_est_user_uncertainty, heatmap_comb_model_and_useer_unc, \
    heatmap_diff_est_true_user_unc, heatmaps_subplots
from plots import heatmap

e = 0.00001

 
'''
utility / cost
ROI: utility/cost - 1
Multi-task active learning (related but not really useful)

'''

 
def get_model_uncertainty(m_unc_nbr, predict_proba):
    if m_unc_nbr == 1:
        predict_proba_m = [(1-abs(x[0] - x[1])) for x in predict_proba]
    elif m_unc_nbr == 2:
        predict_proba_m = [1-max(x[0], x[1]) for x in predict_proba]
    elif m_unc_nbr == 3:
        predict_proba_m = [min(x[0], x[1]) * -1 for x in predict_proba]
    elif m_unc_nbr == 4:
        predict_proba_m = [(x[0] * x[1]) for x in predict_proba]
    elif m_unc_nbr == 5:
        predict_proba_m = [x[1] for x in predict_proba]
    elif m_unc_nbr == 6:
        predict_proba_m = [x[0] for x in predict_proba]
    elif m_unc_nbr == 7:
        pp = predict_proba
        predict_proba_m = []
        for k in range(len(predict_proba)):
            p_norm = pp[k][0]
            p_anom = pp[k][1]
            if p_norm != 0:
                e_0 = p_norm * math.log2(p_norm)
            else:
                e_0 = 0
            if p_anom != 0:
                e_1 = p_anom * math.log2(p_anom)
            else:
                e_1 = 0
            predict_proba_m.append(-(e_0+e_1))

    else:
        print("Illegal number: " + str(m_unc_nbr))
    return predict_proba_m


 
 
def combine_user_model_unc_array(predict_proba_m, predict_proba_u, combination_nbr):
    for k in range(len(predict_proba_u)):
        predict_proba_u[k] = predict_proba_u[k]
    predict_proba = np.zeros(len(predict_proba_m))
    for j in range(len(predict_proba_m)):
        curr_prob = combine_user_model_unc(predict_proba_m[j], predict_proba_u[j], combination_nbr)
        predict_proba[j] = curr_prob
    return predict_proba


def combine_user_model_unc(predict_proba_m, predict_proba_u, combination_nbr):
    if combination_nbr == 1:
        curr_prob = ((predict_proba_m) * (predict_proba_u))
        curr_prob = (math.pow((predict_proba_m), p_gamma) * math.pow((predict_proba_u), 1))
    elif combination_nbr == 2:
        curr_prob = predict_proba_m * w_m_imp + predict_proba_u * w_usr_imp
    elif combination_nbr == 3:
        curr_prob = ((predict_proba_m + e) ** 2 + (predict_proba_u + e)) ** 2
    elif combination_nbr == 4:
        curr_prob = (predict_proba_m + e) / (predict_proba_u + e)
    return curr_prob


 
def random_order_of_point(total_points):
    order_of_points = []
    for j in range(total_points):
        order_of_points.append(j)
    ret = shuffle(order_of_points)

    return ret


def auc_Iforest(data, budget):
    c = data.contamination
    forest = IsolationForest(random_state=0, contamination=c).fit(data.getX())
    if split:
        pr = data.get_test_X()
    else:
        pr = data.getX()
    predict = forest.predict(pr) * -1
    auc = np.full(budget, data.get_auc(predict))
    return auc


def auc_unsup_SSkNNO(data, budget):
    c = data.contamination
    if m_ssdo:
        '''prior_detector = IsolationForest(n_estimators=100, contamination=c, behaviour='new')
        prior_detector.fit(data.getX())
        tr_prior = prior_detector.decision_function(data.getX()) * -1
        tr_prior = tr_prior + abs(min(tr_prior))
        detector = SSDO(contamination=c, unsupervised_prior='other').fit(data.getX(), prior=tr_prior)
        '''
        detector = SSDO(contamination=c).fit(data.getX())
    else:
        detector = SSkNNO(contamination=c).fit(data.getX())
    if split:
        pr = data.get_test_X()
    else:
        pr = data.getX() 
    predict = detector.predict_proba(pr, method='squash')[:, 1]
    auc = np.full(budget, data.get_auc(predict))
    return auc


def auc_sup_SSkNNO(data, budget):
    c = data.contamination
    if m_ssdo:
        detector = SSDO(contamination=c)
    else:
        detector = SSkNNO(contamination=c)
    detector.fit(data.getX(), data.true_labels)
    if split:
        pr = data.get_test_X()
    else:
        pr = data.getX() 
    predict = detector.predict_proba(pr, method='squash')[:, 1]
    auc = np.full(budget, data.get_auc(predict))
    return auc


def auc_random_sampling(data, budget):
    X = data.getX()
    c = data.contamination
    order = random_order_of_point(len(X))
    reward = []
    budget = budget-1
    if m_ssdo:
        detector = SSDO(contamination=c)
    else:
        detector = SSkNNO(contamination=c, weighted=True)
    detector.fit(X, data.get_known_labels())

    if m_ssdo:
        detector = SSDO(contamination=c)
    else:
        detector = SSkNNO(contamination=c, weighted=True)
    detector.fit(X, data.get_known_labels())
    if split:
        pr = data.get_test_X()
    else:
        pr = data.getX() 
    predict = detector.predict_proba(pr, method='squash')[:, 1]
    data.add_auc(predict)

    for i in range(budget):
        curr_index = order[i]
        rewards = detector.predict_proba(X)
        data.query(curr_index)

        if m_ssdo:
            detector = SSDO(contamination=c)
        else:
            detector = SSkNNO(contamination=c, weighted=True)
        detector.fit(X, data.get_known_labels())
        if split:
            pr = data.get_test_X()
        else:
            pr = data.getX() 
        predict = detector.predict_proba(pr, method='squash')[:, 1]
        data.add_auc(predict)

        reward.append(abs(rewards[curr_index][0] - rewards[curr_index][1]))



        if i % 20 == 0:
            print("Random episode: " + str(i) + " of " + str(budget))
    return data.get_history_auc(), reward


def all_knowing(data, budget, combination_nbr, m_unc_nbr):
    c = data.contamination
    X = data.getX()
    m_uncertainty = []
    for i in range(budget): 
        if m_ssdo:
            detector = SSDO(contamination=c)
        else:
            detector = SSkNNO(contamination=c, weighted=True)
        detector.fit(X, y=data.get_known_labels())
        predict_proba = detector.predict_proba(X, method='squash')
        if split:
            pr = data.get_test_X()
        else:
            pr = data.getX() 
        predict = detector.predict_proba(pr, method='squash')[:, 1]
        data.add_auc(predict) 
        predict_proba_m = get_model_uncertainty(m_unc_nbr, predict_proba) 
        predict_proba_u = data.true_user_certainty
        predict_proba = combine_user_model_unc_array(predict_proba_m, predict_proba_u, combination_nbr)

        m_uncertainty.append(query_lowest_predict_proba_return_reward(data, predict_proba, predict_proba_m, predict_proba_m))

        if i % 20 == 0:
            print("All knowing: " + str(i) + " of " + str(budget))     

    return m_uncertainty


def auc_uncertainty_sampling_ssKNNo(data, budget, m_unc_nbr):
    auc = []
    X = data.getX()
    c = data.contamination
    for i in range(budget): 
        if m_ssdo:
            detector = SSDO(contamination=c)
        else:
            detector = SSkNNO(contamination=c, weighted=True) 
        detector.fit(X, y=data.get_known_labels())
        predict_proba = detector.predict_proba(X, method='squash')
        if split:
            pr = data.get_test_X()
        else:
            pr = data.getX()
        predict = detector.predict_proba(pr, method='squash')[:,1]
        data.add_auc(predict)
        auc.append(data.get_auc(predict)) 
        predict_proba = get_model_uncertainty(m_unc_nbr, predict_proba)

        query_highest_predict_proba(data, predict_proba, predict_proba) 

        if i % 20 == 0: 
            print("ssKNNo UC: " + str(i) + " of " + str(budget))  

        '''
        for i in range(5):
            detector = SSkNNO(contamination=c, weighted=True)
            detector.fit(X)
            pr = data.xx[data.foldtest[i]]
            prediction = detector.predict(pr)

            comp = data.yy[data.foldtest[i]]
            a = roc_auc_score(comp, prediction)
            print(a)
        '''



    return auc


def lucky_strat(data, budget, m_unc_nbr=1):
    c = data.contamination
    X = data.getX()
    for i in range(budget): 
        if m_ssdo:
            detector = SSDO(contamination=c)
        else:
            detector = SSkNNO(contamination=c, weighted=True)
        detector.fit(X, data.get_known_labels())
        predict_proba = detector.predict_proba(X, method='squash')
        if split:
            pr = data.get_test_X()
        else:
            pr = data.getX() 
        predict = detector.predict_proba(pr, method='squash')[:, 1]
        data.add_auc(predict) 
        predict_proba = get_model_uncertainty(m_unc_nbr, predict_proba)
        query_lucky(data, predict_proba, predict_proba)

        if i % 50 == 0: 
            print("Lucky strat: " + str(i) + " of " + str(budget))

    return data.get_history_auc()


 
def gain_strat(data, budget, combination_nbr, m_unc_nbr):
    c = data.contamination
    X = data.getX()
    for i in range(budget): 
        if m_ssdo:
            detector = SSDO(contamination=c)
        else:
            detector = SSkNNO(contamination=c, weighted=True)
        detector.fit(X, data.get_known_labels())
        predict_proba = detector.predict_proba(X, method='squash')
        if split:
            pr = data.get_test_X()
        else:
            pr = data.getX() 
        predict = detector.predict_proba(pr, method='squash')[:, 1]
        data.add_auc(predict) 
        predict_proba_m = get_model_uncertainty(m_unc_nbr, predict_proba)
        model_uncertainty = predict_proba_m 
        predict_proba_u = [x for x in data.get_est_uses_uncertainty()]
        for k in predict_proba_u:
            if k < 0:
                print("WWWWW")

        predict_proba = combine_user_model_unc_array(predict_proba_m, predict_proba_u, combination_nbr)

        query_highest_predict_proba(data, predict_proba, model_uncertainty)

        if i % 1 == 0: 
            print("Gain episode: " + str(i) + " of " + str(budget))     
            heatmaps_subplots(data, m_unc_nbr, combination_nbr, "Gain", i)

    return data.get_history_auc()
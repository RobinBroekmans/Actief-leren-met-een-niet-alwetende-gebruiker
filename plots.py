import time
import datetime
import numpy
from matplotlib.pyplot import colorbar
from sklearn.neighbors._ball_tree import BallTree

import Strategies
from Strategies import *
from Parameters import *


def plot(data, budget, title):
    start = time.time()
    queries = []
    d = data.fresh_copy()
    for i in range(budget):
        queries.append(i)

    if user_model_uncertainty_plot:
        user_vs_model_uncertainty(data, title)

    if p_baseline_sup: 
        print("Supervised baseline")
        sum_auc = np.zeros_like(queries) 
        for i in range(p_repetitions):
            d1 = d.fresh_copy(True, i)
            data_bsup = KnnDistanceUserUncertainty(d1, nbs)
            auc_bsup = auc_sup_SSkNNO(data_bsup, budget) 
            sum_auc = np.add(sum_auc, auc_bsup) 
        auc_bsup = np.true_divide(sum_auc, p_repetitions) 

    if p_baseline_unsup: 
        print("Unsupervised baseline")
        sum_auc = np.zeros_like(queries)
        for i in range(p_repetitions):
            d1 = d.fresh_copy(True, i)
            data_bunsup = KnnDistanceUserUncertainty(d1, nbs)
            auc_unsup = auc_unsup_SSkNNO(data_bunsup, budget)
            sum_auc = np.add(sum_auc, auc_unsup)
        auc_unsup = np.true_divide(sum_auc, p_repetitions)

    if p_baseline_isol_unsup: 
        print("Unsupervised isolation forest baseline")
        sum_auc = np.zeros_like(queries)
        for i in range(p_repetitions):
            d1 = d.fresh_copy(True, i)
            data_bisounsup = KnnDistanceUserUncertainty(d1, nbs)
            auc_bisounsup = auc_Iforest(data_bisounsup, budget)
            sum_auc = np.add(sum_auc, auc_bisounsup)
        auc_bisounsup = np.true_divide(sum_auc, p_repetitions)

    if p_lucky:
        sum_auc = np.zeros_like(queries)
        for i in range(p_repetitions):
            d1 = d.fresh_copy(True, i)
            data_lucky = lucky(budget, d1)
            z_auc_lucky = data_lucky.get_history_auc()
            write_to_file(d1.get_name(), "Lucky", z_auc_lucky)
            sum_auc = np.add(sum_auc, data_lucky.get_history_auc())
        auc_lucky = np.true_divide(sum_auc, p_repetitions)

    if p_us:
        sum_auc = np.zeros_like(queries)
        auc_array = [] 
        aantal_shuffles = p_repetitions/n_splits
        split_counter = -1
        wq_us = []
        for i in range(p_repetitions): 
            t_i = int(i / aantal_shuffles)
            d1 = d.fresh_copy(True, i)
            split_counter = split_counter + 1  
            print('Split: ' + str(d1.it) + "  Part: " + str(int(i % aantal_shuffles)))
            if len(gain_m) > 1:
                raise RuntimeWarning("!!! Uncertainty sampling only takes the first combination number and can not work with more!!")
            auc_uc_ssknno, data_ssknno = us(budget, d1, queries, gain_m[0])

            sum_auc = np.add(sum_auc,auc_uc_ssknno)
            auc_array.append(auc_uc_ssknno)
            wq_us.append(np.cumsum(data_ssknno.wasted_query))
            if save:
                write_to_file(d1.get_name(), "UncertaintySampling", auc_uc_ssknno) 
        auc_uc_ssknno = np.true_divide(sum_auc, p_repetitions)

        auc_array = np.array(auc_array)
        std_auc_uc_ssknno = np.std(np.array(auc_array), axis=0)
        auc_uc_ssknno = auc_array.mean(axis=0)
        wq_uc_ssknno = np.array(wq_us).mean(axis=0)
        if save:
            write_to_file(d1.get_name(), "UncertaintySampling", wq_uc_ssknno, wq=True)

    if p_random:
        print("Start random sampling")
        sum_auc = np.zeros_like(queries)
        wq_random = []
        for i in range(p_repetitions):
            d1 = d.fresh_copy(True, i)
            auc_rs, data_rs = random_samp(budget, d1, queries)
            write_to_file(d1.get_name(), "Random", auc_rs)
            sum_auc = np.add(sum_auc, auc_rs)
            wq_random.append(np.cumsum(data_rs.wasted_query))
        auc_rs = np.true_divide(sum_auc, p_repetitions)
        wq_random = np.array(wq_random).mean(axis=0)
        if save:
            write_to_file(d1.get_name(), "Random", wq_random, wq=True)

    if p_gain:
        print("Start gain_strat")
        needed_dim = len(gain_m) * len(gain_comb)
        sum_auc = np.zeros((needed_dim,budget))
        wq_gain = []
        for i in range(p_repetitions):
            d1 = d.fresh_copy(True, i)
            data_s2 = gain(budget, d1, queries)
            for k in range(len(data_s2)):
                z_method_name = "gain" + str(k)
                if save:
                    write_to_file(d1.get_name(), z_method_name, data_s2[k].get_history_auc())
                sum_auc[k] = np.add(sum_auc[k], data_s2[k].get_history_auc())
                wq_gain.append(np.cumsum(data_s2[k].wasted_query))
        auc_gain = np.true_divide(sum_auc, p_repetitions)
        wq_gain = np.array(wq_gain).mean(axis=0)
        if save:
            write_to_file(d1.get_name(), "Gain", wq_gain, wq=True)

    if p_gainreg:
        print("!! Not saved to file!! Strat3")
        d1 = d.fresh_copy()
        data_gr = KnnDistanceUserUncertainty(d1, nbs)
        if q_prior:
            data_gr = KnnPriorAndDistanceUserUncertainty(d1, nbs)
        if q_adv_prior:
            data_gr = KnnGoodPriorAndDistanceUserUncertainty(d1, nbs)
        if q_best_prior:
            data_gr = KnnPriorDistanceAndLabelInformation(d1, nbs)

        auc_gr = gain_strat(data_gr, budget)

        if q_analyse:
            t = "Strat 3"
            exp_user = data_gr.query_expected_chance[:budget]
            true_user = data_gr.query_chance[:budget]
            plot_diff_true_vs_exp_user_uncertainty(queries, t, true_user, exp_user)

            t = "Strat 3: Query result analysis"
            labels = data_gr.query_results[:budget]
            query_results(queries, auc_gr, labels, t)

    if p_allknowing:
        print("Start gain_strat")
        needed_dim = len(gain_m) * len(gain_comb)
        sum_auc = np.zeros((needed_dim, budget))
        wq_allk = []
        for i in range(p_repetitions):
            d1 = d.fresh_copy(True, i)
            data_all_knowing = all_k(budget, d1)
            for k in range(len(data_all_knowing)):
                z_method_name = "All Knowing" + str(k)
                write_to_file(d1.get_name(), z_method_name, data_all_knowing[k].get_history_auc())
                sum_auc[k] = np.add(sum_auc[k], data_all_knowing[k].get_history_auc())
                wq_allk.append(np.cumsum(data_all_knowing[k].wasted_query))
        auc_allK = np.true_divide(sum_auc, p_repetitions)
        wq_allk = np.array(wq_allk).mean(axis=0)

    if auc_plot:
        plt.clf()
        fig, ax1 = plt.subplots()
        ax1.set_ylabel("AUC (volle lijn)")
        ax1.set_xlabel("Aantal queries")
        plt.title(title)

        ax2 = ax1.twinx()
        ax2.set_ylabel('Queries met idk label (stippelijn)')

        if p_us:
            ax1.plot(queries, auc_uc_ssknno, label="Uncertainty", color='C0') 
            ax2.plot(queries, wq_uc_ssknno, '--', label="Uncertainty", color='C0')
            print('uc: ' + str(wq_uc_ssknno[-1]))
        if p_gain:
            ax1.plot(queries, auc_gain[0], label="Gain", color='C1')
            ax2.plot(queries,wq_gain, '--', label="Gain", color='C1')
            print('gain: ' + str(wq_gain[-1]))
        if p_allknowing:
            ax1.plot(queries, auc_allK[0], label='Gain1', color='C1')
            ax2.plot(queries, wq_allk, '--', label="Gain1", color='C1')
            print('gain1: ' + str(wq_allk[-1]))
        ax2.set_ylim([0, queries[-1]])
        ax1.legend(loc=0) 


        if p_lucky:
            plt.plot(queries, auc_lucky, label="Lucky")
        if p_random:
            plt.plot(queries, auc_rs, label="Random")      
        if p_gainreg:
            plt.plot(queries, auc_gr, label="Custom strategy2")      
        if p_baseline_unsup:
            plt.plot(queries, auc_unsup, label="Unsupervised SSkNNO")
        if p_baseline_isol_unsup:
            plt.plot(queries, auc_bisounsup, label="Unsupervised isolation forest")
        if p_baseline_sup:
            plt.plot(queries, auc_bsup, label="Supervised SSkNNO") 
        fig.tight_layout() 
        if p_figsave:
            now = datetime.datetime.now()
            name= 'auc' + str(now.hour) + str(now.minute) + str(now.second) + '.png'
            fig.savefig(name, bbox_inches='tight')
        plt.show()

    if three_factors_plots:
        seperate_3_factors_plot(data_s2[-1])

    if rmse_plot:
        plt.clf()
        plt.ylabel("RMSE")
        plt.xlabel(" 
        plt.title(title + " ")
        if p_us:
            rmse_knn_ssknno = data_ssknno.get_history_rmse()
            plt.plot(queries, rmse_knn_ssknno, label="US")
        if p_random:
            rmse_knn_rs = data_rs.get_history_rmse()
            plt.plot(queries, rmse_knn_rs, label="random")
        if p_gain:
            rmse_knn_s2 = data_s2[0].get_history_rmse()
            plt.plot(queries, rmse_knn_s2, label="Gain")
        if p_allknowing:
            pass

        plt.legend(loc="upper left")
        plt.show()

    if final_rmse_plot:
        plt.clf()
        plt.ylabel("Zoom-in on the end: RMSE")
        plt.xlabel(" 
        plt.title(title + " using knn as user uncertainty model")
        if p_us:
            rmse_knn_ssknno = data_ssknno.get_history_rmse()[-100:]
            plt.plot(queries[-100:], rmse_knn_ssknno, label="ssknno")
        if p_random:
            rmse_knn_rs = data_rs.get_history_rmse()[-100:]
            plt.plot(queries[-100:], rmse_knn_rs, label="random")
        if p_gain:
            rmse_knn_s2 = data_s2.get_history_rmse()[-100:]
            plt.plot(queries[-100:], rmse_knn_s2, label="s2")
        if p_allknowing:
            pass


        plt.legend(loc="upper left")
        plt.show()

    if mined_uncertainty_plot:
        plt.clf()
        plt.ylabel("Uncertainty that got label")
        plt.xlabel(" 
        plt.title(title) 
        '''
        if p_lucky:
            plt.plot(queries, auc_lucky, label="Lucky")
        '''   
        if p_us:
            cs_us = np.cumsum(data_ssknno.mined_uncertainty)
            plt.plot(queries, cs_us, label="Uncertainty") 
        if p_gain:
            cs_gain = np.cumsum(data_s2[-1].mined_uncertainty)
            plt.plot(queries, cs_gain, label="Gain") 
        if p_lucky:
            cs_gain = np.cumsum(data_lucky.mined_uncertainty)
            plt.plot(queries, cs_gain, label="Lucky")

        if p_allknowing:
            cs_allk = np.cumsum(data_all_knowing[-1].mined_uncertainty)
            plt.plot(queries, cs_allk, label="All Knowing") 

        plt.legend(loc="upper left")
        plt.show()

    if wasted_query_plot:
        plt.clf()
        fig, ax1 = plt.subplots()
        ax1.set_ylabel("Uncertainty that got label")
        ax1.set_xlabel(" 
        plt.title(title) 
        '''
        if p_lucky:
            plt.plot(queries, auc_lucky, label="Lucky")
        '''
        if p_us:
            cs_us = np.cumsum(data_ssknno.mined_uncertainty)
            ax1.plot(queries, cs_us, label="Uncertainty")
        if p_gain:
            cs_gain = np.cumsum(data_s2[-1].mined_uncertainty)
            ax1.plot(queries, cs_gain, label="Gain")
        if p_allknowing:
            cs_allk = np.cumsum(data_all_knowing[-1].mined_uncertainty)
            ax1.plot(queries, cs_allk, label="All Knowing")

        ax2 = ax1.twinx()
        ax2.set_ylabel('Queries with no respone')

        if p_us:
            ax2.plot(queries, np.cumsum(data_ssknno.wasted_query), label = "W: Uncertainty", color='yellow')
        if p_gain:
            ax2.plot(queries, np.cumsum(data_s2[-1].wasted_query), label="W: Gain", color='red')
        if p_allknowing:
            ax2.plot(queries, np.cumsum(data_all_knowing[-1].wasted_query), label="W: All Knowing", color='black')
        ax2.set_ylim([0,queries[-1]])
        ax1.legend(loc=0)
        ax2.legend(loc=1)

        fig.tight_layout() 
        plt.show()

    if False: 
        if p_us:
            queried_points_on_map(data_ssknno, "US")
            if p_allknowing:
                compare_queried_points_on_map(data_ssknno, data_all_knowing[-1], "US and All Knowing")
        if p_gain:
            queried_points_on_map(data_s2[-1], "Gain")
        if p_allknowing:
            queried_points_on_map(data_all_knowing[-1], "All knowing")
        if p_lucky:
            queried_points_on_map(data_lucky, "Lucky")


    end = time.time()
    print("Total time: " + str(round((end - start) / 60)) + " minutes")


def write_to_file(name, method, auc_data,wq=False):
    file_name = make_name(name, method, wq=wq)
    with open(file_name, "ab") as f:
        f.write(b"\n")
        numpy.savetxt(f, auc_data, newline=" ", fmt='%10.7f')
        f.close()


def make_name(dataset_name, method, wq=False): 
    aa= "AA"
    if wq:
        aa = aa + "WQ"
    file_name = aa + dataset_name + "_" + method +"gamma=" +str(p_gamma) +"_budget=" + str(p_budget) + "_nbs=" + str(nbs) + \
                "_w_usr_imp=" + str(w_usr_imp) + "_prior_weight=" + str(prior_weight) + "_w_diff_labels=" + str(
        w_diff_labels) + "_gain_comb=" + str(gain_comb) + "_gain_m=" + str(gain_m) + ".csv"
    return file_name


def all_k(budget, d1):
    data_all_knowing = []
    m_uncertainty = []
    for comb_nr in gain_comb: 
        for m_nr in gain_m: 
            print("Start all knowing: " + str((comb_nr - 1) * 4 + m_nr))
            data_all_knowing.append(KnnDistanceUserUncertainty(d1, nbs))
            if q_prior:
                data_all_knowing[-1] = KnnPriorAndDistanceUserUncertainty(d1, nbs)
            elif q_adv_prior:
                data_all_knowing[-1] = KnnGoodPriorAndDistanceUserUncertainty(d1, nbs)
            elif q_best_prior:
                data_all_knowing[-1] = KnnPriorDistanceAndLabelInformation(d1, nbs)
            m_uncertainty = all_knowing(data_all_knowing[-1], budget, comb_nr, m_nr)

            if False:
                plt.title("All knowing: gain_strat")
                k = [a * b for a, b in zip(m_uncertainty, data_all_knowing.query_chance[:budget])]
                di = [0]
                for i in range(1, len(data_all_knowing.get_history_auc())):
                    di.append(data_all_knowing.get_history_auc()[i] - data_all_knowing.get_history_auc()[i - 1]) 
                scatter = plt.scatter(di, k, c=data_all_knowing.query_results[:budget], label="Score") 
                classes = ['L normal', 'L ano', 'IDK normal', 'IDK ano', 'ERROR IN PLOT!!']
                plt.legend(handles=scatter.legend_elements()[0], labels=classes)
                plt.xlabel("AUC diff")
                plt.ylabel("Gain")
                plt.show()
                plt.clf()

                plt.title("All knowing: model uncertainty")
                di = [0]
                for i in range(1, len(data_all_knowing.get_history_auc())):
                    di.append(data_all_knowing.get_history_auc()[i] - data_all_knowing.get_history_auc()[i - 1]) 
                scatter = plt.scatter(di, m_uncertainty, c=data_all_knowing.query_results[:budget], label="Score") 
                classes = ['L normal', 'L ano', 'IDK normal', 'IDK ano', 'ERROR IN PLOT!!']
                plt.legend(handles=scatter.legend_elements()[0], labels=classes)
                plt.xlabel("AUC diff")
                plt.ylabel("Model uncertainty")
                plt.show()
                plt.clf()

                plt.title("All knowing: user uncertainty")
                k = [a * b for a, b in zip(m_uncertainty, data_all_knowing.query_chance[:budget])]
                di = [0]
                for i in range(1, len(data_all_knowing.get_history_auc())):
                    di.append(data_all_knowing.get_history_auc()[i] - data_all_knowing.get_history_auc()[i - 1]) 
                scatter = plt.scatter(di, data_all_knowing.query_chance[:budget],
                                      c=data_all_knowing.query_results[:budget], label="Score") 
                classes = ['L normal', 'L ano', 'IDK normal', 'IDK ano', 'ERROR IN PLOT!!']
                plt.legend(handles=scatter.legend_elements()[0], labels=classes)
                plt.xlabel("AUC diff")
                plt.ylabel("User uncertainty")
                plt.show()
                plt.clf()

                plt.title("All knowing: Query result analysis")
                plt.scatter(queries, data_all_knowing.get_history_auc(), c=data_all_knowing.query_results[:budget]) 
                plt.xlabel(" 
                plt.ylabel("AUC")
                plt.show()
                plt.clf()
    return data_all_knowing


def gain(budget, d, queries):
    data_s2 = []
    m_uncertainty = []
    count = 0
    for comb_nr in gain_comb:
        for m_nr in gain_m:
            print("Strat2: comb_nr = " + str(comb_nr) + " , m_nr = " + str(m_nr))
            count = count + 1
            d1 = d.fresh_copy()

            data_s2.append(KnnDistanceUserUncertainty(d1, nbs))
            if q_prior:
                data_s2[-1] = KnnPriorAndDistanceUserUncertainty(d1, nbs)
            elif q_adv_prior:
                data_s2[-1] = KnnGoodPriorAndDistanceUserUncertainty(d1, nbs)
            elif q_best_prior:
                data_s2[-1] = KnnPriorDistanceAndLabelInformation2(d1, nbs) 
            m_uncertainty = gain_strat(data_s2[-1], budget, comb_nr, m_nr)

            if q_analyse_gain:
                t = "Gain: usr_w:" + str(w_usr_imp) + " hom:" + str(w_diff_labels) + " prior_w:" + str(prior_weight) + " " 
                exp_user = data_s2[-1].query_expected_chance[:budget]
                true_user = data_s2[-1].query_chance[:budget]   

                plot_model_unc_vs_auc_gain(data_s2[-1], t)

                if False:
                    plt.title("Gain: gain_strat")
                    k = [a * b for a, b in zip(m_uncertainty, data_s2.query_chance[:budget])]
                    di = [0]
                    for i in range(1, len(data_s2.get_history_auc())):
                        di.append(data_s2.get_history_auc()[i] - data_s2.get_history_auc()[i - 1]) 
                    scatter = plt.scatter(di, k, c=data_s2.query_results[:budget], label="Score") 
                    classes = ['L normal', 'L ano', 'IDK normal', 'IDK ano', 'ERROR IN PLOT!!']
                    plt.legend(handles=scatter.legend_elements()[0], labels=classes)
                    plt.xlabel("AUC diff")
                    plt.ylabel("Gain")
                    plt.show()
                    plt.clf()

                    plt.title("Gain: model uncertainty")
                    di = [0]
                    for i in range(1, len(data_s2.get_history_auc())):
                        di.append(data_s2.get_history_auc()[i] - data_s2.get_history_auc()[i - 1]) 
                    scatter = plt.scatter(di, m_uncertainty, c=data_s2.query_results[:budget], label="Score") 
                    classes = ['L normal', 'L ano', 'IDK normal', 'IDK ano', 'ERROR IN PLOT!!']
                    plt.legend(handles=scatter.legend_elements()[0], labels=classes)
                    plt.xlabel("AUC diff")
                    plt.ylabel("Model uncertainty")
                    plt.show()
                    plt.clf()

                    plt.title("Gain: user uncertainty")
                    k = [a * b for a, b in zip(m_uncertainty, data_s2.query_chance[:budget])]
                    di = [0]
                    for i in range(1, len(data_s2.get_history_auc())):
                        di.append(data_s2.get_history_auc()[i] - data_s2.get_history_auc()[i - 1]) 
                    scatter = plt.scatter(di, data_s2.query_chance[:budget], c=data_s2.query_results[:budget],
                                          label="Score") 
                    classes = ['L normal', 'L ano', 'IDK normal', 'IDK ano', 'ERROR IN PLOT!!']
                    plt.legend(handles=scatter.legend_elements()[0], labels=classes)
                    plt.xlabel("AUC diff")
                    plt.ylabel("User uncertainty")
                    plt.show()
                    plt.clf()
    return data_s2


def random_samp(budget, d1, queries):
    data_rs = KnnDistanceUserUncertainty(d1, nbs)
    if q_prior:
        data_rs = KnnPriorAndDistanceUserUncertainty(d1, nbs)
    if q_adv_prior:
        data_rs = KnnGoodPriorAndDistanceUserUncertainty(d1, nbs)
    if q_best_prior:
        data_rs = KnnPriorDistanceAndLabelInformation(d1, nbs)
    auc_rs, reward = auc_random_sampling(data_rs, budget)
    if q_analyse:
        t = "Random"
        exp_user = data_rs.query_expected_chance[:budget]
        true_user = data_rs.query_chance[:budget]
        plot_diff_true_vs_exp_user_uncertainty(queries, t, true_user, exp_user)

        t = "Random Query result analysis"
        query_results(queries, auc_rs, data_rs.query_results[:budget], t)
    return auc_rs, data_rs


def lucky(budget, d1):
    print("Start lucky sampling")
    data_lucky = KnnDistanceUserUncertainty(d1, nbs)
    if q_prior:
        data_lucky = KnnPriorAndDistanceUserUncertainty(d1, nbs)
    if q_adv_prior:
        data_lucky = KnnGoodPriorAndDistanceUserUncertainty(d1, nbs)
    if q_best_prior:
        data_lucky = KnnPriorDistanceAndLabelInformation(d1, nbs)
    lucky_strat(data_lucky, budget)
    return data_lucky


def us(budget, d1, queries, m_unc_nbr):
    print("Start uncertainty sampling (SSkNNO)")
    data_ssknno = KnnDistanceUserUncertainty(d1, nbs)
    if q_prior:
        data_ssknno = KnnPriorAndDistanceUserUncertainty(d1, nbs)
    if q_adv_prior:
        data_ssknno = KnnGoodPriorAndDistanceUserUncertainty(d1, nbs)
    if q_best_prior:
        data_ssknno = KnnPriorDistanceAndLabelInformation(d1, nbs)
    auc_uc_ssknno = auc_uncertainty_sampling_ssKNNo(data_ssknno, budget, m_unc_nbr)
    if q_analyse:
        t = "SSkNNO"
        exp_user = data_ssknno.query_expected_chance[:budget]
        true_user = data_ssknno.query_chance[:budget]
        plot_diff_true_vs_exp_user_uncertainty(queries, t, true_user, exp_user)

        t = "SSkNNO: Query result analysis"
        query_results(queries, auc_uc_ssknno, data_ssknno.query_results[:budget], t)

        t = "uncertainty sampling"
        plot_model_unc_vs_auc_gain(data_ssknno, t)
    return auc_uc_ssknno, data_ssknno  
def heatmap(data, m_unc_nbr, strat_name):
    if data.two_d_flag == 0:
        return
    X = data.getX()[:, :2]
    y = data.get_known_labels() 
    clf = SSkNNO(k=10, contamination=data.contamination)
    margin_size = 0.2
    steps = 50

    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()
    x_range = abs(x_max - x_min)
    y_range = abs(y_max - y_min)

    xmin, xmax = x_min - margin_size * x_range, x_max + margin_size * x_range
    ymin, ymax = y_min - margin_size * y_range, y_max + margin_size * y_range 
    xx, yy = np.meshgrid(
        np.linspace(xmin, xmax, int(steps)),
        np.linspace(ymin, ymax, int(steps)))
    X_mesh = np.c_[xx.ravel(), yy.ravel()] 
    if y is None:
        clf.fit(X)
    else:
        clf.fit(X, y) 
    Z = clf.predict_proba(X_mesh)
    Z = Strategies.get_model_uncertainty(m_unc_nbr, Z)
    Z = np.array(Z)
    Z = Z.reshape(xx.shape) 
    plt.figure(figsize=(9,8)) 
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.colorbar()
    plt.title(strat_name + ": heatmap model uncertainty") 
    plt.contour(xx, yy, Z, np.linspace(0.0, np.max(Z), 10)) 
    plt.scatter(X[:, 0], X[:, 1], cmap=plt.cm.coolwarm, s=40, edgecolors='k', c=y)
    plt.show()  
def heatmap_est_user_uncertainty(data):
    if data.two_d_flag == 0:
        return
    X = data.getX()[:, :2]
    y = data.get_known_labels()
    t_idk = data.times_idk * 3
    y = y + t_idk
    c_y = []
    for k in range(len(y)):
        if y[k] == -1:
            c_y.append('red')
        elif y[k] == 0:
            c_y.append('grey')
        elif y[k] == 1:
            c_y.append('green')
        elif y[k] == 3:
            c_y.append('yellow')

    c_y = np.array(c_y)

    margin_size = 0.2
    steps = 50

    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()
    x_range = abs(x_max - x_min)
    y_range = abs(y_max - y_min)

    xmin, xmax = x_min - margin_size * x_range, x_max + margin_size * x_range
    ymin, ymax = y_min - margin_size * y_range, y_max + margin_size * y_range 
    xx, yy = np.meshgrid(
        np.linspace(xmin, xmax, int(steps)),
        np.linspace(ymin, ymax, int(steps)))
    X_mesh = np.c_[xx.ravel(), yy.ravel()]


    avg_user_uncertainty = 0
    tree = BallTree(X, leaf_size=2)
    dist, indices = tree.query(X_mesh, k=nbs) 

    Z = np.zeros(len(X_mesh))
    for z in range(len(Z)):
        Z[z] = data.est_user_uncertainty_of_point(avg_user_uncertainty, dist[z], indices[z])
    Z = Z.reshape(xx.shape) 
    plt.figure(figsize=(9,8)) 
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.clim(0, 1)
    plt.colorbar()
    plt.title("Geschatte expert onzekerheid heatmap")
    warnings.filterwarnings("ignore", message="No contour levels were found within the data range.")
    plt.contour(xx, yy, Z, np.linspace(0.0, np.max(Z), 10)) 
    scatter = plt.scatter(X[:, 0], X[:, 1], s=40, edgecolors='k', c=c_y) 
    plt.show()  
def heatmap_diff_est_true_user_unc(data):
    Z_est = data.est_user_uncertainty
    Z_true = data.true_user_certainty
    Z = abs(Z_true - Z_est)

    X = data.getX()
    plt.scatter(X[:, 0], X[:, 1], c=Z, cmap=plt.cm.coolwarm)
    plt.colorbar()
    plt.show()


def queried_points_on_map(data, title):
    query_order = data.query_order
    X = data.getX()

    Z = np.zeros(len(X))
    for i in range(len(query_order)):
        queried = query_order[i]
        Z[queried] = i+1

    plt.scatter(X[:, 0], X[:, 1], c=Z, cmap=plt.cm.coolwarm)
    plt.colorbar()
    plt.title(title)
    plt.show()


def compare_queried_points_on_map(data1, data2, title):
    query_order1 = data1.query_order
    query_order2 = data2.query_order
    X = data1.getX()

    diff_query = list(set(query_order1) - set(query_order2))

    Z = np.zeros(len(X))
    for i in range(len(diff_query)):
        queried = diff_query[i]
        Z[queried] = 2

    plt.scatter(X[:, 0], X[:, 1], c=Z)
    plt.title(str(title) + " first not in second")
    plt.show()

    diff_query = list(set(query_order2) - set(query_order1))
    Z = np.zeros(len(X))
    for i in range(len(diff_query)):
        queried = diff_query[i]
        Z[queried] = 2

    plt.scatter(X[:, 0], X[:, 1], c=Z)
    plt.title(str(title) + " second not in first")
    plt.show()  
def heatmap_comb_model_and_useer_unc(data, m_unc_nbr, comb_nbr):
    if data.two_d_flag == 0:
        return
    X = data.getX()[:, :2]
    y = data.get_known_labels() 
    clf = SSkNNO(k=10, contamination=data.contamination)
    clf.fit(X, y)

    t_idk = data.times_idk * 3
    y = y + t_idk
    c_y = []
    for k in range(len(y)):
        if y[k] == -1:
            c_y.append('red')
        elif y[k] == 0:
            c_y.append('grey')
        elif y[k] == 1:
            c_y.append('green')
        elif y[k] == 3:
            c_y.append('yellow')

    c_y = np.array(c_y)

    margin_size = 0.2
    steps = 50

    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()
    x_range = abs(x_max - x_min)
    y_range = abs(y_max - y_min)

    xmin, xmax = x_min - margin_size * x_range, x_max + margin_size * x_range
    ymin, ymax = y_min - margin_size * y_range, y_max + margin_size * y_range 
    xx, yy = np.meshgrid(
        np.linspace(xmin, xmax, int(steps)),
        np.linspace(ymin, ymax, int(steps)))
    X_mesh = np.c_[xx.ravel(), yy.ravel()]


    avg_user_uncertainty = 0
    tree = BallTree(X, leaf_size=2)
    dist, indices = tree.query(X_mesh, k=nbs) 

    Z_u = np.zeros(len(X_mesh))
    for z in range(len(Z_u)):
        Z_u[z] = data.est_user_uncertainty_of_point(avg_user_uncertainty, dist[z], indices[z])

    probs = clf.predict_proba(X_mesh)
    Z_m = Strategies.get_model_uncertainty(m_unc_nbr, probs)

    Z = Strategies.combine_user_model_unc_array(Z_m, Z_u, comb_nbr)
    Z = 1 - Z
    Z = Z.reshape(xx.shape) 
    plt.figure(figsize=(9,8)) 
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.colorbar()
    warnings.filterwarnings("ignore", message="No contour levels were found within the data range.")
    plt.contour(xx, yy, Z, np.linspace(0.0, np.max(Z), 10)) 
    scatter = plt.scatter(X[:, 0], X[:, 1], s=40, edgecolors='k', c=c_y) 
    plt.show()


def heatmaps_subplots(data, m_unc_nbr, comb_nbr, strat_name, iteration_nbr = -1):
    f, axes = plt.subplots(nrows=3, ncols=1)
    ax1 = axes.flat[0]
    ax2 = axes.flat[1]
    ax3 = axes.flat[2]
    f.set_figheight(20)
    f.set_figwidth(20)


    if data.two_d_flag == 0:
        return
    X = data.getX()[:, :2]
    y = data.get_known_labels() 
    clf = SSkNNO(k=10, contamination=data.contamination)
    margin_size = 0.2
    steps = 50

    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()
    x_range = abs(x_max - x_min)
    y_range = abs(y_max - y_min)

    xmin, xmax = x_min - margin_size * x_range, x_max + margin_size * x_range
    ymin, ymax = y_min - margin_size * y_range, y_max + margin_size * y_range 
    xx, yy = np.meshgrid(
        np.linspace(xmin, xmax, int(steps)),
        np.linspace(ymin, ymax, int(steps)))
    X_mesh = np.c_[xx.ravel(), yy.ravel()] 
    if y is None:
        clf.fit(X)
    else:
        clf.fit(X, y) 
    Z = clf.predict_proba(X_mesh)
    Z = Strategies.get_model_uncertainty(m_unc_nbr, Z)
    Z = np.array(Z)
    Z = Z.reshape(xx.shape) 
    cs = ax1.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8) 
    ax1.set_title("Model uncertainty") 
    ax1.contour(xx, yy, Z, np.linspace(0.0, np.max(Z), 10)) 
    ax1.scatter(X[:, 0], X[:, 1], cmap=plt.cm.coolwarm, s=40, edgecolors='k', c=y)
    ax1.legend(labels=['queried', 'niet gequeryd']) 
    X = data.getX()[:, :2]
    y = data.get_known_labels()
    t_idk = data.times_idk * 3
    y = y + t_idk
    c_y = []
    for k in range(len(y)):
        if y[k] == -1:
            c_y.append('red')
        elif y[k] == 0:
            c_y.append('grey')
        elif y[k] == 1:
            c_y.append('green')
        elif y[k] == 3:
            c_y.append('yellow')

    c_y = np.array(c_y)

    margin_size = 0.2
    steps = 50

    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()
    x_range = abs(x_max - x_min)
    y_range = abs(y_max - y_min)

    xmin, xmax = x_min - margin_size * x_range, x_max + margin_size * x_range
    ymin, ymax = y_min - margin_size * y_range, y_max + margin_size * y_range 
    xx, yy = np.meshgrid(
        np.linspace(xmin, xmax, int(steps)),
        np.linspace(ymin, ymax, int(steps)))
    X_mesh = np.c_[xx.ravel(), yy.ravel()]


    avg_user_uncertainty = 0
    tree = BallTree(X, leaf_size=2)
    dist, indices = tree.query(X_mesh, k=nbs) 

    Z = np.zeros(len(X_mesh))
    for z in range(len(Z)):
        Z[z] = data.est_user_uncertainty_of_point(avg_user_uncertainty, dist[z], indices[z])
    Z = Z.reshape(xx.shape) 
    cs2 = ax2.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8) 
    ax2.set_title("Estimated user certainty")
    warnings.filterwarnings("ignore", message="No contour levels were found within the data range.")
    ax2.contour(xx, yy, Z, np.linspace(0.0, np.max(Z), 10)) 
    scatter = ax2.scatter(X[:, 0], X[:, 1], s=40, edgecolors='k', c=c_y)
    ax2.legend(labels=['anomalie', '/', 'normaal', 'idk'], loc=1) 
    X = data.getX()[:, :2]
    y = data.get_known_labels() 
    clf = SSkNNO(k=10, contamination=data.contamination)
    clf.fit(X, y)

    t_idk = data.times_idk * 3
    y = y + t_idk
    c_y = []
    for k in range(len(y)):
        if y[k] == -1:
            c_y.append('red')
        elif y[k] == 0:
            c_y.append('grey')
        elif y[k] == 1:
            c_y.append('green')
        elif y[k] == 3:
            c_y.append('yellow')

    c_y = np.array(c_y)

    margin_size = 0.2
    steps = 50

    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()
    x_range = abs(x_max - x_min)
    y_range = abs(y_max - y_min)

    xmin, xmax = x_min - margin_size * x_range, x_max + margin_size * x_range
    ymin, ymax = y_min - margin_size * y_range, y_max + margin_size * y_range 
    xx, yy = np.meshgrid(
        np.linspace(xmin, xmax, int(steps)),
        np.linspace(ymin, ymax, int(steps)))
    X_mesh = np.c_[xx.ravel(), yy.ravel()]

    avg_user_uncertainty = 0
    tree = BallTree(X, leaf_size=2)
    dist, indices = tree.query(X_mesh, k=nbs) 

    Z_u = np.zeros(len(X_mesh))
    for z in range(len(Z_u)):
        Z_u[z] = data.est_user_uncertainty_of_point(avg_user_uncertainty, dist[z], indices[z])

    probs = clf.predict_proba(X_mesh)
    Z_m = Strategies.get_model_uncertainty(m_unc_nbr, probs)

    Z = Strategies.combine_user_model_unc_array(Z_m, Z_u, comb_nbr) 
    Z = Z.reshape(xx.shape) 
    levels = np.linspace(0.0, 1.0, 15)
    cs3 = ax3.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8, levels=levels) 
    warnings.filterwarnings("ignore", message="No contour levels were found within the data range.")
    ax3.contour(xx, yy, Z, np.linspace(0.0, np.max(Z), 10))
    ax3.set_title("Combined user certainty and model uncertainty") 
    scatter = ax3.scatter(X[:, 0], X[:, 1], s=40, edgecolors='k', c=c_y) 
    ax3.scatter(X[data.last_queried][0], X[data.last_queried][1], marker='*', c='lime', s=350) 

    f.subplots_adjust(right=0.8)
    cbar_ax = f.add_axes([0.95, 0.15, 0.02, 0.7])
    cb = f.colorbar(cs3, cax=cbar_ax, drawedges=True) 
    cb.outline.set_linewidth(4)

    if save:
        folder = './figuren/'
        name = folder + 'gain_strat-strategy-iteration'+str(iteration_nbr)+'.png'
        name = 'gain_strat-strategy-iteration'+str(iteration_nbr)+'.png'
        f.savefig(name)
    f.show()


def plot_stats(data):
    heatmap(data)
    heatmap_est_user_uncertainty(data)
    heatmap_comb_model_and_useer_unc(data)


def plot_model_unc_vs_auc_gain(data, title):
    queried = data.query_counter

    auc_gains = []
    for i in range(queried-1):
        curr_gain = data.auc[i+1] - data.auc[i]
        auc_gains.append(curr_gain) 
    auc_gains.append(-0.001)

    model_uncertainty = []
    for i in range(queried):
        model_uncertainty.append(data.model_uncertainty[i])

    point_exp_certainty = []
    for i in range(queried):
        point_exp_certainty.append(data.query_expected_chance)

    order = []
    for i in range(queried):
        order.append(i)

    plt.scatter(model_uncertainty, auc_gains, c=order) 
    plt.title(title)
    plt.xlabel("Model uncertainty")
    plt.ylabel("auc gain_strat") 
    plt.show()
    plt.clf() 
def intermediate_model_uncertainty_plot(data, model_uncertainty, it_nbr):
    if data.two_d_flag == 0:
        return

    X = data.getX()
    plt.scatter(X[:, 0], X[:, 1], c=model_uncertainty)
    plt.title("Estimated user uncertainty iteration: " + str(it_nbr))
    plt.show()


def query_results(queries, auc, labels, title):
    auc = auc[:]
    plt.title(title)
    scatter = plt.scatter(queries, auc, c=labels)
    classes = ['A, Lab', 'N, lab', 'A, IDK', 'N, IDK']
    plt.legend(handles=scatter.legend_elements()[0], labels=classes) 
    plt.show()
    plt.clf()


def plot_diff_true_vs_exp_user_uncertainty(queries, title, true_user, exp_user):
    diff = abs(exp_user - true_user)
    window = 10
    rolling_avg = rolling_average(diff, window)
    plt.show()"""
    plt.clf()
    plt.title(title + ":Rolling avg ")
    lbl = "W=" + str(window)
    plt.plot(queries, true_user, label="True")
    plt.plot(queries, rolling_avg, label=lbl)
    plt.legend()
    plt.show()
    plt.clf()


def seperate_3_factors_plot(data):
    true_unc = data.true_user_certainty
    prev_queried = data.query_order
    number_of_points = len(data.X) 
    plt.clf()
    unc = []
    """
    homogenity = []
    lokalisatie = []
    ongekendheid = []
    gemiddelde_onzekerheid = data.gemiddelde_ongekendheid
    ctr = []
    for i in range(len(prev_queried)):
        index = prev_queried[i]  
        homogenity.append(data.homogenity[(i*number_of_points)+index])
        lokalisatie.append(data.lokalisatie[(i * number_of_points) + index])
        ongekendheid.append(data.ongekendheid[(i * number_of_points) + index])
        unc.append(true_unc[index])
        ctr.append(i)


    plt.plot(ctr,homogenity)
    plt.plot(ctr,unc)
    plt.title("Homogenity")
    plt.show()
    

    diff_hom = [abs(x - y) for x,y in zip(unc, homogenity)]
    plt.plot(ctr, diff_hom)
    plt.title("Difference homogenity")
    plt.show()
    print("Diff hom: " + str(np.average(diff_hom)))
    
    plt.plot(ctr,lokalisatie)
    plt.title("Lokalisatie")
    plt.plot(ctr,unc)
    plt.show()
    
    
    diff_loc = [abs(x - y) for x, y in zip(unc, lokalisatie)]
    plt.plot(ctr, diff_loc)
    plt.title("Difference Lokalisatie")
    plt.show()
    print("Diff lok: " + str(np.average(diff_loc)))

    a = 0.5
    comb = [(x * a + (1-a) * y) for x, y in zip(homogenity, lokalisatie)]
    diff = [abs(x - y) for x, y in zip(unc, comb)]
    plt.plot(ctr, pd.Series(diff).rolling(window=5).mean(), label="Combined")
    plt.plot(ctr, pd.Series(diff_hom).rolling(window=5).mean(), label="Hom")
    plt.plot(ctr, pd.Series(diff_loc).rolling(window=5).mean(), label="Lok")
    plt.title("Differences: " + str(a))
    plt.legend()
    plt.show()
    print("Diff comb: " + str(np.average(diff)))

    plt.plot(ctr, diff)
    plt.title("Difference combined: " + str(a))
    plt.show()


    plt.plot(ctr,gemiddelde_onzekerheid, label="Gem")
    plt.plot(ctr,unc, label="unc")
    plt.title("Ongekendheid")
    plt.legend()
    plt.show()

    diff_ong = [abs(x - y) for x, y in zip(unc, gemiddelde_onzekerheid)]
    plt.plot(ctr, diff_ong)
    plt.title("Difference gem")
    plt.show()
    print("Diff gem: " + str(np.average(diff_ong)))

    all_comb = [(comb * (1-ongekendheid) + gemiddelde_onzekerheid * ongekendheid) for comb, ongekendheid, gemiddelde_onzekerheid in zip(comb, ongekendheid, gemiddelde_onzekerheid)]
    plt.plot(ctr, all_comb)
    plt.title("DTUK")
    plt.show()

    diff_dtuk = [abs(x-y) for x,y in zip(unc, all_comb)]
    plt.plot(ctr, diff_dtuk)
    plt.title("Difference DTUK")
    plt.show()
    print("Diff DTUK: " + str(np.average(diff_ong)))


def rolling_average(diff, window):
    rolling_avg = np.zeros(len(diff))
    for i in range(len(diff)):
        cumsum = 0
        for j in range(min(window, i+1)):
            cumsum = cumsum + diff[i - j]
        rolling_avg[i] = cumsum / min(window, i+1)
    return rolling_avg


def user_vs_model_uncertainty(data, title, note=None):
    detector = SSkNNO(contamination=data.contamination)
    detector.fit(data.getX()) 
    predict_proba = detector.predict_proba(data.getX(), method='squash')
    model_uncertainty = Strategies.get_model_uncertainty(7, predict_proba)   
    user_uncertainty = data.true_user_certainty
    queried = np.zeros(len(data.true_labels))
    for i in range(len(data.true_labels)):
        if data.queried[i]:
            queried[i] = 5

    labels = data.true_labels

    scatter = plt.scatter(model_uncertainty, user_uncertainty, c=labels)
    classes = ['Normaal', 'Anomalie', 'C']
    plt.legend(handles=scatter.legend_elements()[0], labels=classes)
    plt.xlabel("Model onzekerheid")
    plt.ylabel("Expert zekerheid")
    if note is None:
        note = "" 
    plt.title("Model onzekerheid vs expert zekerheid: " + str(title) + " " + str(note))
    plt.show()
    plt.clf()

    bins = 20
    binner_x = [(1/bins)*k for k in range(1,bins+1)]
    b_unc_x = [(1 / 10) * k for k in range(1, 11)]

    binner_y = [0 for k in range(0, bins)]

    binner_usr_unc_y = [[0]*20 for k in range(0,10)]

    for ii in range(0,len(model_uncertainty)-1):
        curr = model_uncertainty[ii]
        unc = user_uncertainty[ii]
        bin = -1
        found1 = False
        for jj in range(0,len(binner_x)-1):
            treshold = binner_x[jj]
            if curr < treshold and not found1:
                bin = jj
                binner_y[jj] = binner_y[jj] + 1
                found1 = True
                break
        for kk in range(0,10):
            tre = b_unc_x[kk]
            if unc < tre:
                binner_usr_unc_y[kk][bin] = binner_usr_unc_y[kk][bin] + 1
                break


    x_bin_str = [str("[" + str(round(c - binner_x[0], 2)) + "-" + str(round(c, 2)) + "]") for c in binner_x]
    x_unc_bin_str = [str("[" + str(round(c - b_unc_x[0], 2)) + "-" + str(round(c, 2)) + "]") for c in b_unc_x]

    fig, ax = plt.subplots() 
    bottom = np.array([]) 
    for k in binner_usr_unc_y:
        curr = np.array(k)
        if bottom.size == 0:
            ax.bar(x_bin_str, curr, width=0.85)
            bottom = curr
        else:
            ax.bar(x_bin_str, curr, width=0.85, bottom=bottom)
            bottom = bottom + curr
    ax.set_xlabel("Model onzekerheid")
    plt.ylabel("Aantal")
    plt.xticks(rotation=90)
    plt.legend(x_unc_bin_str, title="Expert zekerheden")
    if note is None:
        note = ""
    ax.set_title("Aantal per model onzekerheid (" + str(title) + " " + str(note) + ")")
    plt.show()
    plt.clf()


def main(): 
    start = time.time()

    if d_cardio:
        data = set_up_data_cardio()
        data.print_avg_uncertainty()
        title = "Cardio"  

        plot(data, p_budget, title)

    if d_wilt:
        data = set_up_data_wilt()
        data.print_avg_uncertainty()
        title = "Wilt n=" + str(len(data.getX()))
        plot(data, p_budget, title)

    if d_pageblock:
        data = set_up_data_pageblock()
        data.print_avg_uncertainty()
        title = "PageBlock n=" + str(len(data.getX()))
        plot(data, p_budget, title)

    if d_digits:
        data = set_up_data_digits()
        data.print_avg_uncertainty()
        title = "Digits n=" + str(len(data.getX()))
        plot(data, p_budget, title)

    if d_waveform:
        data = set_up_data_waveform()
        data.print_avg_uncertainty()
        title = "Waveform n=" + str(len(data.getX()))
        plot(data, p_budget, title)

    if d_shuttle:
        data = set_up_data_shuttle()
        data.print_avg_uncertainty()
        title = "Shuttle n=" + str(len(data.getX()))
        plot(data, p_budget, title)

    if d_ionosphere:
        data = set_up_data_ion()
        data.print_avg_uncertainty()
        title = "Ionosphere n=" + str(len(data.getX()))
        plot(data, p_budget, title)

    if d_2D:
        data = set_up_2D_dataset()
        data.print_avg_uncertainty()
        title = "2D blobs" 
        start = time.time()
        plot(data, p_budget, title)
        end = time.time()
        print(str(int(end - start)))

    if d_iris:
        data = set_up_data_iris()
        data.print_avg_uncertainty()
        title = "Iris n=" + str(len(data.getX()))
        plot(data, p_budget, title)

    if d_sketch:
        data = set_up_sketch_data(6)
        data.print_avg_uncertainty()
        title = "Sketch n=" + str(len(data.getX()))
        plot(data, p_budget, title)

    if d_L_d:
        data = set_up_Lorenzo_data()
        data.print_avg_uncertainty()
        title = "L n=" + str(len(data.getX()))
        plot(data, p_budget, title)

    for name in d_name:   
        data = set_up_data_dami(name)
        data.print_avg_uncertainty()
        title = name 
        plot(data, p_budget, title)

    end = time.time()
    print("Total time for all plots: " + str(round((end - start) / 60)) + " minutes")


if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()

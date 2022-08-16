import math
p_figsave = True
split = True
normelize = True
save = True
deterministic = True
samples = 800
nbs = 50 
p_gamma = 1 

p_budget = 200
p_repetitions = 1
n_splits = 10

p_lucky = False
p_us = False
p_random = False
p_allknowing = False
p_gain = True

d_2D = True 
d_ionosphere = False
d_cardio = False
d_name = []  

unc_plot = False 
user_unc_plot = False
user_model_uncertainty_plot = True 

q_analyse = False
q_analyse_gain = False
q_data_set_up = False 
three_factors_plots = False


p_baseline_unsup = False
p_baseline_isol_unsup = False
p_baseline_sup = False


auc_plot = True
final_auc_plot = True
rmse_plot = False 
final_rmse_plot = False
mined_uncertainty_plot = False
wasted_query_plot = False 
queried_points_on_map = False     
gain_comb = [1]      
gain_m = [7]
p_gainreg = False


d_waveform = False
d_pageblock = False
d_iris = False
d_L_d = False
d_digits = False
d_wilt = False 
d_shuttle = False
d_sketch = False 
r_rs = [42, 149, 145, 83, 51, 114, 187, 38, 75, 3, 151, 92, 142, 120, 184, 99, 188, 8, 165, 173, 103, 150, 72, 17, 66,
        106, 76, 77, 93, 74, 119, 65, 159, 29, 147, 135, 26, 24, 30, 139, 164, 126, 143, 148, 134, 6, 133, 64, 35, 158,
        85, 121, 50, 175, 43, 102, 81, 2, 48, 82, 31, 79, 160, 152, 128, 55, 9, 71, 95, 166, 168, 193, 169, 180, 10, 86,
        131, 172, 162, 107, 1, 115, 144, 171, 80, 118, 140, 176, 129, 62, 198, 68, 190, 63, 78, 94, 54, 97, 157, 45, 52]


prior_weight = 0.3 
w_usr_imp = 0.5
w_m_imp = 1 - w_usr_imp

w_diff_labels = 0.5 
m_ssdo = False
m_gmm = False
probabilistic_sampling = False

q_prior = False
q_adv_prior = False
q_best_prior = True
k_ssknno = 0 
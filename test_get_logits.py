from utils.functions import get_opfeatures, logistic_regression_nodewise, logistic_regression_nodewise_online

data_file = 'SD-STGCN/dataset/highSchool/data/SIR/SIR_nsrc1-15_Rzero1-15_gamma0.1-0.4_ls21200_nf16_entire.pickle'
model_file = 'SD-STGCN/output/models/highSchool/SIR_nsrc1-15_Rzero1-15_gamma0.1-0.4_ls21200_nf16'
save_file = 'SD-STGCN/output/test_res/highSchool/SIR_nsrc1-15_Rzero1-15_gamma0.1-0.4_ls21200_nf16'
train_pct = 0.9434
n_node = 774

# features, labels = get_opfeatures(data_file, model_file, train_pct, n_node, n_frame=16)
# regression_matrix = logistic_regression_nodewise(features, labels)

coefficients, intercepts = logistic_regression_nodewise_online(data_file, model_file, save_file, train_pct, n_node, batch_size=100, n_frame=16, val_pct=0, end=-1)

print('ok')
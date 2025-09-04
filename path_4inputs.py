# version_name
ver = "vdiff_l2_reg_4inputs"

#data file directorty
read_dir = "C:/Users/Admin/Desktop/DeepMoney/"

#data file name
after_3months_predict = "kospi200f-943-3months-norm.csv"

#model directory
model_dir = "C:/Users/Admin/Desktop/DeepMoney/model_dir/" + ver

#result file directory
result_dir = "C:/Users/Admin/Desktop/DeepMoney/results_" + ver

#optimizing information file directory - hyperparameters
hyperpara_info_dir = "C:/Users/Admin/Desktop/DeepMoney/hyperparameters/"

class Config(object):

    init_scale = 0.05
    learning_rate = 0.001
    num_layers = 2
    num_steps = 50
    hidden_size = 30
    batch_size = 20
    input_size = 7
    output_size = 1
    rnn_mode = "basic"
    iter_steps = 5000
    step_interval = 1
    test_start = "2017-07-10"
    predict_term = 20
    model_reset = True
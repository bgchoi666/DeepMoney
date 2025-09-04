# version_name
ver = "vdiff_65days_totnorm0304"

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
    num_steps = 20
    hidden_size = 500
    batch_size = 20
    input_size = 943
    output_size = 1
    rnn_mode = "basic"
    iter_steps = 1000
    step_interval = 10
    train_start =  "2000-01-01"
    grad_train_terms = ["2000-01-01", "2018-01-01", "2018-02-01", "2018-03-01", "2018-04-01", "018-05-01", "2018-06-01", "2018-07-01", "2018-08-01"]
    test_start = "2018-01-01"
    test_end = "2019-03-01"
    predict_term = 65
    model_reset = True
    shuffle = True
    read_mode = "norm"
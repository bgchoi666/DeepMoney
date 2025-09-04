# version_name
ver = "20norm_1days_close_open_delay"

#data file directorty
read_dir = "../fnguide data/"

#data file name
#after_3months_predict = "kospi200f-943-3months-norm.csv"

#model directory
model_dir = "model_dir/" + ver

#result file directory
result_dir = "results_" + ver

#optimizing information file directory - hyperparameters
hyperpara_info_dir = "hyperparameters/"

class Config(object):

    init_scale = 0.05
    learning_rate = 0.001
    num_layers = 2
    num_steps = 20
    hidden_size = 300
    batch_size = 20
    input_size = 943
    output_size = 1
    rnn_mode = "basic"
    iter_steps = 1000
    step_interval = 1
    test_start = "2018-01-01"
    test_end = "2018-12-31"
    train_start = "2000-01-28"
    predict_term = 1
    read_mode = "norm"
    conversion = "diff"
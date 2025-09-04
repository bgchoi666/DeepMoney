# version_name
ver = "vma_only"

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
    num_steps = 10
    hidden_size = 300
    batch_size = 200
    input_size = 943
    output_size = 1
    rnn_mode = "basic"
    iter_steps = 1000
    step_interval = 10
    test_start = "2018-03-02"
    predict_term = 65

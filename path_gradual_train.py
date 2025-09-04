# version_name
ver = "gradual_train"

#data file directorty
read_dir = "C:/Users/Admin/Desktop/DeepMoney/"

#data file name
file_name = "s&p500-519-rate.csv"
market = "s&p500-519"

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
    input_size = 519
    output_size = 1
    rnn_mode = "basic"
    iter_steps = 1000
    step_interval = 20
    train_start =  "2000-01-01"
    grad_train_terms = ["2000-01-01", "2018-01-01", "2018-02-01", "2018-03-01", "2018-04-01", "2018-05-01", "2018-06-01", "2018-07-01"]
    test_start = "2017-04-03"
    test_end = "2017-05-01"
    predict_term = 65
    model_reset = True
    shuffle = True
    read_mode = "norm"
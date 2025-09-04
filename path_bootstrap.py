# version_name
ver = "bootstrap_5days"

#data file directorty
read_dir = "C:/Users/Admin/Desktop/DeepMoney/"

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
    hidden_size =200
    batch_size = 20
    input_size = 943
    output_size = 1
    rnn_mode = "basic"
    iter_steps = 1000
    step_interval = 1
    train_start =  "2000-01-01"
    test_start = "2017-04-03"
    test_end = "2018-08-01"
    predict_term = 1
    model_reset = True
    shuffle = True

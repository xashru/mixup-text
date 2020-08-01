class TaskConfig:
    num_class = None
    train_file = None
    val_file = None
    test_file = None
    sequence_len = None
    eval_interval = None
    patience = None


def get_task_config(task_name):
    config = TaskConfig()
    if task_name == 'trec':
        config.num_class = 6
        config.train_file = 'data/trec/train.csv'
        config.val_file = None
        config.test_file = 'data/trec/test.csv'
        config.sequence_len = 30
        config.eval_interval = 20
        config.patience = 30
    elif task_name == 'sst1':
        config.num_class = 5
        config.train_file = 'data/sst1/train_sent.csv'
        config.val_file = 'data/sst1/val.csv'
        config.test_file = 'data/trec/test.csv'
        config.sequence_len = 50
        config.eval_interval = 30
        config.patience = 50
    elif task_name == 'imdb':
        config.num_class = 2
        config.train_file = 'data/imdb/train.csv'
        config.val_file = None
        config.test_file = 'data/imdb/test.csv'
        config.sequence_len = 400
        config.eval_interval = 50
        config.patience = 50
    elif task_name == 'agnews':
        config.num_class = 4
        config.train_file = 'data/agnews/train.csv'
        config.val_file = None
        config.test_file = 'data/agnews/test.csv'
        config.sequence_len = 80
        config.eval_interval = 50
        config.patience = 50
    else:
        raise ValueError('Task not supported')
    return config

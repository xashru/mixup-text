class TaskConfig:
    num_class = None
    train_file = None
    val_file = None
    test_file = None
    sequence_len = None
    eval_interval = None
    patience = None


def get_task_files(task_name):
    config = TaskConfig()
    if task_name == 'trec':
        config.num_class = 5
        config.train_file = 'data/trec/train.csv'
        config.val_file = None
        config.test_file = 'data/trec/test.csv'

    return config


def main():
    config = get_task_files('trec')
    print(config.train_file)

main()

import yaml

Root = 'data/melanoma_classification'


def create_config():
    data = dict(
        epochs = 10,
        batch_size = 16,
        base_lr = 0.001,
        wt_decay = 0.0015,
        beta1 = 0.6,
        input_size = 256,
        num_channels = 3,
        csv_file = 'Melanoma_skin_cancer_detection/data/melanoma-classification/train_fold.csv',
        train_img = 'Melanoma_skin_cancer_detection/data/melanoma-classification/jpeg/train',
        fold = 0
    )
    with open('config.yml', 'w') as file:
        yaml.dump(data, file, default_flow_style=False)

if __name__ == '__main__':
    create_config()
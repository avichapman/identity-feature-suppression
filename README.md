# identity-feature-suppression
This is a Python3 / [Pytorch](https://pytorch.org/) implementation of [ASIF](https://arxiv.org/abs/2209.14553), as described in the paper **"Regularizing Neural Network Training via Identity-wise Discriminative Feature Suppression"**, by Avraham Chapman and Lingqiao Liu.

## Abstract

It is well-known that a deep neural network has a strong fitting capability and can easily achieve a low training error even with randomly assigned class labels. When the number of training samples is small, or the class labels are noisy, networks tend to memorize patterns specific to individual instances to minimize the training error. This leads to the issue of overfitting and poor generalisation performance. This paper explores a remedy by suppressing the network's tendency to rely on instance-specific patterns for empirical error minimisation. The proposed method is based on an adversarial training framework. It suppresses features that can be utilized to identify individual instances among samples within each class. This leads to classifiers only using features that are both discriminative across classes and common within each class. We call our method Adversarial Suppression of Identity Features (ASIF), and demonstrate the usefulness of this technique in boosting generalisation accuracy when faced with small datasets or noisy labels.

## Setup

To run this code you need to install all packages in the 'requirements.txt' file.
```
pip install -r requirements.txt
```

## Training the model

Use the `train.py` script to train the model. To train the default model on 
CIFAR-10 simply use:

```
python3 src/train.py --label_dir ./label_noise --configs_path ./remote/<..CONFIG_FILE..>.csv --config_index 1 --config_count 1 --trial_count 3
```

Parameters:
- label_dir: Path to the location of the label noise files.
- configs_path: Path to a file that contains one or more experiment configurations.
- config_index: The one-based index of the configuration to run.
- config_count: The number of experimental configurations to run in sequence.
- trial_count: If more than one, multiple copies of an single experimental configuration will be run simultaneously. Useful if you have spare computer capacity.

## Citation

If you find this code useful please cite us in your work:

```
@inproceedings{Chapman2022ASIF,
  title={Regularizing Neural Network Training via Identity-wise Discriminative Feature Suppression},
  author={Avraham Chapman and Lingqiao Liu},
  booktitle={DICTA},
  year={2022}
}
```
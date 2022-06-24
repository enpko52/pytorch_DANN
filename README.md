# DANN

The PyTorch implementation of DANN (Domain-Adversarial Training of Neural Networks).

 - [Unsupervised Domain Adaptation by Backpropagation](https://arxiv.org/abs/1409.7495)
 - [Domain-Adversarial Training of Neural Networks](https://arxiv.org/abs/1505.07818)

![dann](https://user-images.githubusercontent.com/97284065/175561529-d2e836b6-deba-42bb-8b5f-ab8f3491c248.png)


## Environment

```
python 3.7
pytorch 1.11.0
torhvision 0.12.0
```


## Usage

If the models need to train, follow the below instruction. 

> Train the models on source-only
> ```
> python main.py --source 'mnist' --target 'mnistm' --mode 'source-only' --train
> ```
> Train the models on DANN
> ```
> python main.py --source 'mnist' --target 'mnistm' --mode 'dann' --train
> ```


If the models only test, follow the below instruction.

> Test the models on source-only
> ```
> python main.py --source 'mnist' --target 'mnistm' --mode 'source-only' --extractor 'weights_filename' --classifier 'weights_filename'
> ```
> Test the models on DANN
> ```
> python main.py --source 'mnist' --target 'mnistm' --mode 'dann' --extractor 'weights_filename' --classifier 'weights_filename'
> ```


## Experiments

`MNIST → MNIST-M`
|             |  Paper | This repo |
| :---------: | :----: | :-------: |
| Source-Only | 0.5225 |   0.6195  |
|    DANN     | 0.7666 |   0.8050  |

The result of experiments is the average of 5 experiments below.


### Details

`Source-Only`
|          | Test 1 | Test 2 | Test 3 | Test 4 | Test 5 |
| -------- |--------| ------ | ------ | ------ | ------ |
| Accuracy | 0.6160 | 0.6251 | 0.6162 | 0.6193 | 0.6208 |

`DANN`
|          | Test 1 | Test 2 | Test 3 | Test 4 | Test 5 |
| -------- |--------| ------ | ------ | ------ | ------ |
| Accuracy | 0.8205 | 0.7816 | 0.8035 | 0.8281 | 0.7911 |


## Visualizations
![visualizations](https://user-images.githubusercontent.com/97284065/175574285-ef19218e-6922-434f-bd06-4913390af4f7.png)

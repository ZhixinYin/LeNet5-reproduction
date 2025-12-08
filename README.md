
# LeNet-5 Paper Reproduction (1998)

## Introduction
This project reproduces the original LeNet-5 architecture as described in LeCun’s 1998 paper.

## Architecture Summary
| Layer  | Type                       | Maps | Kernel | Stride | Output Size  |
| ------ | -------------------------- | ---- | ------ | ------ | ------------ |
| Input  | —                          | 1    | —      | —      | 32×32        |
| C1     | Conv                       | 6    | 5×5    | 1      | 28×28        |
| S2     | AvgPool (Subsampling)      | 6    | 2×2    | 2      | 14×14        |
| C3     | Conv (Partial Connections) | 16   | 5×5    | 1      | 10×10        |
| S4     | AvgPool                    | 16   | 2×2    | 2      | 5×5          |
| C5     | Conv → Dense               | 120  | 5×5    | 1      | 1×1×120      |
| F6     | Dense                      | 84   | —      | —      | 84           |
| Output | Dense                      | 10   | —      | —      | class logits |

## Dataset
The dataset comes from MNIST, which is 28×28 and they are pad to 32×32.

## Training Setup
Learning rate: 0.001
Optimizer: Adam
Batch size: 128
Loss: cross entropy
Epochs: 10

## Reproduced Results
| Metric            | Value |
| ----------------- | ----- |
| Training accuracy | 99.37% |
| Test accuracy     | 98.52% |

![](images/TrainingAccuracy.png)


## Difference from original paper
1. Output layer is dense, while the original layer uses RBF units
2. Adam optimisation is used, while the original one is simply SGD
3. Learning rate decay is not used and fixed in this reproduction
4. There is no data augmentation

## References
LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998).
Gradient-Based Learning Applied to Document Recognition.
Proceedings of the IEEE.

myleott. (2014). mnist_png: MNIST in PNG format [Data set].
GitHub. https://github.com/myleott/mnist_png

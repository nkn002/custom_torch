# Experiments with PyTorch GPU Operations

## Testing

### SUM (sum_two_tensors)

Ten random tests are conducted to verify the correctness of the SUM operation on GPU. For each test:

- Random dimensions for tensors `a` and `b` are generated.
- Random values are assigned to `a` and `b`.
- The SUM operation is performed using `sum_two_tensors`.
- The result is compared with the PyTorch `+` operator.
- The correctness of the operation is verified.

### Matrix Multiplication (MM)

Ten random tests are conducted to verify the correctness of the MM operation on GPU. For each test:

- Random dimensions for matrices `a` and `b` are generated.
- Random values are assigned to `a` and `b`.
- Matrix multiplication is performed using the `mm` function.
- The result is compared with the PyTorch `@` operator.
- The correctness of the operation is verified.

### Matrix Multiplication with Transpose (MM with Transpose)

Ten random tests are conducted to verify the correctness of the MM with transpose operation on GPU. For each test:

- Random dimensions for matrices `input` and `weight` are generated.
- Random values are assigned to `input` and `weight`.
- Matrix multiplication with transpose is performed using both the PyTorch `mm` function and a custom `transpose` function.
- The results of both operations are compared.
- The correctness of the operation is verified.

## Running the Code

To run the experiments, ensure that you have a CUDA-enabled GPU and the required PyTorch environment set up. Execute the provided code snippet, and the experiments will be conducted automatically.

```
cd deep-codegen
python test.py
```

# LeNet-300-100 Training with PyTorch

This README provides an overview of the LeNet-300-100 training script implemented in PyTorch. The script trains a LeNet-300-100 neural network on the MNIST dataset, and it includes custom GPU operations for linear layers.

## Code Overview

The code includes the following key components:

1. **Custom Linear Layer:** This script defines a custom linear layer (`CustomLinear`) that extends PyTorch's functionality. The custom linear layer implements GPU-accelerated matrix multiplication and supports GPU operations.

2. **LeNet-300-100 Model:** The script defines a LeNet-300-100 neural network model that can use either the standard PyTorch linear layers or custom linear layers based on the user's choice.

3. **Training and Evaluation:** The script loads the MNIST dataset, performs training, and evaluates the model's performance. It includes options to specify the number of epochs, learning rate decay, and other training parameters.

## Requirements

Before running the code, ensure you have the following requirements:

- Python 3.x
- PyTorch
- CUDA-compatible GPU (if using GPU acceleration)
- Required Python libraries (numpy, tqdm, scikit-learn, torchvision)

## Running the Code

To run the LeNet-300-100 training script, execute the following command:

```bash
cd deep-codegen
python train_lenet.py [--num-epoches NUM_EPOCHES] [--decay DECAY] [--threshold THRESHOLD] [--learning-rate LEARNING_RATE] [--custom CUSTOM]
```

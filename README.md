# Neural Network Matrix Library For ESP32

A lightweight and efficient neural network library implemented in C++ with a focus on memory efficiency and automatic differentiation. The library provides core matrix operations and neural network primitives suitable for resource-constrained environments.

## Features

- Efficient matrix operations with automatic differentiation
- Smart memory management using intrusive pointers
- Customizable tensor operations
- Gradient computation and backpropagation
- Support for various activation functions
- Memory-efficient implementation suitable for embedded systems

## Core Components

### Matrix Operations
- Matrix multiplication
- Element-wise operations (addition, subtraction)
- Dot product
- Custom operation support

### Neural Network Features
- Automatic differentiation
- Gradient clipping
- Customizable loss functions
- Flexible layer architecture
- Activation functions (LeakyReLU implemented, extensible)

### Memory Management
- Smart pointer implementation for automatic memory handling
- Reference counting through intrusive pointers
- Memory allocation strategies for different environments

## Usage

### Basic Matrix Operations
```cpp
Value a(2, 2, data_a, "matrix_a");
Value b(2, 2, data_b, "matrix_b");

// Matrix multiplication
Value c = a * b;

// Element-wise operations
Value d = a + b;
Value e = a - b;

// Activation function
Value f = a.leakyrelu();
```

### Training Loop Example
```cpp
// Forward pass
Value hidden = input * weights1;
Value activated = hidden.leakyrelu();
Value output = activated * weights2;

// Compute loss and backward pass
float loss = compute_loss(output, target);
output.backward();

// Update weights
weights1.update(learning_rate);
weights2.update(learning_rate);
```

## Project Structure

- `include/matrix.h`: Core matrix operations and tensor implementations
- `include/value.h`: Autograd value wrapper for tensors
- `include/minimal_intrusive_ptr.hpp`: Memory management utilities

## Building

The library is header-only and can be included directly in your project. It requires a C++11 compatible compiler.

## Integration

1. Copy the include files to your project
2. Include the necessary headers
3. Configure memory management according to your environment
4. Start using the matrix and neural network operations

## Contributing

Contributions are welcome! Areas for improvement:
- Additional activation functions
- More optimization algorithms
- Extended matrix operations
- Performance optimizations
- Additional loss functions

## License

[Your chosen license]
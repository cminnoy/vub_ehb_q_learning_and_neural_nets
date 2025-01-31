#include <iostream>
#include <vector>
#include <random>
#include <algorithm>

// Example activation functions
double relu(double x) {
    return std::max(0.0, x);
}

double relu_derivative(double x) {
    return x > 0 ? 1.0 : 0.0;
}

class NeuralNetwork {
public:
    // layer_sizes: e.g., {2, 4, 3, 1} means 2 inputs, 2 hidden layers (4 & 3 neurons), 1 output.
    NeuralNetwork(const std::vector<int>& layer_sizes)
        : layer_sizes(layer_sizes), num_layers(layer_sizes.size())
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(-0.5, 0.5);

        // For each layer transition, initialize weight matrix and bias vector.
        for (size_t layer = 0; layer < num_layers - 1; layer++) {
            int in_size = layer_sizes[layer];
            int out_size = layer_sizes[layer + 1];

            // Create weight matrix: dimensions in_size x out_size
            std::vector<std::vector<double>> weight_matrix(in_size, std::vector<double>(out_size));
            for (auto &row : weight_matrix)
                for (auto &val : row)
                    val = dis(gen);
            weights.push_back(weight_matrix);

            // Create bias vector: length out_size (initialized to 0)
            biases.push_back(std::vector<double>(out_size, 0.0));
        }
    }

    // Forward pass.
    // The function takes an input vector and returns the network's output.
    std::vector<double> forward(const std::vector<double>& input,
                                std::vector<std::vector<double>>& layer_activations,
                                std::vector<std::vector<double>>& layer_zs) 
    {
        // Clear any previous activations.
        layer_activations.clear();
        layer_zs.clear();

        // The activation of the input layer is the input itself.
        std::vector<double> activation = input;
        layer_activations.push_back(activation); // Store input layer activations

        // For each layer transition
        for (size_t layer = 0; layer < num_layers - 1; layer++) {
            int out_size = layer_sizes[layer + 1];
            std::vector<double> z(out_size, 0.0); // weighted input + bias

            // Compute z = activation * weights[layer] + biases[layer]
            for (int j = 0; j < out_size; j++) {
                for (size_t i = 0; i < activation.size(); i++) {
                    z[j] += activation[i] * weights[layer][i][j];
                }
                z[j] += biases[layer][j];
            }
            layer_zs.push_back(z);

            // If this is not the output layer, apply ReLU. For the output layer you might
            // want to apply a different function (or none at all).
            if (layer < num_layers - 2) {
                for (auto &val : z)
                    val = relu(val);
            }
            activation = z;
            layer_activations.push_back(activation);
        }
        return activation;
    }

    // Simplified backward pass for Mean Squared Error (MSE) loss.
    // Note: For a complete implementation, you might want to store activations from the forward pass.
    void backward(const std::vector<double>& input,
                  const std::vector<double>& target,
                  double alpha) 
    {
        // Store activations and z-values from each layer.
        std::vector<std::vector<double>> activations;
        std::vector<std::vector<double>> zs;
        std::vector<double> output = forward(input, activations, zs);

        // Compute output error: delta = (output - target) 
        // (for MSE and linear output activation, this is the gradient at the output)
        std::vector<std::vector<double>> deltas(num_layers - 1);
        int output_size = layer_sizes.back();
        deltas[num_layers - 2] = std::vector<double>(output_size, 0.0);
        for (int j = 0; j < output_size; j++) {
            // For a linear activation at the output layer:
            deltas[num_layers - 2][j] = output[j] - target[j];
        }

        // Backpropagate the error to previous layers
        // Note: layer indices for weights/biases: 0 to num_layers-2.
        for (int layer = num_layers - 2; layer > 0; layer--) {
            int layer_size = layer_sizes[layer];      // current layer size
            int next_layer_size = layer_sizes[layer + 1]; // size of layer after current layer

            deltas[layer - 1] = std::vector<double>(layer_size, 0.0);
            for (int i = 0; i < layer_size; i++) {
                double error = 0.0;
                // Sum over neurons in the next layer.
                for (int j = 0; j < next_layer_size; j++) {
                    error += deltas[layer][j] * weights[layer][i][j];
                }
                // Multiply by derivative of the activation function.
                // Here we use ReLU derivative for hidden layers.
                deltas[layer - 1][i] = error * relu_derivative(zs[layer - 1][i]);
            }
        }

        // Update weights and biases using the deltas.
        for (size_t layer = 0; layer < num_layers - 1; layer++) {
            int in_size = layer_sizes[layer];
            int out_size = layer_sizes[layer + 1];

            // Update weights: weights[layer][i][j] -= alpha * activation[i] * delta[j]
            for (int i = 0; i < in_size; i++) {
                for (int j = 0; j < out_size; j++) {
                    weights[layer][i][j] -= alpha * activations[layer][i] * deltas[layer][j];
                }
            }
            // Update biases: biases[layer][j] -= alpha * delta[j]
            for (int j = 0; j < out_size; j++) {
                biases[layer][j] -= alpha * deltas[layer][j];
            }
        }
    }

    // Utility: print the network's parameters.
    void print() const {
        std::cout << "Neural Network Contents:\n";
        for (size_t layer = 0; layer < num_layers - 1; layer++) {
            std::cout << "Layer " << layer << " -> " << layer + 1 << " weights:\n";
            for (const auto &row : weights[layer]) {
                for (const auto &value : row) {
                    std::cout << value << " ";
                }
                std::cout << "\n";
            }
            std::cout << "Biases:\n";
            for (const auto &bias : biases[layer]) {
                std::cout << bias << " ";
            }
            std::cout << "\n";
        }
    }

private:
    std::vector<int> layer_sizes;  // e.g., {input_size, hidden1_size, ..., output_size}
    int num_layers;                // total number of layers

    // Each weight matrix is a 2D vector: for layer L,
    // dimensions are [layer_sizes[L]] x [layer_sizes[L+1]]
    std::vector<std::vector<std::vector<double>>> weights;
    // Each bias vector corresponds to layer L+1.
    std::vector<std::vector<double>> biases;
};

int main() {
    // Define a network with 2 inputs, 2 hidden layers (4 and 3 neurons) and 1 output.
    std::vector<int> layers = {2, 4, 3, 1};
    NeuralNetwork nn(layers);

    // Example input and target
    std::vector<double> input = {0.5, -0.2};
    std::vector<double> target = {0.8};

    // Run a forward pass
    std::vector<std::vector<double>> activations, zs;
    std::vector<double> output = nn.forward(input, activations, zs);

    std::cout << "Initial output:\n";
    for (auto v : output)
        std::cout << v << " ";
    std::cout << "\n";

    // Perform a backward pass with a learning rate alpha
    double alpha = 0.01;
    nn.backward(input, target, alpha);

    // Print updated network parameters
    nn.print();

    return 0;
}

#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <thread>
#include <chrono>

// Double Q-Learning based on 2015 paper of van Hasselt et al. (https://arxiv.org/abs/1509.06461)

// Define actions
enum class Action { Up, Down, Left, Right };

// Define state as a pair of integers (x, y)
using State = std::pair<int, int>;

// Neural network structure
struct NeuralNetwork {
    std::vector<std::vector<double>> weights1; // Input to hidden
    std::vector<std::vector<double>> weights2; // Hidden to output
    std::vector<double> biases1;               // Hidden layer biases
    std::vector<double> biases2;               // Output layer biases
    int input_size, hidden_size, output_size;

    NeuralNetwork(int input_size, int hidden_size, int output_size)
        : input_size(input_size), hidden_size(hidden_size), output_size(output_size) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(-0.5, 0.5);

        // Initialize weights and biases
        weights1 = std::vector<std::vector<double>>(input_size, std::vector<double>(hidden_size));
        weights2 = std::vector<std::vector<double>>(hidden_size, std::vector<double>(output_size));
        biases1 = std::vector<double>(hidden_size, 0.0);
        biases2 = std::vector<double>(output_size, 0.0);

        for (auto& row : weights1)
            for (auto& val : row) val = dis(gen);

        for (auto& row : weights2)
            for (auto& val : row) val = dis(gen);
    }

    // Forward pass
    std::vector<double> forward(const State& state) {
        std::vector<double> input = {static_cast<double>(state.first), static_cast<double>(state.second)};
        std::vector<double> hidden(hidden_size, 0.0);
        std::vector<double> output(output_size, 0.0);

        // Input to hidden
        for (int i = 0; i < hidden_size; ++i) {
            for (int j = 0; j < input_size; ++j) {
                hidden[i] += input[j] * weights1[j][i];
            }
            hidden[i] += biases1[i];
            hidden[i] = std::max(0.0, hidden[i]); // ReLU
        }

        // Hidden to output
        for (int i = 0; i < output_size; ++i) {
            for (int j = 0; j < hidden_size; ++j) {
                output[i] += hidden[j] * weights2[j][i];
            }
            output[i] += biases2[i];
        }
        return output; // Linear output
    }

    // Backpropagation
    void backward(const State& state, const std::vector<double>& target, double alpha) {
        std::vector<double> input = {static_cast<double>(state.first), static_cast<double>(state.second)};
        std::vector<double> hidden(hidden_size, 0.0);
        std::vector<double> output = forward(state);

        // Calculate gradients
        std::vector<double> output_error(output_size, 0.0);
        for (int i = 0; i < output_size; ++i) {
            output_error[i] = output[i] - target[i];
        }

        std::vector<double> hidden_error(hidden_size, 0.0);
        for (int i = 0; i < hidden_size; ++i) {
            for (int j = 0; j < output_size; ++j) {
                hidden_error[i] += output_error[j] * weights2[i][j];
            }
            hidden_error[i] = hidden_error[i] > 0 ? hidden_error[i] : 0; // ReLU derivative
        }

        // Update weights2 and biases2
        for (int i = 0; i < hidden_size; ++i) {
            for (int j = 0; j < output_size; ++j) {
                weights2[i][j] -= alpha * hidden[i] * output_error[j];
            }
        }
        for (int j = 0; j < output_size; ++j) {  // Corrected bias loop
            biases2[j] -= alpha * output_error[j];
        }

        // Update weights1 and biases1
        for (int i = 0; i < input_size; ++i) {
            for (int j = 0; j < hidden_size; ++j) {
                weights1[i][j] -= alpha * input[i] * hidden_error[j];
            }
        }
        for (int j = 0; j < hidden_size; ++j) {  // Corrected bias loop
            biases1[j] -= alpha * hidden_error[j];
        }
    }

    void print() const {
        std::cout << "Neural Network Contents:\n";

        // Print weights1
        std::cout << "Weights1:\n";
        for (const auto& row : weights1) {
            for (const auto& value : row) {
                std::cout << value << " ";
            }
            std::cout << "\n";
        }

        // Print biases1
        std::cout << "Biases1:\n";
        for (const auto& value : biases1) {
            std::cout << value << " ";
        }
        std::cout << "\n";

        // Print weights2
        std::cout << "Weights2:\n";
        for (const auto& row : weights2) {
            for (const auto& value : row) {
                std::cout << value << " ";
            }
            std::cout << "\n";
        }

        // Print biases2
        std::cout << "Biases2:\n";
        for (const auto& value : biases2) {
            std::cout << value << " ";
        }
        std::cout << "\n";
    }
};

// Rewards
const double REWARD_SEED_1 = 1.0;
const double REWARD_SEED_6 = 10.0;
const double REWARD_CAT = -10.0;
const double REWARD_EMPTY = -1.0;

// Hyperparameters
double epsilon = 1.0;
const double epsilon_min = 0.01;
const double epsilon_decay = 0.995;
const double learning_rate = 0.1;
const double Gamma = 0.9;
const double tau = 0.005; // Soft update factor
const int GRID_SIZE = 3;

// Function to get reward
double getReward(const State& state) {
    static bool seed_available = true;
    if (state == std::make_pair(0, 0)) {
        if (seed_available) {
            seed_available = false;
            return REWARD_SEED_1;
        } else {
            return REWARD_EMPTY;
        }
    }
    if (state == std::make_pair(2, 2)) return REWARD_SEED_6;
    if (state == std::make_pair(1, 1)) return REWARD_CAT;
    return REWARD_EMPTY;
}

// Check if a state is terminal
bool isTerminal(const State& state) {
    return state == std::make_pair(2, 2) || state == std::make_pair(1, 1);
}

// Get all possible actions
std::vector<Action> getActions() {
    return {Action::Up, Action::Down, Action::Left, Action::Right};
}

std::string actionToString(Action action) {
    switch (action) {
        case Action::Up: return "Up";
        case Action::Down: return "Down";
        case Action::Left: return "Left";
        case Action::Right: return "Right";
    }
    return "Unknown";
}

// Take an action and return the next state
State takeAction(const State& state, Action action) {
    int x = state.first;
    int y = state.second;

    switch (action) {
        case Action::Up:
            if (x > 0) x--;
            break;
        case Action::Down:
            if (x < GRID_SIZE - 1) x++;
            break;
        case Action::Left:
            if (y > 0) y--;
            break;
        case Action::Right:
            if (y < GRID_SIZE - 1) y++;
            break;
    }
    return {x, y};
}

// Action selection using epsilon-greedy
Action chooseAction(const State& state, NeuralNetwork& net) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<> dis(0.0, 1.0);

    if (dis(gen) < epsilon) {
        // Explore
        std::uniform_int_distribution<> action_dist(0, 3);
        return static_cast<Action>(action_dist(gen));
    } else {
        // Exploit
        std::vector<double> q_values = net.forward(state);
        return static_cast<Action>(std::distance(q_values.begin(), std::max_element(q_values.begin(), q_values.end())));
    }
}

// Print the grid
void printGrid(const State& pigeon) {
    for (int i = 0; i < GRID_SIZE; ++i) {
        for (int j = 0; j < GRID_SIZE; ++j) {
            if (pigeon == std::make_pair(i, j)) {
                std::cout << "P  ";
            } else if (std::make_pair(i, j) == std::make_pair(0, 0)) {
                std::cout << "S1 ";
            } else if (std::make_pair(i, j) == std::make_pair(2, 2)) {
                std::cout << "S6 ";
            } else if (std::make_pair(i, j) == std::make_pair(1, 1)) {
                std::cout << "C  ";
            } else {
                std::cout << ".  ";
            }
        }
        std::cout << "\n";
    }
}

int main() {
    NeuralNetwork net(2, 10, 4);
    NeuralNetwork target_net = net; // Target network
    int episodes = 1000;

    double max_reward = -std::numeric_limits<double>::infinity();
    unsigned int max_reward_encountered = 0;
    const int early_stopping_threshold = 10; // Number of episodes to trigger early stopping

    for (int episode = 0; episode < episodes; ++episode) {
        State pigeon = {2, 0}; // Starting position
        double total_reward = 0.0;

        while (!isTerminal(pigeon)) {
            printGrid(pigeon);

            // Choose action
            Action action = chooseAction(pigeon, net);
            std::cout << "Action taken: " << actionToString(action) << std::endl;

            // Take action
            State next_state = takeAction(pigeon, action);
            double reward = getReward(next_state);
            total_reward += reward;

            // Update Q-values using DDQN
            std::vector<double> q_values = net.forward(pigeon);
            std::vector<double> target_q_values = q_values;

            if (isTerminal(next_state)) {
                target_q_values[static_cast<int>(action)] = reward;
            } else {
                std::vector<double> next_q_values = net.forward(next_state);
                std::vector<double> target_q_values_next = target_net.forward(next_state);
                int best_action = std::distance(next_q_values.begin(), std::max_element(next_q_values.begin(), next_q_values.end()));
                target_q_values[static_cast<int>(action)] = reward + Gamma * target_q_values_next[best_action];
            }

            // Train the network
            net.backward(pigeon, target_q_values, learning_rate);

            pigeon = next_state; // Move to next state

            std::cout << "Total reward: " << total_reward << '\n';
        }

        // Early stopping logic
        if (total_reward < max_reward) {
            max_reward_encountered = 0; // Reset counter if the reward decreases
        } else if (total_reward > max_reward) {
            max_reward = total_reward; // Update max reward
            max_reward_encountered = 0; // Reset counter
        } else {
            max_reward_encountered++; // Increment counter if reward remains the same
        }
        if (max_reward_encountered > early_stopping_threshold) {
            std::cout << "Early stopping triggered at episode: " << episode << "\n";
            target_net = net;
            break;
        }

        // Periodically update the target network
        if (episode % 10 == 0) {
            target_net = net;
        }

        // Decay epsilon
        epsilon = std::max(epsilon_min, epsilon * epsilon_decay);

        std::cout << "Episode " << episode << " Total Reward: " << total_reward << "\n";
    }

    std::cout << "Training complete.\n";

    // After training loop
    std::cout << "Final Neural Network State:\n";
    target_net.print();

    // Test the final policy
    std::cout << "Testing the final policy:\n";
    State pigeon = {2, 0};
    printGrid(pigeon);
    unsigned int steps = 0;
    while (!isTerminal(pigeon)) {
        ++steps;
        Action action = chooseAction(pigeon, target_net);
        pigeon = takeAction(pigeon, action);
        std::cout << "Action taken: " << actionToString(action) << std::endl;
        printGrid(pigeon);
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        if (steps > 10) {
            std::cout << "Too many steps. Exiting.\n";
            break;
        }
    }
    std::cout << "Game Over.\n";

    return 0;
}

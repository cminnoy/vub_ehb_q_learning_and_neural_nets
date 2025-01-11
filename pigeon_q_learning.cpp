#include <iostream>
#include <unordered_map>
#include <map>
#include <vector>
#include <random>
#include <tuple>
#include <string>
#include <iomanip>
#include <functional>

// Define actions
enum class Action { Up, Down, Left, Right };

// Define state as a pair of integers (x, y)
using State = std::pair<int, int>;

// Specialize std::hash for State
namespace std {
    template <>
    struct hash<State> {
        std::size_t operator()(const State& state) const {
            return std::hash<int>()(state.first) ^ (std::hash<int>()(state.second) << 1);
        }
    };
}

// Q-table structure
std::unordered_map<State, std::unordered_map<Action, double>> Q_table;

// Grid size
const int GRID_SIZE = 3;

// Rewards
const double REWARD_SEED_1 = 1.0;
const double REWARD_SEED_6 = 10.0;
const double REWARD_CAT = -10.0;
const double REWARD_EMPTY = -1.0;

// Epsilon decay parameters
double epsilon = 1.0; // Start with full exploration
const double epsilon_min = 0.01;
const double epsilon_decay = 0.995;

// Learning rate and discount factor
const double alpha = 0.1;  // Learning rate
const double Gamma = 0.9;  // Discount factor

// Random generator
std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<> dis(0.0, 1.0);

// Function to get reward for a specific state
double getReward(const State &state) {
    static bool seed_available = true;
    if (state == std::make_pair(0, 0)) { seed_available = false; return seed_available ? REWARD_SEED_1 : REWARD_EMPTY; }
    if (state == std::make_pair(2, 2)) return REWARD_SEED_6;
    if (state == std::make_pair(1, 1)) return REWARD_CAT;
    return REWARD_EMPTY;
}

// Check if a state is terminal
bool isTerminal(const State &state) {
    return state == std::make_pair(2, 2) || state == std::make_pair(1, 1);
}

// Get all possible actions
std::vector<Action> getActions() {
    return {Action::Up, Action::Down, Action::Left, Action::Right};
}

// Take an action and return the next state
State takeAction(const State &state, Action action) {
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

// Choose action using epsilon-greedy policy
Action chooseAction(const State &state) {
    if (dis(gen) < epsilon) {
        // Explore: choose a random action
        auto actions = getActions();
        std::uniform_int_distribution<> action_dist(0, actions.size() - 1);
        return actions[action_dist(gen)];
    } else {
        // Exploit: choose the best action from Q-table; default is up 
        double max_q = -1e9;
        Action best_action = Action::Up;
        for (const auto &action : getActions()) {
            double q_value = Q_table[state][action];
            if (q_value > max_q) {
                max_q = q_value;
                best_action = action;
            }
        }
        return best_action;
    }
}

// Print the grid state
void printGrid(const State &pigeon) {
    for (int i = 0; i < GRID_SIZE; i++) {
        for (int j = 0; j < GRID_SIZE; j++) {
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
        std::cout << std::endl;
    }
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

void printQTable() {
    // Define action labels
    const std::map<Action, std::string> action_labels = {
        {Action::Up, "Up"},
        {Action::Down, "Down"},
        {Action::Left, "Left"},
        {Action::Right, "Right"}
    };

    // Column width for alignment
    const int col_width = 12;

    // Print the header row with action labels
    std::cout << std::setw(col_width) << "State\\Action";
    for (const auto& [action, label] : action_labels) {
        std::cout << std::setw(col_width) << label;
    }
    std::cout << "\n";

    // Print the Q-values for each state
    for (int x = 0; x < GRID_SIZE; ++x) {
        for (int y = 0; y < GRID_SIZE; ++y) {
            State state = {x, y};

            // Print the state
            std::cout << std::setw(col_width) << "(" + std::to_string(state.first) + "," + std::to_string(state.second) + ")";

            // Print Q-values for actions
            for (const auto& action : {Action::Up, Action::Down, Action::Left, Action::Right}) {
                if (Q_table[state].find(action) != Q_table[state].end()) {
                    std::cout << std::setw(col_width) << std::fixed << std::setprecision(2) << Q_table[state][action];
                } else {
                    std::cout << std::setw(col_width) << "0.00"; // Default Q-value if not present
                }
            }
            std::cout << "\n";
        }
    }
    std::cout << "End of Q-Table.\n";
}

State getRandomStartPosition() {
    std::uniform_int_distribution<> pos_dist(0, GRID_SIZE - 1);
    State start;
    do {
        start = {pos_dist(gen), pos_dist(gen)};
    } while (start == std::make_pair(1, 1) || start == std::make_pair(2, 2)); // Exclude terminal states
    return start;
}

int main() {
    //State pigeon = {2, 0}; // Starting position
    int max_reward_encountered = 0;
    double max_reward = -std::numeric_limits<double>::infinity();
    int iterations = 1000; // Number of iterations for training

    for (int episode = 0; episode < iterations; ++episode) {
#if 0
        State pigeon = getRandomStartPosition();
#else
        State pigeon = {2, 0}; // Reset to start position
#endif
        double total_reward = 0.0;

        while (!isTerminal(pigeon)) {
            printGrid(pigeon);

            // Choose action
            Action action = chooseAction(pigeon);
            std::cout << "Action taken: " << actionToString(action) << std::endl;
            
            // Take action
            State next_state = takeAction(pigeon, action);
            double reward = getReward(next_state);
            total_reward += reward;

            // Update Q-value
            double max_next_q = -1e9;
            for (const auto &next_action : getActions()) {
                max_next_q = std::max(max_next_q, Q_table[next_state][next_action]);
            }

            double old_q_value = Q_table[pigeon][action];
            Q_table[pigeon][action] += alpha * (reward + Gamma * max_next_q - Q_table[pigeon][action]);
            double new_q_value = Q_table[pigeon][action];

            std::cout << "Updated Q-value for state (" << pigeon.first << "," << pigeon.second << ") and action " << actionToString(action) << ": " << old_q_value << " -> " << new_q_value << std::endl;

            pigeon = next_state; // Move to the next state

            std::cout << "Total reward: " << total_reward << '\n';

            printQTable();

            if (isTerminal(pigeon)) {
                printGrid(pigeon);
                std::cout << "Episode " << episode + 1 << " ended with points = " << total_reward << ".\n";
                break;
            }
        }

        if (total_reward < max_reward) {
            max_reward_encountered = 0;
        } else if (total_reward > max_reward) {
            max_reward = total_reward;
            max_reward_encountered = 0;
        } else {
            max_reward_encountered++;
        }

        if (max_reward_encountered >= 10) {
            std::cout << "Early stopping. Got 10 episodes with max reward of " << max_reward << " in a row." << std::endl;
            break;
        }

        // Decay epsilon
        epsilon = std::max(epsilon * epsilon_decay, epsilon_min);
    }

    std::cout << "Training complete.\n";
    printQTable();
    return 0;
}

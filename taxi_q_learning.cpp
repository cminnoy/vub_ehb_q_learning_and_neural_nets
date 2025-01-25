#include <iostream>
#include <unordered_map>
#include <map>
#include <vector>
#include <random>
#include <tuple>
#include <string>
#include <iomanip>
#include <thread>
#include <functional>

// Define actions
enum class Action { Up, Down, Left, Right };

// Define state as: (x, y, person_in_car)
struct State {
    std::pair<int,int> location;
    bool person_in_car;

    State(int x, int y, bool person_in_car) : location(x, y), person_in_car(person_in_car) {}
};

bool operator==(const State &lhs, const State &rhs) {
    return lhs.location == rhs.location and lhs.person_in_car == rhs.person_in_car;
}

// Specialize std::hash for State
namespace std {
    template <>
    struct hash<State> {
        std::size_t operator()(const State& state) const {
            return std::hash<int>()(state.location.first) ^
                   (std::hash<int>()(state.location.second) << 1) ^
                   (std::hash<bool>()(state.person_in_car) << 2);
        }
    };
}

// Q-table structure
std::unordered_map<State, std::unordered_map<Action, double>> Q_table;

// Grid size
const int GRID_SIZE = 5;

// Rewards
const double REWARD_PERSON = 1.0;
const double REWARD_HOME_SUCCESS = 10.0;
const double REWARD_HOME_EMPTY = -10.0;
const double REWARD_PIT = -10.0;
const double REWARD_EMPTY = -1.0;

// Epsilon decay parameters
double epsilon = 1.0; // Start with full exploration
const double epsilon_min = 0.01;
const double epsilon_decay = 0.995;

// Learning rate and discount factor
const double alpha = 0.1;  // Learning rate
const double Gamma = 0.9;  // Discount factor

// Person waiting
bool person_waiting = true;

// Random generator
std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<> dis(0.0, 1.0);

// Function to get reward for a specific state
double getReward(State const &state) {
    if (state.location == std::make_pair(3,0) and state.person_in_car == false) { return REWARD_PERSON; }
    if (state.location == std::make_pair(0,3)) return state.person_in_car ? REWARD_HOME_SUCCESS : REWARD_HOME_EMPTY;
    if (state.location == std::make_pair(2,2)) return REWARD_PIT;
    return REWARD_EMPTY;
}

// Check if a state is terminal
bool isTerminal(const State &state) {
    return state.location == std::make_pair(2, 2) or state.location == std::make_pair(0, 3);
}

// Get all possible actions
std::vector<Action> getActions() {
    return {Action::Up, Action::Down, Action::Left, Action::Right};
}

// Take an action and return the next state
State takeAction(const State &state, Action const action) {
    int x = state.location.first;
    int y = state.location.second;
    bool person = state.person_in_car;

    auto action_allowed = [x, y](const State &state, Action action) -> bool {
        switch (action) {
            case Action::Up: return x > 0;
            case Action::Down: return x < GRID_SIZE - 1;
            case Action::Left: return y > 0 and
                                !(state.location == std::make_pair(0, 2) or
                                  state.location == std::make_pair(3, 1) or
                                  state.location == std::make_pair(4, 1) or
                                  state.location == std::make_pair(3, 2) or
                                  state.location == std::make_pair(4, 2));
            case Action::Right: return y < GRID_SIZE - 1 and
                                !(state.location == std::make_pair(0, 1) or
                                  state.location == std::make_pair(3, 0) or
                                  state.location == std::make_pair(4, 0) or
                                  state.location == std::make_pair(3, 1) or
                                  state.location == std::make_pair(4, 1));
        }
        return false;
    };

    bool allowed = action_allowed(state, action);
    std::cout << "Allowed: " << (allowed ? "Yes" : "No") << std::endl;
    if (not allowed) return state;

    switch (action) {
        case Action::Up:
            x--;
            break;
        case Action::Down:
            x++;
            break;
        case Action::Left:
            y--;
            break;
        case Action::Right:
            y++;
            break;
    }

    if (x == 3 and y == 0 and person_waiting) {
        person = true;
        person_waiting = false;
    }

    return {x, y, person};
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
void printGrid(const State &taxi) {
    for (int i = 0; i < GRID_SIZE; i++) {
        for (int j = 0; j < GRID_SIZE; j++) {
            if (taxi.location == std::make_pair(i, j)) {
                std::cout << "Tx ";
            } else if (std::make_pair(i, j) == std::make_pair(3, 0)) {
                if (person_waiting) std::cout << "Pe ";
                else std::cout << ".  ";
            } else if (std::make_pair(i, j) == std::make_pair(0, 3)) {
                std::cout << "Ho ";
            } else if (std::make_pair(i, j) == std::make_pair(2, 2)) {
                std::cout << "Pt ";
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
    const int col_width = 15;

    // Print the header row with action labels
    std::cout << std::setw(col_width) << "State\\Action";
    for (const auto& [action, label] : action_labels) {
        std::cout << std::setw(col_width) << label;
    }
    std::cout << "\n";

    // Print the Q-values for each state
    for (int x = 0; x < GRID_SIZE; ++x) {
        for (int y = 0; y < GRID_SIZE; ++y) {
            for (bool person : {false, true}) { // Iterate over person_in_car values
                State state(x, y, person);

                // Print the state
                std::string state_info = "(" + std::to_string(x) + "," + std::to_string(y) + "," + (person ? "P" : "NP") + ")";
                std::cout << std::setw(col_width) << state_info;

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
    }

    std::cout << "End of Q-Table.\n";
}


int main() {
    unsigned int episodes = 500; // Number of episodes for training
    unsigned int home_runs = 0; // Number of successful home runs
    double max_reward = -std::numeric_limits<double>::infinity();
    unsigned int max_reward_encountered = 0;
    const int early_stopping_threshold = 10; // Number of episodes to trigger early stopping

    for (int episode = 0; episode < episodes; ++episode) {
        State taxi = {3, 1, false}; // Reset to start position
        person_waiting = true;
        double total_reward = 0.0;

        while (!isTerminal(taxi)) {
            std::cout << "\x1b[2J\x1b[H"; // Clear the terminal
            std::cout << "Episode " << (episode + 1) <<
                         "  (person in car " << taxi.person_in_car <<
                         ", person waiting " << person_waiting <<
                         ", epsilon " << epsilon <<
                         ", max reward " << max_reward <<
                         ", home runs " << home_runs <<
                         "):\n";
            printGrid(taxi);

            // Choose action
            Action action = chooseAction(taxi);
            std::cout << "Action taken: " << actionToString(action) << std::endl;
            
            // Take action
            State next_state = takeAction(taxi, action);
            double reward = getReward(next_state);
            total_reward += reward;

            // Update Q-value
            double max_next_q = -1e9;
            for (const auto &next_action : getActions()) {
                max_next_q = std::max(max_next_q, Q_table[next_state][next_action]);
            }

            double old_q_value = Q_table[taxi][action];
            Q_table[taxi][action] += alpha * (reward + Gamma * max_next_q - Q_table[taxi][action]);
            double new_q_value = Q_table[taxi][action];

            std::cout << "Updated Q-value for state (" << taxi.location.first << "," << taxi.location.second << ") and action " << actionToString(action) << ": " << old_q_value << " -> " << new_q_value << std::endl;

            taxi = next_state; // Move to the next state
            std::cout << "Total reward: " << total_reward << '\n';
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        if (taxi.location == std::make_pair(0, 3) and taxi.person_in_car) {
            home_runs++;
        }

        // Early stopping logic
        if (total_reward < max_reward) {
            max_reward_encountered = 0;
        } else if (total_reward > max_reward) {
            max_reward = total_reward;
            max_reward_encountered = 0;
        } else {
            max_reward_encountered++;
        }
        if (max_reward_encountered >= early_stopping_threshold) {
            std::cout << "Early stopping triggered after " << episode + 1 << " episodes. Max reward = " << max_reward << ".\n";
            break;
        }

        // Decay epsilon
        epsilon = std::max(epsilon * epsilon_decay, epsilon_min);

        std::cout << "Episode " << episode + 1 << " ended with points = " << total_reward << ".\n";
    }

    std::cout << "\x1b[2J\x1b[H"; // Clear the terminal
    std::cout << "Training complete.\n";

    // After training loop
    std::cout << "Final Q-table:\n";
    printQTable();

    // Test the final policy
    std::cout << "Testing the final policy:\n";
    State taxi = {3, 1, false};
    person_waiting = true;
    printGrid(taxi);
    unsigned int steps = 0;
    epsilon = 0.0; // No exploration
    while (!isTerminal(taxi)) {
        ++steps;
        Action action = chooseAction(taxi);
        taxi = takeAction(taxi, action);
        std::cout << "Action taken: " << actionToString(action) << std::endl;
        printGrid(taxi);
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        if (steps > 20) {
            std::cout << "Too many steps. Exiting.\n";
            break;
        }
    }
    std::cout << "Game Over.\n";
    return 0;
}

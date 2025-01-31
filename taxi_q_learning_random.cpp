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

// Define state as: (x, y, person_x, person_y, person_in_car)
struct State {
    std::pair<int, int> location;
    std::pair<int, int> person_location;
    bool person_in_car;

    State(int x, int y, int pers_x, int pers_y, bool person_in_car)
        : location(x, y), person_location(pers_x, pers_y), person_in_car(person_in_car) {}
};

bool operator==(const State &lhs, const State &rhs) {
    return lhs.location == rhs.location and
           lhs.person_location == rhs.person_location and
           lhs.person_in_car == rhs.person_in_car;
}

std::ostream & operator<<(std::ostream &os, const State &state) {
    os << "(" << state.location.first << "," << state.location.second << "," << state.person_location.first << ',' << state.person_location.second << ',' << state.person_in_car << ")";
    return os;
}

// Specialize std::hash for State
namespace std {
    template <>
    struct hash<State> {
        std::size_t operator()(const State &state) const {
            return std::hash<int>()(state.location.first) ^
                   (std::hash<int>()(state.location.second) << 1) ^
                   (std::hash<int>()(state.person_location.first) << 2) ^
                   (std::hash<int>()(state.person_location.second) << 3) ^
                   (std::hash<bool>()(state.person_in_car) << 4);
        }
    };
}

// Q-table structure
std::unordered_map<State, std::unordered_map<Action, double>> Q_table;

// Grid size
const int GRID_SIZE = 5;

// Rewards
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

// Person location
std::pair<int, int> person_location;
bool person_waiting = true;

// Random generator
std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<> dis(0.0, 1.0);
std::uniform_int_distribution<> random_location(0, GRID_SIZE - 1);

// Function to get reward for a specific state
double getReward(const State &state) {
    if (state.location == std::make_pair(0, 3)) {
        return state.person_in_car ? REWARD_HOME_SUCCESS : REWARD_HOME_EMPTY;
    }
    if (state.location == std::make_pair(2, 2)) {
        return REWARD_PIT;
    }
    return REWARD_EMPTY;
}

// Check if a state is terminal
bool isTerminal(const State &state) {
    return state.location == std::make_pair(2, 2) || state.location == std::make_pair(0, 3);
}

// Get all possible actions
std::vector<Action> getActions() {
    return {Action::Up, Action::Down, Action::Left, Action::Right};
}

// Take an action and return the next state
State takeAction(const State &state, Action action) {
    int x = state.location.first;
    int y = state.location.second;
    bool person_in_car = state.person_in_car;

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
        case Action::Up: x--; break;
        case Action::Down: x++; break;
        case Action::Left: y--; break;
        case Action::Right: y++; break;
    }

    // Picking up person
    if (x == person_location.first && y == person_location.second && person_waiting) {
        person_in_car = true;
        person_waiting = false;
    }

    // Return new state
    if (person_in_car) // to reduce state space, we don't track person location when they are in the car
        return {x, y, 0, 0, person_in_car};
    else
        return {x, y, person_location.first, person_location.second, person_in_car};
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
            } else if (std::make_pair(i, j) == person_location) {
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

auto generateRandomPersonLocation() {
    std::pair<int, int> person_location = {random_location(gen), random_location(gen)};
    while (person_location == std::make_pair(0, 3) || person_location == std::make_pair(2, 2)) {
        person_location = {random_location(gen), random_location(gen)};
    }
    return person_location;
}

int main() {
    const double min_delay = 1.0;       // in milliseconds
    const double max_delay = 100.0;     // in milliseconds
    unsigned int episodes = 4000; // Number of episodes for training
    unsigned int home_runs = 0; // Number of successful home runs
    double max_reward = -std::numeric_limits<double>::infinity();
    unsigned int max_reward_encountered = 0;
    const int early_stopping_threshold = 10; // Number of episodes to trigger early stopping

    for (int episode = 0; episode < episodes; ++episode) {
        double progress = static_cast<double>(episode) / episodes;
        double delay_ms = min_delay + (max_delay - min_delay) * std::log10(1 + 9 * progress);

        // Randomly place the person
        person_location = generateRandomPersonLocation();

        State taxi = {3, 1, person_location.first, person_location.second, false};
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
            std::cout << "Orignial state: " << taxi << std::endl;
            State next_state = takeAction(taxi, action);
            std::cout << "Next state: " << next_state << std::endl;
            double reward = getReward(next_state);
            std::cout << "Reward: " << reward << std::endl;
            total_reward += reward;

            // Update Q-value
            double max_next_q = -1e9;
            for (const auto &next_action : getActions()) {
                max_next_q = std::max(max_next_q, Q_table[next_state][next_action]);
            }

            double old_q_value = Q_table[taxi][action];
            Q_table[taxi][action] += alpha * (reward + Gamma * max_next_q - Q_table[taxi][action]);
            double new_q_value = Q_table[taxi][action];

            std::cout << "Updated Q-value for state " << taxi << " and action " << actionToString(action) << ": " << old_q_value << " -> " << new_q_value << std::endl;

            taxi = next_state; // Move to the next state
            std::cout << "Total reward: " << total_reward << '\n';
            std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(delay_ms)));
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

    // Test the final policy
    std::cout << "Testing the final policy:\n";

    // Reset the initial state
    person_location = generateRandomPersonLocation();
    State taxi = {3, 1, person_location.first, person_location.second, false};
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

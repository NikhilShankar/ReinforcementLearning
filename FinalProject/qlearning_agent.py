import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import pickle
from battleship_env import BattleshipEnv, setup_results_directory


class QLearningAgent:
    """
    Q-Learning agent for the Battleship game.
    """

    def __init__(self, grid_size=5, learning_rate=0.1, discount_factor=0.95,
                 exploration_rate=1.0, exploration_decay=0.995, min_exploration_rate=0.01):
        self.grid_size = grid_size
        self.name = "QLearningAgent"
        self.state_size = 3 ** (grid_size * grid_size)   # 3 possible values for each cell (0, 1, 2)
        self.action_size = grid_size * grid_size

        # Q-learning hyperparameters
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate

        # Initialize Q-table - this would be enormous for larger grids!
        # For a 5x5 grid, there would be 3^25 possible states which is impractical
        # Instead, we'll use a more compact state representation
        self.reset()

    def reset(self):
        """Reset the agent's state for a new game."""
        self.available_actions = set(range(self.grid_size * self.grid_size))

        # For a practical Q-table, we'll use a dictionary to store only the visited state-action pairs
        # State will be represented as a tuple of the flattened grid values
        self.q_table = {}

    def _state_to_key(self, observation):
        """
        Convert the observation to a hashable state key.
        We'll use a simplified representation to keep the state space manageable.

        Args:
            observation: The current state observation (player's view of the grid)

        Returns:
            A hashable key representing the state
        """
        # Convert the 2D array to a tuple (which is hashable)
        return tuple(observation.flatten())

    def _get_q_value(self, state_key, action):
        """
        Get Q-value for a state-action pair.

        Args:
            state_key: Hashable state key
            action: Action index

        Returns:
            Q-value for the state-action pair
        """
        if (state_key, action) not in self.q_table:
            # Initialize Q-value to 0 for unseen state-action pairs
            self.q_table[(state_key, action)] = 0.0

        return self.q_table[(state_key, action)]

    def _set_q_value(self, state_key, action, value):
        """
        Set Q-value for a state-action pair.

        Args:
            state_key: Hashable state key
            action: Action index
            value: New Q-value
        """
        self.q_table[(state_key, action)] = value

    def act(self, observation):
        """
        Select an action based on the current state using epsilon-greedy policy.

        Args:
            observation: The current state observation (player's view of the grid)

        Returns:
            Action to take (cell to fire at)
        """
        state_key = self._state_to_key(observation)

        # Filter actions to only include available ones
        valid_actions = list(self.available_actions)

        if not valid_actions:
            # If we've somehow tried all cells, return a random cell
            # (should never happen in a normal game)
            return np.random.randint(0, self.grid_size * self.grid_size)

        # Exploration: choose a random action
        if np.random.rand() < self.exploration_rate:
            action = np.random.choice(valid_actions)
        # Exploitation: choose the best action
        else:
            # Get Q-values for all valid actions
            q_values = [self._get_q_value(state_key, a) for a in valid_actions]

            # Find indices of actions with maximum Q-value (there might be multiple)
            max_indices = np.where(q_values == np.max(q_values))[0]

            # Select a random action among those with maximum Q-value
            action = valid_actions[np.random.choice(max_indices)]

        # Remove the chosen action from available actions
        self.available_actions.remove(action)

        return action

    def update(self, observation, action, reward, next_observation, done, info):
        """
        Update the Q-table based on the outcome of the action.

        Args:
            observation: Previous observation
            action: Action taken
            reward: Reward received
            next_observation: New observation
            done: Whether the episode is done
            info: Additional information
        """
        state_key = self._state_to_key(observation)
        next_state_key = self._state_to_key(next_observation)

        # Get current Q-value
        current_q = self._get_q_value(state_key, action)

        # Calculate new Q-value (standard Q-learning formula)
        if done:
            # If the episode is done, there is no future reward
            new_q = current_q + self.learning_rate * (reward - current_q)
        else:
            # Get maximum Q-value for the next state (considering only available actions)
            valid_actions = list(self.available_actions)
            if valid_actions:
                next_q_values = [self._get_q_value(next_state_key, a) for a in valid_actions]
                max_next_q = np.max(next_q_values)
            else:
                max_next_q = 0.0

            new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)

        # Update Q-table
        self._set_q_value(state_key, action, new_q)

        # Decay exploration rate
        self.exploration_rate = max(self.min_exploration_rate,
                                    self.exploration_rate * self.exploration_decay)

    def save(self, filepath):
        """Save the agent."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'name': self.name,
                'grid_size': self.grid_size,
                'q_table': self.q_table,
                'learning_rate': self.learning_rate,
                'discount_factor': self.discount_factor,
                'exploration_rate': self.exploration_rate,
                'exploration_decay': self.exploration_decay,
                'min_exploration_rate': self.min_exploration_rate
            }, f)

    def load(self, filepath):
        """Load the agent from a file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.name = data['name']
            self.grid_size = data['grid_size']
            self.q_table = data['q_table']
            self.learning_rate = data['learning_rate']
            self.discount_factor = data['discount_factor']
            self.exploration_rate = data['exploration_rate']
            self.exploration_decay = data['exploration_decay']
            self.min_exploration_rate = data['min_exploration_rate']
            self.reset()
            # Restore Q-table from saved data
            self.q_table = data['q_table']


def train_qlearning_agent(episodes=1000, grid_size=5, ships_config=None,
                          learning_rate=0.1, discount_factor=0.95,
                          exploration_rate=1.0, exploration_decay=0.995,
                          min_exploration_rate=0.01):
    """
    Train the Q-learning agent.

    Args:
        episodes: Number of games to play
        grid_size: Size of the game grid
        ships_config: Configuration of ships [ship1_size, ship2_size, ...]
        learning_rate: Learning rate for Q-learning
        discount_factor: Discount factor for future rewards
        exploration_rate: Initial exploration rate
        exploration_decay: Rate at which exploration decays
        min_exploration_rate: Minimum exploration rate

    Returns:
        agent: The trained agent
        results: Dictionary containing performance metrics
    """
    if ships_config is None:
        ships_config = [3, 2]  # Default ship configuration for 5x5 grid

    env = BattleshipEnv(grid_size=grid_size, ships_config=ships_config)
    agent = QLearningAgent(grid_size=grid_size, learning_rate=learning_rate,
                           discount_factor=discount_factor,
                           exploration_rate=exploration_rate,
                           exploration_decay=exploration_decay,
                           min_exploration_rate=min_exploration_rate)

    # Set up results directory
    results_dir = setup_results_directory(agent.name, "training")

    # Initialize metrics tracking
    shots_history = []
    rewards_history = []
    win_rate = 0

    for episode in range(episodes):
        observation = env.reset()
        agent.reset()
        done = False
        episode_shots = 0
        episode_reward = 0

        while not done:
            action = agent.act(observation)
            next_observation, reward, done, info = env.step(action)
            agent.update(observation, action, reward, next_observation, done, info)

            observation = next_observation
            episode_shots += 1
            episode_reward += reward

        # Record metrics
        shots_history.append(episode_shots)
        rewards_history.append(episode_reward)
        if env.hits == env.total_ship_cells:
            win_rate += 1

        # Print progress every 100 episodes
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{episodes} completed.")
            print(f"Recent average shots: {np.mean(shots_history[-100:]):.2f}")
            print(f"Recent average reward: {np.mean(rewards_history[-100:]):.2f}")
            print(f"Exploration rate: {agent.exploration_rate:.4f}")

    # Calculate final metrics
    win_rate = win_rate / episodes
    avg_shots = np.mean(shots_history)
    avg_reward = np.mean(rewards_history)

    # Save results
    results = {
        'episodes': episodes,
        'win_rate': win_rate,
        'avg_shots': avg_shots,
        'avg_reward': avg_reward,
        'shots_history': shots_history,
        'rewards_history': rewards_history
    }

    # Save agent
    agent_path = os.path.join(results_dir, 'qlearning_agent.pkl')
    agent.save(agent_path)

    # Save metrics as CSV
    metrics_df = pd.DataFrame({
        'episode': range(1, episodes + 1),
        'shots': shots_history,
        'reward': rewards_history
    })
    metrics_df.to_csv(os.path.join(results_dir, 'metrics.csv'), index=False)

    # Plot learning curves
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

    # Plot shots curve
    ax1.plot(range(1, episodes + 1), shots_history)
    ax1.set_title(f'Q-Learning Agent Shots Performance\nAvg. Shots: {avg_shots:.2f}, Win Rate: {win_rate:.2f}')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Shots to Complete Game')
    ax1.grid(True)

    # Plot rewards curve
    ax2.plot(range(1, episodes + 1), rewards_history)
    ax2.set_title(f'Q-Learning Agent Rewards\nAvg. Reward: {avg_reward:.2f}')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Total Episode Reward')
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'performance.png'))
    plt.close()

    # Plot moving averages
    window_size = min(100, episodes)
    shots_moving_avg = [np.mean(shots_history[max(0, i - window_size):i])
                        for i in range(1, episodes + 1)]
    rewards_moving_avg = [np.mean(rewards_history[max(0, i - window_size):i])
                          for i in range(1, episodes + 1)]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

    # Plot shots moving average
    ax1.plot(range(1, episodes + 1), shots_moving_avg)
    ax1.set_title(f'Q-Learning Agent Shots Moving Average ({window_size} episodes)')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Average Shots to Complete Game')
    ax1.grid(True)

    # Plot rewards moving average
    ax2.plot(range(1, episodes + 1), rewards_moving_avg)
    ax2.set_title(f'Q-Learning Agent Rewards Moving Average ({window_size} episodes)')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Average Episode Reward')
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'moving_average.png'))
    plt.close()

    print(f"\nTraining completed for Q-Learning Agent")
    print(f"Average shots to win: {avg_shots:.2f}")
    print(f"Average reward: {avg_reward:.2f}")
    print(f"Win rate: {win_rate:.2f}")
    print(f"Results saved to: {results_dir}")

    return agent, results


def evaluate_qlearning_agent(agent, test_episodes=50, grid_size=5, ships_config=None, fixed_test=False):
    """
    Evaluate the Q-learning agent on test episodes.

    Args:
        agent: The agent to evaluate
        test_episodes: Number of test episodes
        grid_size: Size of the game grid
        ships_config: Configuration of ships
        fixed_test: Whether to use fixed ship positions for testing

    Returns:
        results: Dictionary containing performance metrics
    """
    from battleship_env import create_test_scenario

    if ships_config is None:
        ships_config = [3, 2]  # Default ship configuration for 5x5 grid

    # Set up results directory
    results_dir = setup_results_directory(agent.name, "testing")

    # Initialize metrics tracking
    shots_history = []
    rewards_history = []
    win_rate = 0

    # Set exploration rate to minimum for evaluation
    original_exploration_rate = agent.exploration_rate
    agent.exploration_rate = agent.min_exploration_rate

    for episode in range(test_episodes):
        if fixed_test:
            # Use the standard test scenario
            env = create_test_scenario(grid_size=grid_size, ships_config=ships_config)
        else:
            # Use random ship placements
            env = BattleshipEnv(grid_size=grid_size, ships_config=ships_config)

        agent.reset()
        observation = env.reset()  # This is redundant for fixed_test but kept for consistency
        done = False
        episode_shots = 0
        episode_reward = 0

        while not done:
            action = agent.act(observation)
            observation, reward, done, info = env.step(action)
            episode_shots += 1
            episode_reward += reward

            # Prevent infinite loops
            if episode_shots > grid_size * grid_size:
                break

        # Record metrics
        shots_history.append(episode_shots)
        rewards_history.append(episode_reward)
        if env.hits == env.total_ship_cells:
            win_rate += 1

    # Restore original exploration rate
    agent.exploration_rate = original_exploration_rate

    # Calculate final metrics
    win_rate = win_rate / test_episodes
    avg_shots = np.mean(shots_history)
    avg_reward = np.mean(rewards_history)

    # Save results
    results = {
        'test_episodes': test_episodes,
        'win_rate': win_rate,
        'avg_shots': avg_shots,
        'avg_reward': avg_reward,
        'shots_history': shots_history,
        'rewards_history': rewards_history,
        'fixed_test': fixed_test
    }

    # Save metrics as CSV
    metrics_df = pd.DataFrame({
        'episode': range(1, test_episodes + 1),
        'shots': shots_history,
        'reward': rewards_history
    })
    metrics_df.to_csv(os.path.join(results_dir, 'test_metrics.csv'), index=False)

    # Plot histogram of shots distribution
    plt.figure(figsize=(10, 6))
    plt.hist(shots_history, bins=range(min(shots_history), max(shots_history) + 2))
    plt.title(f'Q-Learning Agent Test Performance\nAvg. Shots: {avg_shots:.2f}, Win Rate: {win_rate:.2f}')
    plt.xlabel('Shots to Complete Game')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, 'test_histogram.png'))
    plt.close()

    print(f"\nEvaluation completed for Q-Learning Agent")
    print(f"Average shots to win: {avg_shots:.2f}")
    print(f"Average reward: {avg_reward:.2f}")
    print(f"Win rate: {win_rate:.2f}")
    print(f"Results saved to: {results_dir}")

    return results


if __name__ == "__main__":
    # Example usage
    # Train the Q-learning agent (this could take a while)
    agent, train_results = train_qlearning_agent(episodes=1000)

    # Evaluate on fixed test scenario
    test_results_fixed = evaluate_qlearning_agent(agent, test_episodes=50, fixed_test=True)

    # Evaluate on random scenarios
    test_results_random = evaluate_qlearning_agent(agent, test_episodes=50, fixed_test=False)
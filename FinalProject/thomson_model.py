import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import pickle
from battleship_env import BattleshipEnv, setup_results_directory


class ThomsonSamplingAgent:
    """
    Thomson Sampling agent for the Battleship game.
    """

    def __init__(self, grid_size=5, alpha_prior=1.0, beta_prior=1.0):
        self.grid_size = grid_size
        self.name = "ThomsonSamplingAgent"
        self.action_size = grid_size * grid_size

        # Prior parameters for Beta distribution
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior

        self.reset()

    def reset(self):
        """Reset the agent's state for a new game."""
        self.available_actions = set(range(self.grid_size * self.grid_size))

        # Initialize alpha and beta parameters for each cell
        # Alpha represents "pseudo-hits" and beta represents "pseudo-misses"
        self.alpha = np.ones((self.grid_size, self.grid_size)) * self.alpha_prior
        self.beta = np.ones((self.grid_size, self.grid_size)) * self.beta_prior

        ##IDEA 3 Incorporated to thomson_sampling model
        # Create a checkerboard/diagonal pattern with slightly higher initial alpha values
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if (r + c) % 2 == 0:  # Checkerboard pattern
                    self.alpha[r, c] += 0.5  # Slight advantage

    def act(self, observation):
        """
        Select an action based on Thomson Sampling.

        Args:
            observation: The current state observation (player's view of the grid)

        Returns:
            Action to take (cell to fire at)
        """
        # Filter actions to only include available ones
        valid_actions = list(self.available_actions)

        if not valid_actions:
            # If we've somehow tried all cells, return a random cell
            # (should never happen in a normal game)
            return np.random.randint(0, self.grid_size * self.grid_size)

        # Sample from Beta distribution for each valid action
        samples = np.zeros(self.action_size)
        for action in valid_actions:
            row, col = self._action_to_coord(action)
            # Sample from Beta distribution using current alpha and beta
            samples[action] = np.random.beta(self.alpha[row, col], self.beta[row, col])

        # Select action with highest sampled value
        # Only consider valid actions
        valid_samples = [samples[a] for a in valid_actions]
        max_idx = np.argmax(valid_samples)
        action = valid_actions[max_idx]

        # Remove the chosen action from available actions
        self.available_actions.remove(action)

        return action

    def update(self, observation, action, reward, next_observation, done, info):
        """
        Update the Beta distribution parameters based on the outcome of the action.

        Args:
            observation: Previous observation
            action: Action taken
            reward: Reward received
            next_observation: New observation
            done: Whether the episode is done
            info: Additional information
        """
        row, col = self._action_to_coord(action)

        # Update alpha (hit counter) and beta (miss counter) based on reward
        if reward > 0:  # Hit
            self.alpha[row, col] += 1

            # Also update neighboring cells if this was a hit
            # This incorporates the knowledge that ships are continuous
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nr, nc = row + dr, col + dc
                if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                    # Increase alpha slightly for neighboring cells
                    neighbor_action = self._coord_to_action(nr, nc)
                    if neighbor_action in self.available_actions:
                        self.alpha[nr, nc] += 100
        else:  # Miss
            # We could also slightly reduce the probability of adjacent cells
            # on a miss, since ships won't be there if it's isolated
            isolated_miss = True
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nr, nc = row + dr, col + dc
                if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                    # Check if there's a hit in observation
                    if observation[nr, nc] == 1:  # Assuming 1 represents a hit
                        isolated_miss = False
                        break

            if isolated_miss:
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nr, nc = row + dr, col + dc
                    if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                        neighbor_action = self._coord_to_action(nr, nc)
                        if neighbor_action in self.available_actions:
                            # Slightly increase beta for neighboring cells of isolated misses
                            self.beta[nr, nc] += 1

    def _action_to_coord(self, action):
        """Convert action index to grid coordinates."""
        row = action // self.grid_size
        col = action % self.grid_size
        return row, col

    def _coord_to_action(self, row, col):
        """Convert grid coordinates to action index."""
        return row * self.grid_size + col

    def save(self, filepath):
        """Save the agent."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'name': self.name,
                'grid_size': self.grid_size,
                'alpha': self.alpha,
                'beta': self.beta,
                'alpha_prior': self.alpha_prior,
                'beta_prior': self.beta_prior
            }, f)

    def load(self, filepath):
        """Load the agent from a file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.name = data['name']
            self.grid_size = data['grid_size']
            self.alpha_prior = data['alpha_prior']
            self.beta_prior = data['beta_prior']
            self.reset()
            # Restore parameters from saved data
            self.alpha = data['alpha']
            self.beta = data['beta']


def train_thomson_sampling_agent(episodes=1000, grid_size=5, ships_config=None,
                                 alpha_prior=1.0, beta_prior=1.0):
    """
    Train the Thomson Sampling agent.

    Args:
        episodes: Number of games to play
        grid_size: Size of the game grid
        ships_config: Configuration of ships [ship1_size, ship2_size, ...]
        alpha_prior: Prior parameter for alpha
        beta_prior: Prior parameter for beta

    Returns:
        agent: The trained agent
        results: Dictionary containing performance metrics
    """
    if ships_config is None:
        ships_config = [3, 2]  # Default ship configuration for 5x5 grid

    env = BattleshipEnv(grid_size=grid_size, ships_config=ships_config)
    agent = ThomsonSamplingAgent(grid_size=grid_size, alpha_prior=alpha_prior, beta_prior=beta_prior)

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
    agent_path = os.path.join(results_dir, 'thomson_sampling_agent.pkl')
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
    ax1.set_title(f'Thomson Sampling Agent Shots Performance\nAvg. Shots: {avg_shots:.2f}, Win Rate: {win_rate:.2f}')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Shots to Complete Game')
    ax1.grid(True)

    # Plot rewards curve
    ax2.plot(range(1, episodes + 1), rewards_history)
    ax2.set_title(f'Thomson Sampling Agent Rewards\nAvg. Reward: {avg_reward:.2f}')
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
    ax1.set_title(f'Thomson Sampling Agent Shots Moving Average ({window_size} episodes)')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Average Shots to Complete Game')
    ax1.grid(True)

    # Plot rewards moving average
    ax2.plot(range(1, episodes + 1), rewards_moving_avg)
    ax2.set_title(f'Thomson Sampling Agent Rewards Moving Average ({window_size} episodes)')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Average Episode Reward')
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'moving_average.png'))
    plt.close()

    print(f"\nTraining completed for Thomson Sampling Agent")
    print(f"Average shots to win: {avg_shots:.2f}")
    print(f"Average reward: {avg_reward:.2f}")
    print(f"Win rate: {win_rate:.2f}")
    print(f"Results saved to: {results_dir}")

    return agent, results


def evaluate_thomson_sampling_agent(agent, test_episodes=50, grid_size=5, ships_config=None, fixed_test=False):
    """
    Evaluate the Thomson Sampling agent on test episodes.

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
    plt.title(f'Thomson Sampling Agent Test Performance\nAvg. Shots: {avg_shots:.2f}, Win Rate: {win_rate:.2f}')
    plt.xlabel('Shots to Complete Game')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, 'test_histogram.png'))
    plt.close()

    print(f"\nEvaluation completed for Thomson Sampling Agent")
    print(f"Average shots to win: {avg_shots:.2f}")
    print(f"Average reward: {avg_reward:.2f}")
    print(f"Win rate: {win_rate:.2f}")
    print(f"Results saved to: {results_dir}")

    return results


if __name__ == "__main__":
    # Example usage
    # Train the Thomson Sampling agent
    agent, train_results = train_thomson_sampling_agent(episodes=1000)

    # Evaluate on fixed test scenario
    test_results_fixed = evaluate_thomson_sampling_agent(agent, test_episodes=50, fixed_test=True)

    # Evaluate on random scenarios
    test_results_random = evaluate_thomson_sampling_agent(agent, test_episodes=50, fixed_test=False)
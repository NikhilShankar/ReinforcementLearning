import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import pickle
from tqdm import tqdm
from copy import deepcopy

from battleship_env import BattleshipEnv, setup_results_directory


def compare_agents_same_config(agents, test_episodes=50, grid_size=5, ships_config=None):
    """
    Compare multiple agents on exactly the same test scenarios, where agents race to solve
    the same battleship configuration. This evaluates which agent can find ships faster.

    Args:
        agents: List of agent objects to compare
        test_episodes: Number of test episodes
        grid_size: Size of the game grid
        ships_config: Configuration of ships
        fixed_test: Whether to use fixed ship positions for testing

    Returns:
        results: Dictionary containing performance metrics for all agents
    """
    if ships_config is None:
        ships_config = [3, 2]  # Default ship configuration for 5x5 grid

    # Set up results directory
    results_dir = setup_results_directory("same_config_comparison", "testing")

    agent_names = [agent.name for agent in agents]

    # Initialize metrics tracking
    all_shots_history = {name: [] for name in agent_names}
    all_win_rates = {name: 0 for name in agent_names}
    all_wins = {name: 0 for name in agent_names}  # Track who wins each episode
    episode_winners = []  # Store the winner for each episode

    # For each episode, we'll run all agents simultaneously to see who solves it first
    for episode in range(test_episodes):
        # Create a random ship configuration to use for all agents
        base_env = BattleshipEnv(grid_size=grid_size, ships_config=ships_config)
        ship_grid = base_env.ship_grid.copy()
        print(f"ship_grid {ship_grid}")

        # Create parallel environments with identical ship configurations
        envs = []
        for _ in agents:
            env = BattleshipEnv(grid_size=grid_size, ships_config=ships_config)
            env.ship_grid = ship_grid.copy()  # Use the same ship placement
            env.reset()  # Reset other environment variables
            env.ship_grid = ship_grid.copy()  # Make sure ship grid is preserved
            envs.append(env)

        # Reset all agents
        for agent in agents:
            agent.reset()

        # Set up parallel game states
        observations = [env.grid.copy() for env in envs]
        done_flags = [False] * len(agents)
        shot_counts = [0] * len(agents)
        winner_found = False

        # Run all agents in parallel until all are done
        while not all(done_flags):
            for i, (agent, env, obs, done) in enumerate(zip(agents, envs, observations, done_flags)):
                if done:
                    continue

                # Get action from agent
                action = agent.act(obs)

                # Take action in environment
                new_obs, reward, done, info = env.step(action)
                observations[i] = new_obs
                shot_counts[i] += 1
                done_flags[i] = done

                # Check if this agent finished first (winner)
                if done and not winner_found:
                    winner_found = True
                    all_wins[agent.name] += 1
                    episode_winners.append(agent.name)

                # Prevent infinite loops
                if shot_counts[i] > grid_size * grid_size:
                    done_flags[i] = True

        # Record metrics for all agents
        for i, agent in enumerate(agents):
            all_shots_history[agent.name].append(shot_counts[i])
            if envs[i].hits == envs[i].total_ship_cells:
                all_win_rates[agent.name] += 1

        # Print progress
        if (episode + 1) % 10 == 0:
            print(f"Comparison episode {episode + 1}/{test_episodes} completed. {all_shots_history}")

    # Calculate final metrics
    for name in agent_names:
        all_win_rates[name] = all_win_rates[name] / test_episodes

    # Save results
    results = {
        'test_episodes': test_episodes,
        'win_rates': all_win_rates,  # Percentage of episodes where agent found all ships
        'race_wins': all_wins,  # Number of episodes where agent found ships first
        'shots_history': all_shots_history,
        'episode_winners': episode_winners,
        'grid_size': grid_size,
        'ships_config': ships_config
    }

    # Save metrics as CSV
    metrics_df = pd.DataFrame({
        'episode': range(1, test_episodes + 1),
        **{name: all_shots_history[name] for name in agent_names},
        'winner': episode_winners
    })
    metrics_df.to_csv(os.path.join(results_dir, 'comparison_metrics.csv'), index=False)

    # Calculate statistics
    stats_data = {
        'Agent': agent_names,
        'Win Rate': [all_win_rates[name] for name in agent_names],
        'Race Wins': [all_wins[name] for name in agent_names],
        'Race Win %': [all_wins[name] / test_episodes for name in agent_names],
        'Avg Shots': [np.mean(all_shots_history[name]) for name in agent_names],
        'Min Shots': [np.min(all_shots_history[name]) for name in agent_names],
        'Max Shots': [np.max(all_shots_history[name]) for name in agent_names],
        'Std Dev': [np.std(all_shots_history[name]) for name in agent_names]
    }

    stats_df = pd.DataFrame(stats_data)
    stats_df.to_csv(os.path.join(results_dir, 'comparison_stats.csv'), index=False)

    # Plot comparisons
    # Box plot of shots distribution
    plt.figure(figsize=(12, 8))
    plt.boxplot([all_shots_history[name] for name in agent_names], labels=agent_names)
    plt.title(f'Shots Distribution Comparison ({test_episodes} episodes)')
    plt.xlabel('Agent')
    plt.ylabel('Shots to Complete Game')
    plt.grid(True, axis='y')
    plt.savefig(os.path.join(results_dir, 'boxplot_comparison.png'))
    plt.close()

    # Bar chart of average shots
    plt.figure(figsize=(12, 8))
    avg_shots = [np.mean(all_shots_history[name]) for name in agent_names]
    std_shots = [np.std(all_shots_history[name]) for name in agent_names]

    bars = plt.bar(agent_names, avg_shots, yerr=std_shots, capsize=10)
    plt.title(f'Average Shots Comparison ({test_episodes} episodes)')
    plt.xlabel('Agent')
    plt.ylabel('Average Shots to Complete Game')
    plt.grid(True, axis='y')

    # Add value labels on top of the bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                 f'{height:.1f}', ha='center', va='bottom')

    plt.savefig(os.path.join(results_dir, 'avgshots_comparison.png'))
    plt.close()

    # Bar chart of win rates
    plt.figure(figsize=(12, 8))
    win_rates = [all_win_rates[name] for name in agent_names]

    bars = plt.bar(agent_names, win_rates)
    plt.title(f'Win Rate Comparison ({test_episodes} episodes)')
    plt.xlabel('Agent')
    plt.ylabel('Win Rate')
    plt.ylim(0, 1.1)  # Win rate should be between 0 and 1
    plt.grid(True, axis='y')

    # Add percentage labels on top of the bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                 f'{height:.1%}', ha='center', va='bottom')

    plt.savefig(os.path.join(results_dir, 'winrate_comparison.png'))
    plt.close()

    # Bar chart of race wins (new metric)
    plt.figure(figsize=(12, 8))
    race_wins = [all_wins[name] for name in agent_names]
    race_win_pct = [all_wins[name] / test_episodes for name in agent_names]

    bars = plt.bar(agent_names, race_win_pct)
    plt.title(f'Race Win Percentage ({test_episodes} episodes)')
    plt.xlabel('Agent')
    plt.ylabel('Percentage of Games Won')
    plt.ylim(0, 1.1)
    plt.grid(True, axis='y')

    # Add percentage labels on top of the bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                 f'{height:.1%}', ha='center', va='bottom')

    plt.savefig(os.path.join(results_dir, 'race_wins_comparison.png'))
    plt.close()

    # Line plot of shots per episode for all agents
    plt.figure(figsize=(12, 8))
    for name in agent_names:
        plt.plot(range(1, test_episodes + 1), all_shots_history[name], label=name)

    plt.title(f'Shots per Episode Comparison ({test_episodes} episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Shots to Complete Game')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, 'shots_comparison.png'))
    plt.close()

    print(f"\nComparison completed for {', '.join(agent_names)}")
    print(f"Results saved to: {results_dir}")

    # Print statistics
    print("\nPerformance Statistics:")
    print(stats_df.to_string(index=False))

    return results


def compare_by_grid_size_same_config(agents, grid_sizes=[3, 5, 7, 10], test_episodes=20):
    """
    Compare agents across different grid sizes using the same-configuration approach.

    Args:
        agents: List of agent objects to compare
        grid_sizes: List of grid sizes to test
        test_episodes: Number of episodes per grid size

    Returns:
        Dictionary of results for each grid size
    """
    results_by_size = {}

    # Set up results directory
    results_dir = setup_results_directory("grid_size_same_config", "testing")

    for grid_size in grid_sizes:
        print(f"\nTesting with grid size {grid_size}x{grid_size}")

        # Scale ship configuration with grid size
        ship_count = max(1, grid_size // 2)
        ships_config = []
        for i in range(ship_count):
            ship_size = min(grid_size // 2, i + 2)
            ships_config.append(ship_size)

        print(f"Using ships configuration: {ships_config}")

        # Compare agents with this grid size
        results = compare_agents_same_config(
            agents,
            test_episodes=test_episodes,
            grid_size=grid_size,
            ships_config=ships_config,
            fixed_test=True
        )

        results_by_size[grid_size] = results

    # Generate charts and metrics similar to the original function
    agent_names = [agent.name for agent in agents]

    # Create a plot of average shots by grid size
    plt.figure(figsize=(12, 8))

    for name in agent_names:
        avg_shots = [np.mean(results_by_size[size]['shots_history'][name])
                     for size in grid_sizes]
        plt.plot(grid_sizes, avg_shots, marker='o', label=name)

    plt.title('Performance Across Grid Sizes')
    plt.xlabel('Grid Size')
    plt.ylabel('Average Shots to Complete Game')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, 'grid_size_performance.png'))
    plt.close()

    # Create a plot of race wins by grid size
    plt.figure(figsize=(12, 8))

    for name in agent_names:
        race_win_pct = [results_by_size[size]['race_wins'][name] / results_by_size[size]['test_episodes']
                        for size in grid_sizes]
        plt.plot(grid_sizes, race_win_pct, marker='o', label=name)

    plt.title('Race Win Percentage Across Grid Sizes')
    plt.xlabel('Grid Size')
    plt.ylabel('Race Win Percentage')
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, 'grid_size_race_wins.png'))
    plt.close()

    # Save summary data as CSV
    summary_data = []

    for size in grid_sizes:
        for name in agent_names:
            avg_shots = np.mean(results_by_size[size]['shots_history'][name])
            win_rate = results_by_size[size]['win_rates'][name]
            race_wins = results_by_size[size]['race_wins'][name]
            race_win_pct = race_wins / results_by_size[size]['test_episodes']

            summary_data.append({
                'Grid Size': size,
                'Agent': name,
                'Avg Shots': avg_shots,
                'Win Rate': win_rate,
                'Race Wins': race_wins,
                'Race Win %': race_win_pct
            })

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(results_dir, 'grid_size_summary.csv'), index=False)

    print(f"\nGrid size comparison completed.")
    print(f"Results saved to: {results_dir}")

    return results_by_size


def compare_by_ship_config_same_config(agents, base_grid_size=5, ship_configs=None, test_episodes=20):
    """
    Compare agents across different ship configurations using the same-configuration approach.

    Args:
        agents: List of agent objects to compare
        base_grid_size: Base grid size to use
        ship_configs: List of ship configurations to test
        test_episodes: Number of episodes per configuration

    Returns:
        Dictionary of results for each ship configuration
    """
    if ship_configs is None:
        # Default ship configurations to test
        ship_configs = [
            [2],  # One small ship
            [3],  # One medium ship
            [2, 2],  # Two small ships
            [3, 2],  # Medium and small ships (default)
            [3, 3],  # Two medium ships
            [4, 2],  # Large and small ships
            [3, 2, 2]  # Medium and two small ships
        ]

    results_by_config = {}

    # Set up results directory
    results_dir = setup_results_directory("ship_config_same_config", "testing")

    for config_idx, ships_config in enumerate(ship_configs):
        config_name = f"Config_{config_idx + 1}_{'-'.join(map(str, ships_config))}"
        print(f"\nTesting with ship configuration: {ships_config}")

        # Compare agents with this ship configuration
        results = compare_agents_same_config(
            agents,
            test_episodes=test_episodes,
            grid_size=base_grid_size,
            ships_config=ships_config,
            fixed_test=True
        )

        results_by_config[config_name] = results

    # Generate charts and metrics similar to the original function
    agent_names = [agent.name for agent in agents]
    config_names = list(results_by_config.keys())

    # Create a plot of average shots by ship configuration
    plt.figure(figsize=(14, 8))

    x = np.arange(len(config_names))
    width = 0.8 / len(agent_names)

    for i, name in enumerate(agent_names):
        avg_shots = [np.mean(results_by_config[config]['shots_history'][name])
                     for config in config_names]

        plt.bar(x + i * width - 0.4 + width / 2, avg_shots, width, label=name)

    plt.title('Performance Across Ship Configurations')
    plt.xlabel('Ship Configuration')
    plt.ylabel('Average Shots to Complete Game')
    plt.xticks(x, [c.split('_', 2)[2] for c in config_names], rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.grid(True, axis='y')
    plt.savefig(os.path.join(results_dir, 'ship_config_performance.png'))
    plt.close()

    # Create a plot of race win percentage by ship configuration
    plt.figure(figsize=(14, 8))

    for i, name in enumerate(agent_names):
        race_win_pct = [results_by_config[config]['race_wins'][name] / results_by_config[config]['test_episodes']
                        for config in config_names]

        plt.bar(x + i * width - 0.4 + width / 2, race_win_pct, width, label=name)

    plt.title('Race Win Percentage Across Ship Configurations')
    plt.xlabel('Ship Configuration')
    plt.ylabel('Race Win Percentage')
    plt.ylim(0, 1.1)
    plt.xticks(x, [c.split('_', 2)[2] for c in config_names], rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.grid(True, axis='y')
    plt.savefig(os.path.join(results_dir, 'ship_config_race_wins.png'))
    plt.close()

    # Save summary data as CSV
    summary_data = []

    for config in config_names:
        for name in agent_names:
            avg_shots = np.mean(results_by_config[config]['shots_history'][name])
            win_rate = results_by_config[config]['win_rates'][name]
            race_wins = results_by_config[config]['race_wins'][name]
            race_win_pct = race_wins / results_by_config[config]['test_episodes']

            summary_data.append({
                'Ship Config': config.split('_', 2)[2],
                'Agent': name,
                'Avg Shots': avg_shots,
                'Win Rate': win_rate,
                'Race Wins': race_wins,
                'Race Win %': race_win_pct
            })

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(results_dir, 'ship_config_summary.csv'), index=False)

    print(f"\nShip configuration comparison completed.")
    print(f"Results saved to: {results_dir}")

    return results_by_config


if __name__ == "__main__":
    # Example usage
    from random_agent import RandomAgent, train_random_agent
    from smart_agent import SmartAgent, train_smart_agent
    from qlearning_agent import QLearningAgent, train_qlearning_agent
    from FinalProject.thomson_model import ThomsonSamplingAgent

    print("Training Random Agent...")
    random_agent, _ = train_random_agent(episodes=100)

    print("\nTraining Smart Agent...")
    smart_agent, _ = train_smart_agent(episodes=100)

    print("\nTraining Q-Learning Agent...")
    qlearning_agent, _ = train_qlearning_agent(episodes=500)

    # Create ThomsonSamplingAgent (assuming it doesn't need training)
    thomson_agent = ThomsonSamplingAgent()

    agents = [random_agent, smart_agent, qlearning_agent, thomson_agent]

    # Basic comparison with same configuration
    print("\nPerforming basic comparison with same configuration...")
    comparison_results = compare_agents_same_config(agents, test_episodes=50, fixed_test=True)

    # Grid size comparison with same configuration
    print("\nComparing across different grid sizes with same configuration...")
    grid_results = compare_by_grid_size_same_config(agents, grid_sizes=[3, 5, 7], test_episodes=20)

    # Ship configuration comparison with same configuration
    print("\nComparing across different ship configurations with same configuration...")
    config_results = compare_by_ship_config_same_config(agents, test_episodes=20)
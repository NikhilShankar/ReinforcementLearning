import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import pickle
from datetime import datetime
import seaborn as sns
from battleship_env import BattleshipEnv, create_test_scenario, setup_results_directory


class AutomatedTournament:
    """
    Non-UI version of the battleship tournament that runs games automatically and generates
    comprehensive statistics about agent performance and ship distance analysis.
    """

    def __init__(self, agents, grid_size=5, ships_config=None, results_dir=None):
        """
        Initialize the automated tournament.

        Args:
            agents: List of agent objects to compare
            grid_size: Size of the game grid
            ships_config: Configuration of ships [ship1_size, ship2_size, ...]
            results_dir: Directory to save results (if None, a new one will be created)
        """
        self.agents = agents
        self.agent_names = [agent.name for agent in agents]
        self.grid_size = grid_size

        if ships_config is None:
            self.ships_config = [3, 2]  # Default ship configuration for 5x5 grid
        else:
            self.ships_config = ships_config

        if results_dir is None:
            self.results_dir = setup_results_directory("automated", "tournament")
        else:
            self.results_dir = results_dir
            os.makedirs(self.results_dir, exist_ok=True)

        # Initialize metrics tracking
        self.shots_history = {name: [] for name in self.agent_names}
        self.wins = {name: 0 for name in self.agent_names}
        self.losses = {name: 0 for name in self.agent_names}

        # Ship distance tracking
        # For each agent, track performance based on ship distances
        # Format: {agent_name: {distance_value: {'wins': count, 'losses': count, 'total_games': count}}}
        self.ship_distance_stats = {name: {} for name in self.agent_names}

    def run_tournament(self, episodes=100, verbose=True):
        """
        Run an automated tournament between all agents for a specified number of games.

        Args:
            episodes: Number of episodes (games) to run
            verbose: Whether to print progress updates

        Returns:
            Dictionary containing tournament results
        """
        for episode in range(episodes):
            # Create a random board configuration
            base_env = BattleshipEnv(grid_size=self.grid_size, ships_config=self.ships_config)
            ship_grid = base_env.ship_grid.copy()

            # Extract ship positions and calculate distance between ships
            ship_positions = self._get_ship_positions(ship_grid)
            ship_distance = self._calculate_ship_distance(ship_positions)

            # For debugging: Print the ship grid and calculated distance
            if verbose and episode < 5:  # Only for first few episodes
                print(f"Episode {episode + 1} Ship Grid:")
                for row in ship_grid:
                    print(" ".join(str(cell) for cell in row))
                print(f"Calculated distance between ships: {ship_distance}")
                print("-" * 30)

            # Track shots for each agent on this episode
            episode_shots = {}

            # Each agent plays on the same board configuration
            for agent in self.agents:
                # Reset the agent
                agent.reset()

                # Create a new environment with the same ship configuration
                env = BattleshipEnv(grid_size=self.grid_size, ships_config=self.ships_config)
                env.ship_grid = ship_grid.copy()
                env.reset()
                env.ship_grid = ship_grid.copy()  # Ensure ship positions are preserved

                # Run the agent on this environment
                observation = env.grid.copy()
                done = False
                agent_shots = 0

                while not done:
                    action = agent.act(observation)
                    next_observation, reward, done, info = env.step(action)

                    # If the agent supports update method, use it
                    if hasattr(agent, 'update') and callable(getattr(agent, 'update')):
                        agent.update(observation, action, reward, next_observation, done, info)

                    observation = next_observation
                    agent_shots += 1

                    # Prevent infinite loops
                    if agent_shots > self.grid_size * self.grid_size:
                        break

                # Record metrics for this agent
                self.shots_history[agent.name].append(agent_shots)
                episode_shots[agent.name] = agent_shots

            # Determine the winner for this episode (agent with fewest shots)
            winner = min(episode_shots, key=episode_shots.get)
            self.wins[winner] += 1

            # Record losses for others
            for agent_name in self.agent_names:
                if agent_name != winner:
                    self.losses[agent_name] += 1

            # Update ship distance statistics - ensure the distance key is a string
            for agent_name in self.agent_names:
                # Convert distance to string to ensure consistent dictionary keys
                dist_key = str(ship_distance)

                # Initialize distance entry if it doesn't exist
                if dist_key not in self.ship_distance_stats[agent_name]:
                    self.ship_distance_stats[agent_name][dist_key] = {
                        'wins': 0,
                        'losses': 0,
                        'total_games': 0
                    }

                # Update stats for this distance
                self.ship_distance_stats[agent_name][dist_key]['total_games'] += 1

                if agent_name == winner:
                    self.ship_distance_stats[agent_name][dist_key]['wins'] += 1
                else:
                    self.ship_distance_stats[agent_name][dist_key]['losses'] += 1

            # Print progress
            if verbose and (episode + 1) % 10 == 0:
                print(f"Tournament episode {episode + 1}/{episodes} completed.")

        # Calculate final statistics
        win_percentages = {name: (self.wins[name] / episodes) * 100 for name in self.agent_names}
        loss_percentages = {name: (self.losses[name] / episodes) * 100 for name in self.agent_names}
        avg_shots = {name: np.mean(self.shots_history[name]) for name in self.agent_names}

        # Generate all metrics and visualizations
        if verbose:
            print("Generating metrics and visualizations...")
        self.generate_reports(episodes)

        return {
            'wins': self.wins,
            'losses': self.losses,
            'win_percentages': win_percentages,
            'loss_percentages': loss_percentages,
            'avg_shots': avg_shots,
            'shots_history': self.shots_history,
            'ship_distance_stats': self.ship_distance_stats
        }

    def _get_ship_positions(self, ship_grid):
        """
        Extract positions of all ships from the grid.

        Args:
            ship_grid: 2D grid with ship IDs

        Returns:
            List of lists, where each inner list contains position tuples for a ship
        """
        # Create a list to hold positions for each ship
        ship_positions = []

        # For each ship ID (1-based indexing in the grid)
        for ship_id in range(2, len(self.ships_config) + 2):
            positions = []
            # Iterate through the 2D grid
            for i in range(len(ship_grid)):
                for j in range(len(ship_grid[i])):
                    if ship_grid[i][j] == ship_id:
                        # Store as 2D coordinates (row, col)
                        positions.append((i, j))

            # Sort positions to ensure consistency
            positions.sort()
            ship_positions.append(positions)

        return ship_positions

    def _calculate_ship_distance(self, ship_positions):
        """
        Calculate the minimum Manhattan distance between ships.

        Args:
            ship_positions: List of lists, where each inner list contains position tuples for a ship

        Returns:
            int: Number of empty cells between the closest points of the two ships
        """
        # Check if we have at least two ships with valid positions
        if len(ship_positions) < 2:
            return 0
        print(f"Cells ship positions: {ship_positions}")

        # Get cells for first two ships
        ship1_cells = ship_positions[0]
        ship2_cells = ship_positions[1]
        print(f"Cells ship1 {ship1_cells}, ship2 {ship2_cells}")

        if not ship1_cells or not ship2_cells:
            return 0

        # Find minimum Manhattan distance between any cell of ship1 and any cell of ship2
        min_manhattan = float('inf')

        for pos1 in ship1_cells:
            row1, col1 = pos1
            for pos2 in ship2_cells:
                row2, col2 = pos2

                # Manhattan distance calculation: |row2-row1| + |col2-col1|
                manhattan_dist = abs(row2 - row1) + abs(col2 - col1)

                if manhattan_dist < min_manhattan:
                    min_manhattan = manhattan_dist

        # Convert Manhattan distance to "empty cells between ships"
        # Manhattan distance of 1 means ships are adjacent (0 cells between)
        cells_between = max(0, min_manhattan - 1)
        print(f"Cells between: {cells_between}")
        return cells_between

    def generate_reports(self, episodes):
        """Generate all metrics reports and visualizations."""
        # Generate win/loss reports
        self._generate_win_loss_report(episodes)

        # Generate shots analysis
        self._generate_shots_analysis()

        # Generate ship distance analysis
        self._generate_ship_distance_analysis()

    def _generate_win_loss_report(self, episodes):
        """Generate win/loss statistics report."""
        # Create dataframe for overall stats
        stats_data = {
            'Agent': self.agent_names,
            'Wins': [self.wins[name] for name in self.agent_names],
            'Losses': [self.losses[name] for name in self.agent_names],
            'Win Percentage': [(self.wins[name] / episodes) * 100 for name in self.agent_names],
            'Loss Percentage': [(self.losses[name] / episodes) * 100 for name in self.agent_names]
        }

        stats_df = pd.DataFrame(stats_data)
        stats_df.to_csv(os.path.join(self.results_dir, 'win_loss_stats.csv'), index=False)

        # Create win/loss visualization
        plt.figure(figsize=(12, 6))

        x = range(len(self.agent_names))
        width = 0.35

        win_bars = plt.bar([i - width / 2 for i in x],
                           [self.wins[name] for name in self.agent_names],
                           width,
                           label='Wins')

        loss_bars = plt.bar([i + width / 2 for i in x],
                            [self.losses[name] for name in self.agent_names],
                            width,
                            label='Losses')

        plt.xlabel('Agent')
        plt.ylabel('Number of Games')
        plt.title('Wins and Losses by Agent')
        plt.xticks(x, self.agent_names)
        plt.legend()

        # Add value labels
        for bar in win_bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                     f'{height}', ha='center', va='bottom')

        for bar in loss_bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                     f'{height}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'win_loss_comparison.png'))
        plt.close()

    def _generate_shots_analysis(self):
        """Generate shots analysis report and visualizations."""
        # Calculate shot statistics
        shot_stats = {}
        for name in self.agent_names:
            shots = self.shots_history[name]
            shot_stats[name] = {
                'avg': np.mean(shots),
                'min': np.min(shots),
                'max': np.max(shots),
                'median': np.median(shots),
                'std': np.std(shots)
            }

        # Create dataframe
        shot_rows = []
        for name, stats in shot_stats.items():
            shot_rows.append({
                'Agent': name,
                'Avg Shots': stats['avg'],
                'Min Shots': stats['min'],
                'Max Shots': stats['max'],
                'Median Shots': stats['median'],
                'Std Dev': stats['std']
            })

        shots_df = pd.DataFrame(shot_rows)
        shots_df.to_csv(os.path.join(self.results_dir, 'shot_statistics.csv'), index=False)

        # Create average shots comparison visualization
        plt.figure(figsize=(10, 6))

        bars = plt.bar(range(len(self.agent_names)),
                       [shot_stats[name]['avg'] for name in self.agent_names])

        plt.xlabel('Agent')
        plt.ylabel('Average Shots per Game')
        plt.title('Average Shots Comparison')
        plt.xticks(range(len(self.agent_names)), self.agent_names)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                     f'{height:.2f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'avg_shots_comparison.png'))
        plt.close()

        # Create box plot for shot distribution
        plt.figure(figsize=(12, 8))

        plt.boxplot([self.shots_history[name] for name in self.agent_names],
                    labels=self.agent_names)

        plt.title('Shots Distribution Comparison')
        plt.xlabel('Agent')
        plt.ylabel('Shots to Complete Game')
        plt.grid(True, axis='y')

        # Add average shots text
        for i, name in enumerate(self.agent_names):
            avg = shot_stats[name]['avg']
            plt.text(i + 1, avg, f'Avg: {avg:.2f}',
                     ha='center', va='center',
                     bbox=dict(facecolor='white', alpha=0.8))

        plt.savefig(os.path.join(self.results_dir, 'shots_distribution.png'))
        plt.close()

    def _generate_ship_distance_analysis(self):
        """Generate analysis of ship distance impact on win/loss rates."""
        # Create directory for distance analysis
        distance_dir = os.path.join(self.results_dir, 'distance_analysis')
        os.makedirs(distance_dir, exist_ok=True)

        # For each agent, create analysis of win rates based on ship distance
        win_rate_by_distance = {}
        for agent_name in self.agent_names:
            distance_stats = self.ship_distance_stats[agent_name]

            # Calculate win rate for each distance
            win_rates = []
            distances = []
            games_counts = []

            # Convert string keys back to floats for sorting
            for distance_str, stats in sorted(distance_stats.items(), key=lambda x: float(x[0])):
                total_games = stats['total_games']
                if total_games > 0:
                    win_rate = (stats['wins'] / total_games) * 100
                    win_rates.append(win_rate)
                    distances.append(float(distance_str))  # Convert back to float for plotting
                    games_counts.append(total_games)

            win_rate_by_distance[agent_name] = {
                'distances': distances,
                'win_rates': win_rates,
                'games_counts': games_counts
            }

            # Save the data to CSV
            df = pd.DataFrame({
                'Ship Distance': distances,
                'Win Rate (%)': win_rates,
                'Games Played': games_counts
            })
            df.to_csv(os.path.join(distance_dir, f'{agent_name}_distance_win_rates.csv'), index=False)

        # Create plots for each agent
        for agent_name, data in win_rate_by_distance.items():
            if not data['distances']:  # Skip if no data
                continue

            # Create win rate vs distance plot
            plt.figure(figsize=(12, 6))

            # Use scatter plot with point size based on game count
            sizes = [max(count * 2, 10) for count in data['games_counts']]  # Scale up the sizes, min size 10
            sc = plt.scatter(data['distances'], data['win_rates'], s=sizes, alpha=0.6)

            # Add trendline
            if len(data['distances']) > 1:
                z = np.polyfit(data['distances'], data['win_rates'], 1)
                p = np.poly1d(z)

                # Create smooth line for the trendline
                x_trend = np.linspace(min(data['distances']), max(data['distances']), 100)
                plt.plot(x_trend, p(x_trend), "r--", alpha=0.8)

                # Add correlation coefficient
                correlation = np.corrcoef(data['distances'], data['win_rates'])[0, 1]
                plt.text(0.05, 0.95, f"Correlation: {correlation:.2f}",
                         transform=plt.gca().transAxes, fontsize=12,
                         verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.1))

            plt.title(f'{agent_name} - Win Rate by Ship Distance')
            plt.xlabel('Distance Between Ships')
            plt.ylabel('Win Rate (%)')
            plt.grid(True, alpha=0.3)

            # Add a note that point size represents number of games
            plt.text(0.05, 0.05, "Point size indicates number of games",
                     transform=plt.gca().transAxes, fontsize=10,
                     verticalalignment='bottom', bbox=dict(boxstyle='round', alpha=0.1))

            plt.savefig(os.path.join(distance_dir, f'{agent_name}_win_rate_by_distance.png'))
            plt.close()

        # Create comparative plot with all agents
        plt.figure(figsize=(14, 8))
        for agent_name, data in win_rate_by_distance.items():
            if not data['distances']:  # Skip if no data
                continue
            plt.plot(data['distances'], data['win_rates'], 'o-', label=agent_name, alpha=0.7)

        plt.title('Comparative Win Rates by Ship Distance')
        plt.xlabel('Distance Between Ships')
        plt.ylabel('Win Rate (%)')
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.savefig(os.path.join(distance_dir, 'comparative_win_rates.png'))
        plt.close()

        # Create heatmap of distance vs win rate for all agents
        # Only proceed if we have distance data
        all_distances = []
        for agent_data in win_rate_by_distance.values():
            all_distances.extend(agent_data['distances'])

        if all_distances:  # Only proceed if we have distances to analyze
            # Get unique distances, sorted
            unique_distances = sorted(set(all_distances))

            # Create a dataframe for the heatmap
            heatmap_data = []
            for agent_name in self.agent_names:
                if agent_name not in win_rate_by_distance:
                    continue

                agent_data = win_rate_by_distance[agent_name]
                agent_distances = agent_data['distances']
                agent_win_rates = agent_data['win_rates']

                # Create a row for each agent
                row_data = {}
                row_data['Agent'] = agent_name

                # Add win rate for each distance
                for dist in unique_distances:
                    if dist in agent_distances:
                        idx = agent_distances.index(dist)
                        row_data[f'Dist_{dist}'] = agent_win_rates[idx]
                    else:
                        row_data[f'Dist_{dist}'] = np.nan

                heatmap_data.append(row_data)

            # Create dataframe for the heatmap
            if heatmap_data:
                heatmap_df = pd.DataFrame(heatmap_data)
                heatmap_df.set_index('Agent', inplace=True)

                # Create the heatmap
                plt.figure(figsize=(len(unique_distances) * 1.2 + 2, len(self.agent_names) * 0.8 + 2))

                # Format the data for the heatmap
                data_columns = [f'Dist_{dist}' for dist in unique_distances]

                # Only create heatmap if we have data
                if not heatmap_df.empty and len(data_columns) > 0:
                    ax = sns.heatmap(heatmap_df[data_columns],
                                     annot=True,
                                     fmt=".1f",
                                     cmap="RdYlGn",
                                     cbar_kws={'label': 'Win Rate (%)'},
                                     linewidths=0.5)

                    # Update x-axis labels to show actual distances
                    ax.set_xticklabels([str(dist) for dist in unique_distances])

                    plt.title('Win Rate by Ship Distance - All Agents')
                    plt.xlabel('Distance Between Ships')
                    plt.ylabel('Agent')

                    plt.tight_layout()
                    plt.savefig(os.path.join(distance_dir, 'distance_win_rate_heatmap.png'))
                    plt.close()

    @staticmethod
    def load_agent_from_file(filepath, agent_class):
        """
        Load an agent from a file using the provided agent class.

        Args:
            filepath: Path to the agent file
            agent_class: The class of the agent to be loaded

        Returns:
            The loaded agent
        """
        agent = agent_class()
        agent.load(filepath)
        return agent


def run_automated_tournament(agents, num_games=1000, grid_size=5, ships_config=None, results_dir=None):
    """
    Run an automated tournament without UI and generate comprehensive reports.

    Args:
        agents: List of pre-trained agent objects to compare
        num_games: Number of games to run
        grid_size: Size of the game grid
        ships_config: Configuration of ships [ship1_size, ship2_size, ...]
        results_dir: Directory to save results (if None, a new one will be created)

    Returns:
        tournament: Tournament object with results
        results: Dictionary of tournament results
    """
    print(f"Starting automated tournament for {num_games} games...")

    # Create tournament with provided agents
    tournament = AutomatedTournament(agents, grid_size=grid_size, ships_config=ships_config, results_dir=results_dir)

    # Run the tournament
    print(f"Running {num_games} games...")
    results = tournament.run_tournament(episodes=num_games)

    # Print summary statistics
    print("\nTournament Results:")
    print(f"Total games: {num_games}")

    print("\nWins:")
    for name, wins in results['wins'].items():
        win_pct = results['win_percentages'][name]
        print(f"{name}: {wins} wins ({win_pct:.2f}%)")

    print("\nLosses:")
    for name, losses in results['losses'].items():
        loss_pct = results['loss_percentages'][name]
        print(f"{name}: {losses} losses ({loss_pct:.2f}%)")

    print("\nAverage Shots:")
    for name, avg in results['avg_shots'].items():
        print(f"{name}: {avg:.2f} shots on average")

    print(f"\nDetailed reports and visualizations saved to: {tournament.results_dir}")

    return tournament, results


if __name__ == "__main__":
    # This is just an example of how to use the tournament
    # You would replace this with your actual agents

    # Import your agents (these are just placeholders)
    # from your_agent_module import YourAgentClass
    from random_agent import RandomAgent
    from smart_agent import SmartAgent

    # Create instances of your pre-trained agents
    agent1 = RandomAgent()  # Replace with your actual agent instances
    agent2 = SmartAgent()  # Replace with your actual agent instances

    # Set parameters for the automated tournament
    num_games = 1000  # Number of games to play
    grid_size = 5  # Size of the grid
    ships_config = [3, 2]  # Ship sizes

    # Run the tournament with your agents
    tournament, results = run_automated_tournament(
        agents=[agent1, agent2],  # Pass your pre-trained agents directly
        num_games=num_games,
        grid_size=grid_size,
        ships_config=ships_config
    )
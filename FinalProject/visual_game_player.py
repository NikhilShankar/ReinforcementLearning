import numpy as np
import matplotlib.pyplot as plt
import time
import os
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
from battleship_env import BattleshipEnv, create_test_scenario



class VisualGamePlayer:
    """
    A class for visualizing a step-by-step game of Battleship between two agents.
    Displays the game state after each move and waits for user input to continue.
    """

    def __init__(self, agent1, agent2, agent1_name=None, agent2_name=None,
                 grid_size=5, ships_config=None, use_gui=True):
        """
        Initialize the visual game player.

        Args:
            agent1: First agent
            agent2: Second agent
            agent1_name: Custom name for agent1 (optional)
            agent2_name: Custom name for agent2 (optional)
            grid_size: Size of the game grid
            ships_config: Configuration of ships [ship1_size, ship2_size, ...]
            use_gui: Whether to use a Tkinter GUI window (True) or command line interface (False)
        """
        self.agent1 = agent1
        self.agent2 = agent2

        # Use custom names if provided, otherwise use agent object names
        self.agent1_name = agent1_name if agent1_name is not None else agent1.name
        self.agent2_name = agent2_name if agent2_name is not None else agent2.name

        self.grid_size = grid_size

        if ships_config is None:
            self.ships_config = [3, 2]  # Default ship configuration for 5x5 grid
        else:
            self.ships_config = ships_config

        self.use_gui = use_gui
        self.current_step = 0

        # If using GUI, initialize the window
        if use_gui:
            self.root = None
            self.canvas1 = None
            self.canvas2 = None
            self.step_button = None
            self.auto_play_button = None
            self.reset_button = None
            self.status_label = None
            self.is_auto_playing = False
            self.auto_play_speed = 1.0  # seconds between moves

    def _initialize_gui(self):
        """Initialize the Tkinter GUI window."""
        self.root = tk.Tk()
        self.root.title("Battleship Agent Visualization")
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Create frames for the two agents
        agent1_frame = ttk.LabelFrame(main_frame, text=self.agent1_name, padding="5")
        agent1_frame.grid(row=0, column=0, padx=10, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))

        agent2_frame = ttk.LabelFrame(main_frame, text=self.agent2_name, padding="5")
        agent2_frame.grid(row=0, column=1, padx=10, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Create canvases for the plots
        self.fig1, self.ax1 = plt.subplots(figsize=(6, 5))
        self.canvas1 = FigureCanvasTkAgg(self.fig1, master=agent1_frame)
        self.canvas1.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.fig2, self.ax2 = plt.subplots(figsize=(6, 5))
        self.canvas2 = FigureCanvasTkAgg(self.fig2, master=agent2_frame)
        self.canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Control panel
        control_frame = ttk.Frame(main_frame, padding="5")
        control_frame.grid(row=1, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E))

        # Status label
        self.status_label = ttk.Label(control_frame, text="Ready to start. Press 'Next Step' to begin.")
        self.status_label.grid(row=0, column=0, columnspan=3, pady=5)

        # Step button
        self.step_button = ttk.Button(control_frame, text="Next Step", command=self.step)
        self.step_button.grid(row=1, column=0, padx=5)

        # Auto-play button
        self.auto_play_button = ttk.Button(control_frame, text="Auto Play", command=self.toggle_auto_play)
        self.auto_play_button.grid(row=1, column=1, padx=5)

        # Reset button
        self.reset_button = ttk.Button(control_frame, text="Reset Game", command=self.reset)
        self.reset_button.grid(row=1, column=2, padx=5)

        # Speed control
        speed_frame = ttk.Frame(control_frame)
        speed_frame.grid(row=2, column=0, columnspan=3, pady=5)

        ttk.Label(speed_frame, text="Auto-play speed:").grid(row=0, column=0, padx=5)

        self.speed_var = tk.DoubleVar(value=1.0)
        speed_scale = ttk.Scale(speed_frame, from_=0.1, to=3.0, orient=tk.HORIZONTAL,
                                variable=self.speed_var, length=200,
                                command=self._update_speed)
        speed_scale.grid(row=0, column=1, padx=5)

        self.speed_label = ttk.Label(speed_frame, text="1.0s")
        self.speed_label.grid(row=0, column=2, padx=5)

        # Add keyboard bindings
        self.root.bind('<space>', lambda event: self.step())
        self.root.bind('<Return>', lambda event: self.step())

        # Configure grid weights for resizing
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)

        # Set minimum size
        self.root.minsize(800, 600)

    def _update_speed(self, value):
        """Update the auto-play speed."""
        self.auto_play_speed = float(value)
        self.speed_label.config(text=f"{self.auto_play_speed:.1f}s")

    def _on_closing(self):
        """Handle window closing event."""
        self.is_auto_playing = False
        plt.close(self.fig1)
        plt.close(self.fig2)
        self.root.destroy()

    def toggle_auto_play(self):
        """Toggle auto-play mode."""
        self.is_auto_playing = not self.is_auto_playing

        if self.is_auto_playing:
            self.auto_play_button.config(text="Stop")
            self.root.after(int(self.auto_play_speed * 1000), self._auto_play_step)
        else:
            self.auto_play_button.config(text="Auto Play")

    def _auto_play_step(self):
        """Take a step in auto-play mode."""
        if not self.is_auto_playing:
            return

        if not self.game_over1 or not self.game_over2:
            self.step()
            self.root.after(int(self.auto_play_speed * 1000), self._auto_play_step)
        else:
            self.is_auto_playing = False
            self.auto_play_button.config(text="Auto Play")

    def reset(self):
        """Reset the game with a new random ship configuration."""
        self.current_step = 0
        self.initialize_game(new_ships=True)  # Pass flag to create new ship configuration
        self.update_plots()
        self.status_label.config(text="Game reset with new ship positions. Press 'Next Step' to begin.")
        self.step_button.config(state=tk.NORMAL)

    def initialize_game(self, new_ships=False):
        """Initialize a new game with the same ship configuration for both agents."""
        if new_ships:
            # Create a new random ship configuration
            base_env = BattleshipEnv(grid_size=self.grid_size, ships_config=self.ships_config)
            ship_grid = base_env.ship_grid.copy()
        else:
            # Use the standard test scenario with fixed ship positions
            base_env = create_test_scenario(grid_size=self.grid_size, ships_config=self.ships_config)
            ship_grid = base_env.ship_grid.copy()

        # Reset agents
        self.agent1.reset()
        self.agent2.reset()

        # Create separate environments with identical ship configurations
        self.env1 = BattleshipEnv(grid_size=self.grid_size, ships_config=self.ships_config)
        self.env1.ship_grid = ship_grid.copy()
        self.env1.reset()
        self.env1.ship_grid = ship_grid.copy()  # Make sure ship positions are preserved

        self.env2 = BattleshipEnv(grid_size=self.grid_size, ships_config=self.ships_config)
        self.env2.ship_grid = ship_grid.copy()
        self.env2.reset()
        self.env2.ship_grid = ship_grid.copy()  # Make sure ship positions are preserved

        # Initialize game state
        self.observation1 = self.env1.grid.copy()
        self.observation2 = self.env2.grid.copy()
        self.game_over1 = False
        self.game_over2 = False
        self.shots1 = 0
        self.shots2 = 0
        self.current_step = 0

    def _update_render_plot(self, ax, grid, ship_grid, title, show_ships=True):
        """Update a single agent's plot."""
        ax.clear()

        # Use pcolormesh for proper grid alignment
        x = np.arange(0, self.grid_size + 1)
        y = np.arange(0, self.grid_size + 1)
        X, Y = np.meshgrid(x, y)

        # Player's view
        masked_grid = np.ma.masked_array(grid, grid < 0)

        if show_ships:
            # Create a blended view that shows both ships and shots
            blended_view = grid.copy().astype(float)

            # Mark ship positions with light gray if not hit
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    if ship_grid[i, j] > 0:
                        if grid[i, j] != 2:  # If not hit yet
                            blended_view[i, j] = 0.5  # Light indication of ship

            pcm = ax.pcolormesh(X, Y, blended_view, cmap='Blues_r', vmin=0, vmax=2,
                                edgecolors='k', linewidth=1)
        else:
            # Just show the player's view
            pcm = ax.pcolormesh(X, Y, masked_grid, cmap='Blues', vmin=0, vmax=2,
                                edgecolors='k', linewidth=1)

        # Add labels for hits and misses
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if grid[i, j] == 1:
                    ax.text(j + 0.5, i + 0.5, 'X', ha='center', va='center', color='black')
                elif grid[i, j] == 2:
                    ax.text(j + 0.5, i + 0.5, 'H', ha='center', va='center', color='red', weight='bold')

        # Set limits and labels
        ax.set_xlim(0, self.grid_size)
        ax.set_ylim(self.grid_size, 0)  # Invert y-axis for traditional grid view
        ax.set_title(title)
        ax.set_xticks(np.arange(0.5, self.grid_size + 0.5))
        ax.set_yticks(np.arange(0.5, self.grid_size + 0.5))
        ax.set_xticklabels(np.arange(self.grid_size))
        ax.set_yticklabels(np.arange(self.grid_size))
        ax.set_aspect('equal')

    def update_plots(self):
        """Update both agent plots."""
        if self.use_gui:
            # Agent 1 plot
            self._update_render_plot(
                self.ax1, self.env1.grid, self.env1.ship_grid,
                f"{self.agent1_name} - Shots: {self.shots1}"
            )

            # Agent 2 plot
            self._update_render_plot(
                self.ax2, self.env2.grid, self.env2.ship_grid,
                f"{self.agent2_name} - Shots: {self.shots2}"
            )

            # Update canvases
            self.canvas1.draw()
            self.canvas2.draw()
        else:
            # Command line visualization
            plt.figure(figsize=(12, 5))

            # Agent 1 plot
            plt.subplot(1, 2, 1)
            # Create a custom colormap for visualization
            custom_cmap = plt.cm.Blues.copy()

            # Use pcolormesh for proper grid alignment
            x = np.arange(0, self.grid_size + 1)
            y = np.arange(0, self.grid_size + 1)
            X, Y = np.meshgrid(x, y)

            # Player's view
            masked_grid = np.ma.masked_array(self.env1.grid, self.env1.grid < 0)
            plt.pcolormesh(X, Y, masked_grid, cmap=custom_cmap, vmin=0, vmax=2,
                           edgecolors='k', linewidth=1)

            # Add labels for hits and misses
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    if self.env1.grid[i, j] == 1:
                        plt.text(j + 0.5, i + 0.5, 'X', ha='center', va='center', color='black')
                    elif self.env1.grid[i, j] == 2:
                        plt.text(j + 0.5, i + 0.5, 'H', ha='center', va='center', color='red', weight='bold')

            plt.title(f"{self.agent1_name} - Shots: {self.shots1}")
            plt.xlim(0, self.grid_size)
            plt.ylim(self.grid_size, 0)  # Invert y-axis for traditional grid view
            plt.xticks(np.arange(0.5, self.grid_size + 0.5), np.arange(self.grid_size))
            plt.yticks(np.arange(0.5, self.grid_size + 0.5), np.arange(self.grid_size))

            # Agent 2 plot
            plt.subplot(1, 2, 2)
            plt.pcolormesh(X, Y, np.ma.masked_array(self.env2.grid, self.env2.grid < 0),
                           cmap=custom_cmap, vmin=0, vmax=2, edgecolors='k', linewidth=1)

            # Add labels for hits and misses
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    if self.env2.grid[i, j] == 1:
                        plt.text(j + 0.5, i + 0.5, 'X', ha='center', va='center', color='black')
                    elif self.env2.grid[i, j] == 2:
                        plt.text(j + 0.5, i + 0.5, 'H', ha='center', va='center', color='red', weight='bold')

            plt.title(f"{self.agent2_name} - Shots: {self.shots2}")
            plt.xlim(0, self.grid_size)
            plt.ylim(self.grid_size, 0)  # Invert y-axis for traditional grid view
            plt.xticks(np.arange(0.5, self.grid_size + 0.5), np.arange(self.grid_size))
            plt.yticks(np.arange(0.5, self.grid_size + 0.5), np.arange(self.grid_size))

            plt.tight_layout()
            plt.show()

    def step(self):
        """Take one step in the game (one move for each agent)."""
        self.current_step += 1

        # Agent 1's turn
        if not self.game_over1:
            action1 = self.agent1.act(self.observation1)
            row1, col1 = divmod(action1, self.grid_size)
            next_observation1, reward1, done1, info1 = self.env1.step(action1)

            # If the agent supports update method, use it
            if hasattr(self.agent1, 'update') and callable(getattr(self.agent1, 'update')):
                self.agent1.update(self.observation1, action1, reward1, next_observation1, done1, info1)

            self.observation1 = next_observation1
            self.shots1 += 1
            self.game_over1 = done1

        # Agent 2's turn
        if not self.game_over2:
            action2 = self.agent2.act(self.observation2)
            row2, col2 = divmod(action2, self.grid_size)
            next_observation2, reward2, done2, info2 = self.env2.step(action2)

            # If the agent supports update method, use it
            if hasattr(self.agent2, 'update') and callable(getattr(self.agent2, 'update')):
                self.agent2.update(self.observation2, action2, reward2, next_observation2, done2, info2)

            self.observation2 = next_observation2
            self.shots2 += 1
            self.game_over2 = done2

        # Update visualizations
        self.update_plots()

        # Update status message in GUI mode
        if self.use_gui:
            status_msg = f"Step {self.current_step}: "

            if not self.game_over1:
                row1, col1 = divmod(action1, self.grid_size)
                status_msg += f"{self.agent1_name} fired at ({row1},{col1}) - {info1['message']}. "

            if not self.game_over2:
                row2, col2 = divmod(action2, self.grid_size)
                status_msg += f"{self.agent2_name} fired at ({row2},{col2}) - {info2['message']}."

            if self.game_over1 and self.game_over2:
                if self.shots1 < self.shots2:
                    winner = self.agent1_name
                    shots_diff = self.shots2 - self.shots1
                elif self.shots2 < self.shots1:
                    winner = self.agent2_name
                    shots_diff = self.shots1 - self.shots2
                else:
                    winner = "It's a tie"
                    shots_diff = 0

                status_msg = f"Game over! {winner} wins by {shots_diff} shots!"
                self.step_button.config(state=tk.DISABLED)

            self.status_label.config(text=status_msg)
        else:
            # Command line status display
            if not self.game_over1:
                row1, col1 = divmod(action1, self.grid_size)
                print(f"{self.agent1_name} fired at ({row1},{col1}) - {info1['message']}")

            if not self.game_over2:
                row2, col2 = divmod(action2, self.grid_size)
                print(f"{self.agent2_name} fired at ({row2},{col2}) - {info2['message']}")

            if self.game_over1 and self.game_over2:
                if self.shots1 < self.shots2:
                    winner = self.agent1_name
                    shots_diff = self.shots2 - self.shots1
                elif self.shots2 < self.shots1:
                    winner = self.agent2_name
                    shots_diff = self.shots1 - self.shots2
                else:
                    winner = "It's a tie"
                    shots_diff = 0

                print(f"Game over! {winner} wins by {shots_diff} shots!")
                return False

            # Wait for user input before continuing
            input("Press Enter to continue...")

        return not (self.game_over1 and self.game_over2)

    def play_game(self, random=False):
        """Play a full game of Battleship, stepping through each move."""
        self.initialize_game(new_ships=random)

        if self.use_gui:
            # Initialize GUI window
            self._initialize_gui()
            self.update_plots()

            # Start Tkinter main loop
            self.root.mainloop()
        else:
            # Command line interface
            print(f"Starting game: {self.agent1_name} vs {self.agent2_name}")
            print(f"Both agents will play on the same board configuration.")
            print(f"Press Enter after each step to continue.\n")

            # Show initial state
            self.update_plots()
            input("Press Enter to start...")

            # Step through the game
            continuing = True
            while continuing:
                continuing = self.step()


def example_visual_game():
    """Run an example visual game between two agents."""
    from random_agent import RandomAgent
    from smart_agent import SmartAgent

    # Create agents
    random_agent = RandomAgent(grid_size=5)
    smart_agent = SmartAgent(grid_size=5)

    # Create visual game player with custom names
    player = VisualGamePlayer(
        random_agent, smart_agent,
        agent1_name="Captain Random",
        agent2_name="Admiral Smart",
        use_gui=True  # Set to False for command line interface
    )

    # Play the game
    player.play_game()

    return player


if __name__ == "__main__":
    player = example_visual_game()
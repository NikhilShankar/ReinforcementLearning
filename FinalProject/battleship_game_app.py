# battleship_game_app.py
import os
import csv
import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from model_comparison import load_agent_from_file
from visual_game_player import VisualGamePlayer


class BattleshipGameApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Battleship Game")
        self.root.minsize(900, 700)

        # Initialize data storage paths
        self.models_dir = "final_saved_models"
        self.data_dir = "game_data"
        self.player_info_file = os.path.join(self.data_dir, "player_info.csv")
        self.game_history_file = os.path.join(self.data_dir, "game_history.csv")

        # Ensure data directory exists
        os.makedirs(self.data_dir, exist_ok=True)

        # Initialize player info if it doesn't exist
        self.initialize_player_info()

        # Create frames
        self.create_frames()

        # Load available models
        self.load_player_info()

        # Show selection screen initially
        self.show_selection_screen()

    def initialize_player_info(self):
        """Create player info file if it doesn't exist"""
        if not os.path.exists(self.player_info_file):
            # Create the file with headers
            with open(self.player_info_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['filename', 'player_name', 'algorithm_used', 'description'])

            # Scan model directory and add entries
            if os.path.exists(self.models_dir):
                for filename in os.listdir(self.models_dir):
                    if filename.endswith('.pkl'):
                        # Use filename parts to generate default info
                        name_parts = os.path.splitext(filename)[0].split('_')
                        algorithm = " ".join(name_parts)

                        player_name = f"Captain {algorithm.title()}"
                        description = f"A battleship agent using the {algorithm} algorithm"

                        with open(self.player_info_file, 'a', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow([filename, player_name, algorithm, description])

        # Create game history file if it doesn't exist
        if not os.path.exists(self.game_history_file):
            with open(self.game_history_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['player1_filename', 'player2_filename', 'winner',
                                 'player1_moves', 'player2_moves', 'date'])

    def load_player_info(self):
        """Load player info from CSV file"""
        try:
            self.player_df = pd.read_csv(self.player_info_file)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load player info: {str(e)}")
            self.player_df = pd.DataFrame(columns=['filename', 'player_name', 'algorithm_used', 'description'])

    def create_frames(self):
        """Create main frames for the application"""
        # Main container frame
        self.container = ttk.Frame(self.root, padding=10)
        self.container.pack(fill=tk.BOTH, expand=True)

        # Selection screen frame
        self.selection_frame = ttk.Frame(self.container)

        # Game screen frame (will be created when game starts)
        self.game_frame = None

        # Stats screen frame
        self.stats_frame = ttk.Frame(self.container)

    def show_selection_screen(self):
        """Display the agent selection screen"""
        # Clear existing frames
        for widget in self.container.winfo_children():
            widget.pack_forget()

        self.selection_frame = ttk.Frame(self.container)
        self.selection_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        title_label = ttk.Label(self.selection_frame, text="Battleship Agent Selection",
                                font=("Arial", 18, "bold"))
        title_label.pack(pady=10)

        # Create frames for player selection
        selection_container = ttk.Frame(self.selection_frame)
        selection_container.pack(fill=tk.BOTH, expand=True, pady=10)

        # Player 1 selection
        player1_frame = ttk.LabelFrame(selection_container, text="Select Player 1", padding=10)
        player1_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        # Player 2 selection
        player2_frame = ttk.LabelFrame(selection_container, text="Select Player 2", padding=10)
        player2_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        # Configure columns to expand equally
        selection_container.columnconfigure(0, weight=1)
        selection_container.columnconfigure(1, weight=1)

        # Create agent selection listboxes
        self.create_player_selection(player1_frame, "player1")
        self.create_player_selection(player2_frame, "player2")

        # Button container
        button_frame = ttk.Frame(self.selection_frame)
        button_frame.pack(fill=tk.X, pady=15)

        # Start Game button
        start_button = ttk.Button(button_frame, text="Start Game",
                                  command=self.start_game, width=20)
        start_button.pack(side=tk.LEFT, padx=10)

        # View Stats button
        stats_button = ttk.Button(button_frame, text="View Player Statistics",
                                  command=self.show_stats_screen, width=20)
        stats_button.pack(side=tk.RIGHT, padx=10)

    # Modified create_player_selection method to fix the selection issue

    def create_player_selection(self, parent, player_var_name):
        """Create player selection components"""
        # Create listbox
        listbox_frame = ttk.Frame(parent)
        listbox_frame.pack(fill=tk.BOTH, expand=True)

        # Scrollbar
        scrollbar = ttk.Scrollbar(listbox_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Player listbox
        listbox = tk.Listbox(listbox_frame, height=10,
                             yscrollcommand=scrollbar.set,
                             selectmode=tk.SINGLE,
                             exportselection=0)  # This is the key fix - prevent selection from being exported
        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar.config(command=listbox.yview)

        # Populate listbox with player names
        for i, row in self.player_df.iterrows():
            listbox.insert(tk.END, row['player_name'])

        # Store reference to listbox
        setattr(self, f"{player_var_name}_listbox", listbox)

        # Player info display
        info_frame = ttk.Frame(parent)
        info_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        # Algorithm label
        algorithm_label = ttk.Label(info_frame, text="Algorithm: ")
        algorithm_label.grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)

        algorithm_value = ttk.Label(info_frame, text="")
        algorithm_value.grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)

        # Description label
        desc_label = ttk.Label(info_frame, text="Description: ")
        desc_label.grid(row=1, column=0, sticky=tk.NW, padx=5, pady=2)

        desc_value = ttk.Label(info_frame, text="", wraplength=300)
        desc_value.grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)

        # Store references
        setattr(self, f"{player_var_name}_algorithm", algorithm_value)
        setattr(self, f"{player_var_name}_description", desc_value)

        # Bind selection event
        def on_select(event):
            if listbox.curselection():
                idx = listbox.curselection()[0]
                player_name = listbox.get(idx)
                player_row = self.player_df[self.player_df['player_name'] == player_name].iloc[0]

                algorithm_value.config(text=player_row['algorithm_used'])
                desc_value.config(text=player_row['description'])

        listbox.bind('<<ListboxSelect>>', on_select)

    # Modify the start_game method in battleship_game_app.py to include better path handling and debugging

    def start_game(self):
        """Start the game with selected agents"""
        # Get selected players
        try:
            player1_idx = self.player1_listbox.curselection()[0]
            player2_idx = self.player2_listbox.curselection()[0]
        except IndexError:
            messagebox.showwarning("Selection Required", "Please select both players to start the game")
            return

        player1_name = self.player1_listbox.get(player1_idx)
        player2_name = self.player2_listbox.get(player2_idx)

        # Check if same player is selected twice
        if player1_name == player2_name:
            messagebox.showwarning("Invalid Selection", "Please select two different players")
            return

        # Get filenames from player names
        player1_row = self.player_df[self.player_df['player_name'] == player1_name].iloc[0]
        player2_row = self.player_df[self.player_df['player_name'] == player2_name].iloc[0]

        player1_filename = player1_row['filename']
        player2_filename = player2_row['filename']

        # Load agents with improved error handling
        try:
            # Get absolute paths to agent files
            agent1_path = os.path.abspath(os.path.join(self.models_dir, player1_filename))
            agent2_path = os.path.abspath(os.path.join(self.models_dir, player2_filename))

            # Print debug info
            print(f"Current working directory: {os.getcwd()}")
            print(f"Loading agent 1 from: {agent1_path}")
            print(f"Loading agent 2 from: {agent2_path}")
            print(f"Agent 1 file exists: {os.path.exists(agent1_path)}")
            print(f"Agent 2 file exists: {os.path.exists(agent2_path)}")

            # Check directory contents
            print(f"Contents of {self.models_dir} directory:")
            if os.path.exists(self.models_dir):
                for f in os.listdir(self.models_dir):
                    print(f"  {f}")
            else:
                print(f"  Directory {self.models_dir} does not exist!")

            # Try to load agents
            if not os.path.exists(agent1_path):
                raise FileNotFoundError(f"Agent 1 file not found: {agent1_path}")
            if not os.path.exists(agent2_path):
                raise FileNotFoundError(f"Agent 2 file not found: {agent2_path}")

            agent1 = load_agent_from_file(agent1_path)
            agent2 = load_agent_from_file(agent2_path)

            # Hide selection screen
            self.selection_frame.pack_forget()

            # Create game frame
            self.game_frame = ttk.Frame(self.container)
            self.game_frame.pack(fill=tk.BOTH, expand=True)

            # Create and start the visual game player
            self.game_player = GamePlayerWrapper(
                self.game_frame,
                agent1, agent2,
                player1_name, player2_name,
                player1_filename, player2_filename,
                self.on_game_complete
            )

        except Exception as e:
            # Show detailed error message
            error_msg = f"Failed to start game: {str(e)}\n\n"
            error_msg += f"Player 1: {player1_name} ({player1_filename})\n"
            error_msg += f"Player 2: {player2_name} ({player2_filename})\n\n"
            error_msg += f"Working directory: {os.getcwd()}\n"
            error_msg += f"Models directory: {os.path.abspath(self.models_dir)}"

            messagebox.showerror("Error", error_msg)
            print(error_msg)  # Also print to console for debugging

    def on_game_complete(self, game_result):
        """Handle game completion and save results"""
        # Save game results
        self.save_game_history(game_result)

        # Show results dialog
        self.show_game_results(game_result)

    def save_game_history(self, game_result):
        """Save game results to history file"""
        with open(self.game_history_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                game_result['player1_filename'],
                game_result['player2_filename'],
                game_result['winner_filename'],
                game_result['player1_moves'],
                game_result['player2_moves'],
                pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            ])

    def show_game_results(self, game_result):
        """Show game results dialog"""
        result_window = tk.Toplevel(self.root)
        result_window.title("Game Results")
        result_window.geometry("400x300")
        result_window.transient(self.root)
        result_window.grab_set()

        frame = ttk.Frame(result_window, padding=20)
        frame.pack(fill=tk.BOTH, expand=True)

        # Title
        title = ttk.Label(frame, text="Game Results", font=("Arial", 16, "bold"))
        title.pack(pady=10)

        # Winner
        winner_text = "Winner: " + game_result['winner_name']
        winner_label = ttk.Label(frame, text=winner_text, font=("Arial", 12))
        winner_label.pack(pady=5)

        # Game statistics
        stats_frame = ttk.Frame(frame)
        stats_frame.pack(fill=tk.X, pady=10)

        ttk.Label(stats_frame, text=f"{game_result['player1_name']}:").grid(row=0, column=0, sticky=tk.W, pady=2)
        ttk.Label(stats_frame, text=f"{game_result['player1_moves']} moves").grid(row=0, column=1, sticky=tk.W, pady=2)

        ttk.Label(stats_frame, text=f"{game_result['player2_name']}:").grid(row=1, column=0, sticky=tk.W, pady=2)
        ttk.Label(stats_frame, text=f"{game_result['player2_moves']} moves").grid(row=1, column=1, sticky=tk.W, pady=2)

        # Moves difference
        moves_diff = abs(game_result['player1_moves'] - game_result['player2_moves'])
        ttk.Label(stats_frame, text=f"Difference:").grid(row=2, column=0, sticky=tk.W, pady=2)
        ttk.Label(stats_frame, text=f"{moves_diff} moves").grid(row=2, column=1, sticky=tk.W, pady=2)

        # Buttons
        button_frame = ttk.Frame(frame)
        button_frame.pack(fill=tk.X, pady=15)

        # Return to main menu button
        back_button = ttk.Button(button_frame, text="Return to Main Menu",
                                 command=lambda: [result_window.destroy(), self.return_to_main()])
        back_button.pack(side=tk.LEFT, padx=10)

        # Play again button
        again_button = ttk.Button(button_frame, text="Play Again",
                                  command=lambda: [result_window.destroy(), self.play_again()])
        again_button.pack(side=tk.RIGHT, padx=10)

    def return_to_main(self):
        """Return to main selection screen"""
        if self.game_frame:
            self.game_frame.destroy()
            self.game_frame = None

        self.show_selection_screen()

    def play_again(self):
        """Play another game with the same players"""
        if self.game_frame:
            self.game_frame.destroy()
            self.game_frame = None

            # Restart with same settings
            self.start_game()

    def show_stats_screen(self):
        """Show player statistics screen"""
        # Clear existing frames
        for widget in self.container.winfo_children():
            widget.pack_forget()

        self.stats_frame = ttk.Frame(self.container)
        self.stats_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        title_label = ttk.Label(self.stats_frame, text="Player Statistics",
                                font=("Arial", 18, "bold"))
        title_label.pack(pady=10)

        # Create tabs
        tab_control = ttk.Notebook(self.stats_frame)

        # Overview tab
        overview_tab = ttk.Frame(tab_control)
        tab_control.add(overview_tab, text="Overview")

        # Player details tab
        details_tab = ttk.Frame(tab_control)
        tab_control.add(details_tab, text="Player Details")

        tab_control.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

        # Load game history data
        try:
            history_df = pd.read_csv(self.game_history_file)
            if len(history_df) > 0:
                self.create_overview_stats(overview_tab, history_df)
                self.create_player_stats(details_tab, history_df)
            else:
                ttk.Label(overview_tab, text="No game history found. Play some games first!",
                          font=("Arial", 12)).pack(pady=50)
        except Exception as e:
            ttk.Label(overview_tab, text=f"Error loading game history: {str(e)}",
                      font=("Arial", 12)).pack(pady=50)

        # Back button
        back_button = ttk.Button(self.stats_frame, text="Back to Main Menu",
                                 command=self.show_selection_screen, width=20)
        back_button.pack(pady=15)

    def create_overview_stats(self, parent, history_df):
        """Create overview statistics visualizations"""
        # Create frame for the plot
        plot_frame = ttk.Frame(parent)
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Calculate player win statistics
        player_wins = self.calculate_player_wins(history_df)

        if len(player_wins) > 0:
            # Create figure and axis
            fig, ax = plt.subplots(figsize=(8, 5))

            # Sort by win count descending
            player_wins.sort_values('wins', ascending=False, inplace=True)

            # Get player names
            player_names = []
            for filename in player_wins.index:
                player_row = self.player_df[self.player_df['filename'] == filename]
                if len(player_row) > 0:
                    player_names.append(player_row.iloc[0]['player_name'])
                else:
                    player_names.append(filename)

            # Plot top 5 players
            top_n = min(5, len(player_wins))
            bars = ax.bar(player_names[:top_n], player_wins['wins'].values[:top_n])

            # Add win count labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                        f"{height:.0f}", ha='center', va='bottom')

            ax.set_title('Top Players by Win Count')
            ax.set_xlabel('Player')
            ax.set_ylabel('Number of Wins')

            # Create canvas
            canvas = FigureCanvasTkAgg(fig, master=plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            # Create a table with more detailed stats
            stats_frame = ttk.Frame(parent)
            stats_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

            # Create treeview for stats table
            columns = ('player', 'wins', 'losses', 'total_games', 'avg_moves')
            tree = ttk.Treeview(stats_frame, columns=columns, show='headings')

            # Define headings
            tree.heading('player', text='Player')
            tree.heading('wins', text='Wins')
            tree.heading('losses', text='Losses')
            tree.heading('total_games', text='Total Games')
            tree.heading('avg_moves', text='Avg Moves')

            # Define column widths
            tree.column('player', width=150)
            tree.column('wins', width=80, anchor=tk.CENTER)
            tree.column('losses', width=80, anchor=tk.CENTER)
            tree.column('total_games', width=100, anchor=tk.CENTER)
            tree.column('avg_moves', width=100, anchor=tk.CENTER)

            # Add scrollbar
            scrollbar = ttk.Scrollbar(stats_frame, orient=tk.VERTICAL, command=tree.yview)
            tree.configure(yscroll=scrollbar.set)

            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            tree.pack(fill=tk.BOTH, expand=True)

            # Insert data
            for filename, row in player_wins.iterrows():
                player_row = self.player_df[self.player_df['filename'] == filename]
                if len(player_row) > 0:
                    player_name = player_row.iloc[0]['player_name']
                else:
                    player_name = filename

                # Calculate average moves
                player_moves = []
                for _, game in history_df.iterrows():
                    if game['player1_filename'] == filename:
                        player_moves.append(game['player1_moves'])
                    elif game['player2_filename'] == filename:
                        player_moves.append(game['player2_moves'])

                avg_moves = sum(player_moves) / len(player_moves) if player_moves else 0

                tree.insert('', tk.END, values=(
                    player_name,
                    int(row['wins']),
                    int(row['total_games'] - row['wins']),
                    int(row['total_games']),
                    f"{avg_moves:.1f}"
                ))
        else:
            ttk.Label(plot_frame, text="No win data available yet",
                      font=("Arial", 12)).pack(pady=50)

    def create_player_stats(self, parent, history_df):
        """Create detailed player statistics view"""
        # Create player selection frame
        selection_frame = ttk.Frame(parent)
        selection_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(selection_frame, text="Select Player:").pack(side=tk.LEFT, padx=5)

        # Player selection combobox
        player_names = list(self.player_df['player_name'])
        player_var = tk.StringVar()

        player_combo = ttk.Combobox(selection_frame, textvariable=player_var,
                                    values=player_names, state="readonly", width=30)
        player_combo.pack(side=tk.LEFT, padx=5)

        if player_names:
            player_combo.current(0)

        # Stats display frame
        stats_display = ttk.Frame(parent)
        stats_display.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Function to update stats when player selection changes
        def update_player_stats(*args):
            # Clear existing stats
            for widget in stats_display.winfo_children():
                widget.destroy()

            player_name = player_var.get()
            if not player_name:
                return

            # Get player filename
            player_row = self.player_df[self.player_df['player_name'] == player_name].iloc[0]
            filename = player_row['filename']

            # Calculate detailed stats for this player
            detailed_stats = self.calculate_detailed_player_stats(history_df, filename)

            # Display stats
            ttk.Label(stats_display, text=f"Statistics for {player_name}",
                      font=("Arial", 14, "bold")).pack(pady=10)

            # Create a frame for the stats grid
            grid_frame = ttk.Frame(stats_display)
            grid_frame.pack(fill=tk.X, pady=5)

            # Add stats in a grid
            row = 0
            for stat_name, stat_value in detailed_stats.items():
                ttk.Label(grid_frame, text=stat_name + ":",
                          font=("Arial", 11)).grid(row=row, column=0, sticky=tk.W, padx=10, pady=5)

                ttk.Label(grid_frame, text=str(stat_value),
                          font=("Arial", 11)).grid(row=row, column=1, sticky=tk.W, padx=10, pady=5)
                row += 1

            # If there's game history, show a plot of move count history
            if 'move_history' in detailed_stats and detailed_stats['move_history']:
                # Create figure for move history
                fig, ax = plt.subplots(figsize=(8, 3))

                moves = detailed_stats['move_history']
                games = range(1, len(moves) + 1)

                ax.plot(games, moves, marker='o', linestyle='-', linewidth=2)
                ax.set_title(f'Move History for {player_name}')
                ax.set_xlabel('Game Number')
                ax.set_ylabel('Number of Moves')

                # Add trend line
                z = np.polyfit(games, moves, 1)
                p = np.poly1d(z)
                ax.plot(games, p(games), "r--", alpha=0.8)

                # Create canvas
                canvas_frame = ttk.Frame(stats_display)
                canvas_frame.pack(fill=tk.BOTH, expand=True, pady=10)

                canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
                canvas.draw()
                canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Bind the update function to selection change
        player_var.trace('w', update_player_stats)

        # Initial update
        if player_names:
            update_player_stats()

    def calculate_player_wins(self, history_df):
        """Calculate win statistics for all players"""
        # Initialize a dictionary to track wins and total games
        player_stats = {}

        # Process each game
        for _, game in history_df.iterrows():
            player1 = game['player1_filename']
            player2 = game['player2_filename']
            winner = game['winner']

            # Initialize players if not seen before
            if player1 not in player_stats:
                player_stats[player1] = {'wins': 0, 'total_games': 0}
            if player2 not in player_stats:
                player_stats[player2] = {'wins': 0, 'total_games': 0}

            # Update stats
            player_stats[player1]['total_games'] += 1
            player_stats[player2]['total_games'] += 1

            if winner == player1:
                player_stats[player1]['wins'] += 1
            elif winner == player2:
                player_stats[player2]['wins'] += 1

        # Convert to DataFrame
        stats_df = pd.DataFrame.from_dict(player_stats, orient='index')
        return stats_df

    def calculate_detailed_player_stats(self, history_df, player_filename):
        """Calculate detailed statistics for a specific player"""
        stats = {
            'Total Games': 0,
            'Wins': 0,
            'Losses': 0,
            'Win Rate': '0%',
            'Total Moves': 0,
            'Average Moves': 0,
            'Lowest Moves': 0,
            'Highest Moves': 0,
            'move_history': []  # For plotting
        }

        moves_list = []

        # Process each game
        for _, game in history_df.iterrows():
            player1 = game['player1_filename']
            player2 = game['player2_filename']
            winner = game['winner']

            if player1 == player_filename:
                stats['Total Games'] += 1
                moves = game['player1_moves']
                moves_list.append(moves)
                stats['move_history'].append(moves)

                if winner == player_filename:
                    stats['Wins'] += 1
                else:
                    stats['Losses'] += 1

            elif player2 == player_filename:
                stats['Total Games'] += 1
                moves = game['player2_moves']
                moves_list.append(moves)
                stats['move_history'].append(moves)

                if winner == player_filename:
                    stats['Wins'] += 1
                else:
                    stats['Losses'] += 1

        # Calculate derived statistics
        if stats['Total Games'] > 0:
            stats['Win Rate'] = f"{(stats['Wins'] / stats['Total Games']) * 100:.1f}%"

        if moves_list:
            stats['Total Moves'] = sum(moves_list)
            stats['Average Moves'] = f"{stats['Total Moves'] / len(moves_list):.1f}"
            stats['Lowest Moves'] = min(moves_list)
            stats['Highest Moves'] = max(moves_list)

        return stats


class GamePlayerWrapper:
    """Wrapper for VisualGamePlayer to handle game completion events"""

    def __init__(self, parent_frame, agent1, agent2, player1_name, player2_name,
                 player1_filename, player2_filename, on_complete_callback):
        self.parent_frame = parent_frame
        self.agent1 = agent1
        self.agent2 = agent2
        self.player1_name = player1_name
        self.player2_name = player2_name
        self.player1_filename = player1_filename
        self.player2_filename = player2_filename
        self.on_complete_callback = on_complete_callback

        # Start the game
        self.start_game()

    def start_game(self):
        """Initialize and start the visual game player"""

        # Subclass VisualGamePlayer to add game completion tracking
        class TrackedVisualGamePlayer(VisualGamePlayer):
            def __init__(self, outer, *args, **kwargs):
                self.outer = outer
                super().__init__(*args, **kwargs)

            def step(self):
                # Call the original step method
                continuing = super().step()

                # Check if game is done and handle completion
                if not continuing:
                    self.outer.on_game_complete()

                return continuing

        # Create the tracked game player
        self.game_player = TrackedVisualGamePlayer(
            self,
            self.agent1, self.agent2,
            agent1_name=self.player1_name,
            agent2_name=self.player2_name,
            use_gui=True
        )

        # Start the game
        self.game_player.play_game(True)

    def on_game_complete(self):
        """Handle game completion"""
        # Extract game results
        game_result = {
            'player1_name': self.player1_name,
            'player2_name': self.player2_name,
            'player1_filename': self.player1_filename,
            'player2_filename': self.player2_filename,
            'player1_moves': self.game_player.shots1,
            'player2_moves': self.game_player.shots2
        }

        # Determine winner
        if self.game_player.shots1 < self.game_player.shots2:
            game_result['winner_name'] = self.player1_name
            game_result['winner_filename'] = self.player1_filename
        elif self.game_player.shots2 < self.game_player.shots1:
            game_result['winner_name'] = self.player2_name
            game_result['winner_filename'] = self.player2_filename
        else:
            # It's a tie, but we'll record player1 as winner for simplicity
            game_result['winner_name'] = f"Tie: {self.player1_name} & {self.player2_name}"
            game_result['winner_filename'] = self.player1_filename

        # Call the callback with results
        self.on_complete_callback(game_result)


# Main entrypoint
def main():
    """Main entry point for the application"""
    root = tk.Tk()
    app = BattleshipGameApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
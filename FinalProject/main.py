# main.py - This will be our entry point for the executable

from smart_agent import SmartAgent
from visual_game_player import VisualGamePlayer
from model_comparison import load_agent_from_file
import os
import sys


def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


def main():
    try:
        # Load saved agents using resource path to handle bundled resources
        loaded_smart_agent = load_agent_from_file(resource_path("saved_agents/smart_agent.pkl"))
        loaded_thomson_sampling_agent = load_agent_from_file(resource_path("saved_agents/thomson_sampling_agent.pkl"))

        # Create visual game player with custom names
        player = VisualGamePlayer(
            loaded_smart_agent, loaded_thomson_sampling_agent,
            agent1_name="Captain Smart",
            agent2_name="Captain Thompson",
            use_gui=True  # GUI version for the executable
        )

        # Start the game
        player.play_game(True)
    except Exception as e:
        # Basic error handling to show errors if something goes wrong
        import tkinter as tk
        from tkinter import messagebox
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Error", f"An error occurred: {str(e)}")
        root.destroy()
        raise e


if __name__ == "__main__":
    main()
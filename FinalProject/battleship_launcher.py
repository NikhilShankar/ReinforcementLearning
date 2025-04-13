# battleship_launcher.py
"""
Main entry point for the Battleship Game Application
This launcher file can be used to create the executable
"""

import os
import sys
from battleship_game_app import BattleshipGameApp
import tkinter as tk


def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


def main():
    """Main entry point for the application"""
    try:
        # Set up the tkinter root window
        root = tk.Tk()
        root.title("Battleship Game")

        # Set window icon if available
        try:
            icon_path = resource_path("game_data/icon.ico")
            if os.path.exists(icon_path):
                root.iconbitmap(icon_path)
        except:
            pass  # Skip icon if not available

        # Initialize and start the application
        app = BattleshipGameApp(root)
        root.mainloop()
    except Exception as e:
        # Basic error handling for executable
        import traceback
        error_msg = f"An error occurred: {str(e)}\n\n{traceback.format_exc()}"

        try:
            # Try to show GUI error message
            import tkinter.messagebox as messagebox
            messagebox.showerror("Error", error_msg)
        except:
            # Fallback to console error
            print(error_msg)
            input("Press Enter to exit...")

        sys.exit(1)


if __name__ == "__main__":
    main()
import sys
import os
import subprocess


def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


def print_header():
    print("=========================================")
    print("   VISIONARY HAND TRACKING SUITE")
    print("   Created by @Shehzad")
    print("=========================================")


def run_app(script_path):
    try:
        subprocess.run(
            [sys.executable, script_path],
            check=True
        )
    except subprocess.CalledProcessError as e:
        print("\n❌ Application crashed.")
        print(e)
        input("\nPress Enter to return to menu...")


def main():
    clear_screen()  # clear only once at startup

    while True:
        print_header()
        print("\nSelect an Application:")
        print("1. BoxelXR (3D Voxel Editor)")
        print("2. Neon Bubble Popper (AR Physics Game)")
        print("Q. Quit")

        choice = input("\n> ").strip().lower()

        if choice == '1':
            print("\nLaunching BoxelXR...\n")
            run_app("apps/voxel_editor.py")

        elif choice == '2':
            print("\nLaunching Neon Bubble Popper...\n")
            run_app("apps/neon_popper.py")

        elif choice == 'q':
            print("Exiting...")
            sys.exit()

        else:
            input("\nInvalid choice. Press Enter to try again.")

        print()  # spacing between menu refreshes


if __name__ == "__main__":
    main()

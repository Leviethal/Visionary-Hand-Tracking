import sys
import os


def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


def print_header():
    print("=========================================")
    print("   VISIONARY HAND TRACKING SUITE")
    print("   Created by @Shehzad")
    print("=========================================")


def main():
    while True:
        clear_screen()
        print_header()
        print("\nSelect an Application:")
        print("1. BoxelXR (3D Voxel Editor)")
        print("2. Neon Bubble Popper (AR Physics Game)")
        print("Q. Quit")

        choice = input("\n> ").strip().lower()

        if choice == '1':
            print("Launching BoxelXR...")
            os.system(f'"{sys.executable}" apps/voxel_editor.py')

        elif choice == '2':
            print("Launching Neon Bubble Popper...")
            os.system(f'"{sys.executable}" apps/neon_popper.py')

        elif choice == 'q':
            print("Exiting...")
            sys.exit()
        else:
            input("Invalid choice. Press Enter to try again.")


if __name__ == "__main__":
    main()
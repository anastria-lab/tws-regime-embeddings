import era5Downloader
import sys

def ask_user(question):
    """Asks a yes/no question via input and returns a boolean."""
    check = input(f"{question} (y/n): ").lower().strip()
    try:
        if check[0] == 'y':
            return True
        elif check[0] == 'n':
            return False
        else:
            print("Invalid input, please enter 'y' or 'n'.")
            return ask_user(question)
    except Exception:
        return False

def main():
    print("--- 🌍 TWS Regime Embeddings Pipeline ---")
    
    # Check if user needs new data
    if ask_user("Do you need to update your ERA5 data?"):
        print("🚀 Starting downloader...")
        try:
            era5Downloader.download_era5_data("era5_2002_2026.zip")
            print("✅ Data update complete.")
        except Exception as e:
            print(f"❌ Download failed: {e}")
            sys.exit(1)
    else:
        print("⏭️ Skipping download. Using existing data.")

    # Your next processing steps go here
    print("--- Proceeding to Analysis ---")

if __name__ == "__main__":
    main()
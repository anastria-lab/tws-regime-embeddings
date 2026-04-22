import sys
import era5Downloader
import era5QualityCheck


def ask_user(question):
    """Asks a yes/no question via input and returns a boolean."""
    check = input(f"{question} (y/n): ").lower().strip()
    try:
        if check[0] == "y":
            return True
        elif check[0] == "n":
            return False
        else:
            print("Invalid input, please enter 'y' or 'n'.")
            return ask_user(question)
    except Exception:
        return False


def ask_for_variable(ds):
    variables = era5QualityCheck.list_variables(ds)

    while True:
        var = input("Enter variable name to plot: ").strip()
        if var in variables:
            return var
        print("Invalid variable name. Please choose one from the list above.")


def ask_for_year():
    while True:
        try:
            year = int(input("Enter year (e.g. 2023): ").strip())
            return year
        except ValueError:
            print("Invalid year. Please enter an integer.")


def ask_for_month():
    while True:
        try:
            month = int(input("Enter month (1-12): ").strip())
            if 1 <= month <= 12:
                return month
            print("Month must be between 1 and 12.")
        except ValueError:
            print("Invalid month. Please enter an integer.")


def main():
    print("--- 🌍 TWS Regime Embeddings Pipeline ---")

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

    print("📂 Loading ERA5 data...")
    era5_data = era5QualityCheck.load_era5_data(era5QualityCheck.FILE_PATH)

    if era5_data is None:
        print("❌ Failed to load ERA5 data.")
        sys.exit(1)

    print("✅ ERA5 data loaded.")

    if ask_user("Do you want to inspect dataset info?"):
        era5QualityCheck.get_data_info(era5_data)

    if ask_user("Do you want a lightweight quality check on one timestep?"):
        print("🔍 Running lightweight quality check...")
        try:
            era5QualityCheck.check_data_quality(era5_data, sample_time_index=0)
            print("✅ Quality check complete.")
        except Exception as e:
            print(f"❌ Quality check failed: {e}")

    while ask_user("Do you want to plot a variable for a chosen year and month?"):
        try:
            variable = ask_for_variable(era5_data)
            year = ask_for_year()
            month = ask_for_month()

            print(f"🗺️ Plotting {variable} for {year}-{month:02d}...")
            result = era5QualityCheck.plot_variable_map(
                era5_data,
                variable=variable,
                year=year,
                month=month
            )

            if result is not None:
                print("✅ Plot complete.")
            else:
                print("❌ Plot failed or no matching timestep found.")

        except Exception as e:
            print(f"❌ Plot failed: {e}")

    print("--- Proceeding to Analysis ---")


if __name__ == "__main__":
    main()
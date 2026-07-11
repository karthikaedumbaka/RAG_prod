import requests
from pathlib import Path
import sys

PDF_URL = "https://www.niti.gov.in/sites/default/files/2024-07/SDA_INDIA_0.pdf"

DATA_FOLDER = Path("data")
FILE_NAME = "SDA_INDIA_2023_24.pdf"
SAVE_PATH = DATA_FOLDER / FILE_NAME


def format_size(size):
    """Convert bytes to human-readable format."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} TB"


def download_pdf():
    # Create folder if it doesn't exist
    DATA_FOLDER.mkdir(parents=True, exist_ok=True)

    # Check if file already exists
    if SAVE_PATH.exists():
        file_size = SAVE_PATH.stat().st_size

        print("\n====================================")
        print("📄 PDF already exists!")
        print(f"Location : {SAVE_PATH.resolve()}")
        print(f"Size     : {format_size(file_size)}")
        print("====================================")

        while True:
            choice = input(
                "\nDo you want to download it again? (y/n): "
            ).strip().lower()

            if choice in ("y", "yes"):
                SAVE_PATH.unlink()
                print("Old file deleted.\n")
                break

            elif choice in ("n", "no"):
                print("Using existing PDF.")
                return

            else:
                print("Please enter 'y' or 'n'.")

    print("Downloading PDF...\n")

    try:
        response = requests.get(PDF_URL, stream=True, timeout=60)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        downloaded = 0

        with open(SAVE_PATH, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)
                    downloaded += len(chunk)

                    if total_size:
                        percent = downloaded * 100 / total_size
                        print(
                            f"\rDownloading: {percent:6.2f}% "
                            f"({format_size(downloaded)} / {format_size(total_size)})",
                            end="",
                            flush=True,
                        )

        print("\n\nDownload completed successfully!")
        print(f"Saved to: {SAVE_PATH.resolve()}")

    except requests.exceptions.RequestException as e:
        print(f"\nDownload failed!\nReason: {e}")
        sys.exit(1)


if __name__ == "__main__":
    download_pdf()
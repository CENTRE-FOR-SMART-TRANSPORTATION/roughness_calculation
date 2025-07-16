import os
import pandas as pd


def main():
    filenames = []
    for file in os.listdir('data/raw/L1_GT'):
        if file.endswith('.xlsx') or file.endswith('.txt'):
            filenames.append(file)

    print(f"Found {len(filenames)} files to combine: {filenames}")

    for filename in filenames:
        combined_data = pd.DataFrame()
        for folder in os.listdir(os.path.join('data', 'raw')):
            if not os.path.isdir(os.path.join('data', 'raw', folder)):
                continue
            path = os.path.join('data', 'raw', folder, filename)
            if os.path.exists(path):
                data_ = pd.read_excel(path) if filename.endswith(
                    '.xlsx') else pd.read_csv(path, sep=" ")
                combined_data = pd.concat(
                    [combined_data, data_], ignore_index=True)

        if not combined_data.empty:
            output_path = os.path.join('data', 'combined', filename)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            if filename.endswith('.xlsx'):
                combined_data.to_excel(output_path, index=False)
            else:
                combined_data.to_csv(output_path, index=False, sep=' ')
            print(f"Combined data saved to {output_path}")


if __name__ == '__main__':
    main()

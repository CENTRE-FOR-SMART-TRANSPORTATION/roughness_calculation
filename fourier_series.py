import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

plt.switch_backend('Agg')


def fourier_series_iri(data: pd.DataFrame, nho1, nho2) -> float:
    """
    Calculate the Fourier series coefficients for a given signal.

    Parameters:
    x (pd.DataFrame): Input consisting of intervals and elevations.
    n1 (int): The number of harmonics to consider for the sine terms.
    n2 (int): The number of harmonics to consider for the cosine terms.

    Returns:
    tuple: A tuple containing the sine and cosine coefficients.
    """
    x = data[0].values  # First column (X)
    z = data[1].values  # Second column (Z)

    # Sort data by x
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    z_sorted = z[sort_idx]

    # Step 1: Detrend the data
    model = LinearRegression()
    model.fit(x_sorted.reshape(-1, 1), z_sorted)  # Fit linear model
    z_trend = model.predict(x_sorted.reshape(-1, 1))  # Linear trend
    z_detrended = z_sorted - z_trend  # Detrended data

    # Step 2: Fit Fourier series to detrended data
    L = np.max(x_sorted) - np.min(x_sorted)  # Length of the data
    a0 = np.mean(z_detrended)  # Mean of the detrended data

    # Initialize Fourier fits
    z_fourier_fit_o = a0
    z_fourier_fit_o1 = a0

    # Generate Fourier series fit for nho1
    for n in range(1, nho1 + 1):
        an = np.sum(z_detrended * np.cos(2 * np.pi * n *
                    x_sorted / L)) * 2 / len(z_detrended)
        bn = np.sum(z_detrended * np.sin(2 * np.pi * n *
                    x_sorted / L)) * 2 / len(z_detrended)
        z_fourier_fit_o += an * \
            np.cos(2 * np.pi * n * x_sorted / L) + bn * \
            np.sin(2 * np.pi * n * x_sorted / L)

    # Generate Fourier series fit for nho2
    for n in range(1, nho2 + 1):
        an = np.sum(z_detrended * np.cos(2 * np.pi * n *
                    x_sorted / L)) * 2 / len(z_detrended)
        bn = np.sum(z_detrended * np.sin(2 * np.pi * n *
                    x_sorted / L)) * 2 / len(z_detrended)
        z_fourier_fit_o1 += an * \
            np.cos(2 * np.pi * n * x_sorted / L) + bn * \
            np.sin(2 * np.pi * n * x_sorted / L)

    # Step 3: Add the linear trend back to the Fourier fits
    z_fourier_fit_o += z_trend
    z_fourier_fit_o1 += z_trend

    # Compute absolute difference between the two Fourier series
    abs_difference = np.abs(z_fourier_fit_o - z_fourier_fit_o1)

    # Estimate x-spacing (assuming uniform spacing)
    dx = np.mean(np.diff(x_sorted))

    # Calculate the area between curves (IRI)
    area_between_curves = np.sum(abs_difference) * dx * 1000 / L

    return area_between_curves


def graph(id: np.ndarray, x: np.ndarray, y: np.ndarray, save: bool = True, filename: str = None):
    """
    Create a graph of the given data.

    Parameters:
    id (np.ndarray): The x-axis data.
    x (np.ndarray): The y-axis data for the first line.
    y (np.ndarray): The y-axis data for the second line.
    save (bool): Whether to save the graph as an image file.
    filename (str): The name of the file to save the graph.
    """
    os.makedirs(os.path.dirname(filename),
                exist_ok=True) if save and filename else None

    plt.figure(figsize=(10, 5))
    plt.plot(id, x, label='Ground Truth', color='blue')
    plt.plot(id, y, label='Fourier Series', color='red')
    plt.title('Comparison')
    plt.xlabel('ID')
    plt.ylabel('IRI')
    plt.ylim(0, 10)  # Bound y-axis between 0 and 100
    plt.legend()
    plt.grid()

    if save and filename:
        plt.savefig(filename)
    else:
        plt.show()

    plt.close()


def main():
    """
    Data Structure:
    .
    ├── L1_GT
    │   ├── Excel files ...
    ├── L2_GT
    │   ├── Excel files ...
    ├── S1_GT
    │   ├── Excel files ...
    └── S2_GT
        └── Excel files ...
    """
    nho1 = list(range(1, 8))
    # nho2 = list(range(20, 40)) + list(range(40, 70, 3)) + \
    #     list(range(70, 150, 5)) + list(range(150, 251, 10))
    nho2 = list(range(200, 500, 10)) + list(range(500, 1000, 25))

    filenames = os.listdir(os.path.join('data', 'raw', 'L1_GT'))
    folder_names = os.listdir(os.path.join('data', 'raw'))
    error_calc = pd.DataFrame()
    gt = pd.read_csv(os.path.join('data', 'gt.csv'), header=None)
    gt.sort_values(by=[gt.columns[0], gt.columns[1]], inplace=True)
    gt.fillna(0, inplace=True)

    for nho1_val in tqdm(nho1):
        for nho2_val in tqdm(nho2):
            iris = pd.DataFrame(
                columns=['filename', 'folder', 'iri', 'filtered_iri'])
            # print(f"Processing with nho1: {nho1_val}, nho2: {nho2_val}")
            for folder in folder_names:
                for filename in filenames:
                    path = os.path.join('data', 'raw', folder, filename)
                    if not os.path.exists(path):
                        # print(
                        #     f"Warning: {path} does not exist. Skipping this file.")
                        continue

                    if filename.endswith('.xlsx'):
                        data = pd.read_excel(path, header=None)
                        data = data.apply(
                            pd.to_numeric, errors='coerce').dropna(subset=[0, 1])
                        data.sort_values(by=[0, 1], inplace=True)
                        if data.empty:
                            # print(
                            #     f"Warning: No valid data in {filename}. Skipping this file.")
                            iris = pd.concat(
                                [iris, pd.DataFrame([{'filename': filename, 'folder': folder, 'iri': 0, 'filtered_iri': 0}])], ignore_index=True)
                            continue
                        iri = fourier_series_iri(data, nho1_val, nho2_val)
                        iris = pd.concat(
                            [iris, pd.DataFrame([{'filename': filename, 'folder': folder, 'iri': iri}])], ignore_index=True)

                Q1 = iris['iri'].quantile(0.25)
                Q3 = iris['iri'].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                filtered_iris = iris[(iris['iri'] >= lower_bound) & (
                    iris['iri'] <= upper_bound)].copy()

                # Assign filtered_iri only to the valid indices
                iris.loc[filtered_iris.index,
                         'filtered_iri'] = filtered_iris['iri']

                # Optional: fill missing (NaNs) with 0s if desired
                iris['filtered_iri'].fillna(0, inplace=True)

            # Fill na values with 0
            mae = np.mean(np.abs(gt[2].values - iris['filtered_iri'].values))
            rmse = np.sqrt(np.mean((gt[2].values - iris['iri'].values) ** 2))
            error_calc = pd.concat([error_calc, pd.DataFrame(
                [{'nho1': nho1_val, 'nho2': nho2_val, 'mae': mae, 'rmse': rmse}])], ignore_index=True)

            iris.sort_values(by=['filename', 'folder'], inplace=True)
            iris['gt_mesh'] = gt[2].values
            save_path = f'output/csv/{nho1_val}_{nho2_val}.xlsx'
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            iris.to_excel(save_path, index=False)
            graph(range(len(iris)), iris['gt_mesh'].values, iris['filtered_iri'].values,
                  save=True, filename=f'output/figures/{nho1_val}_{nho2_val}.png')

    error_calc.sort_values(by=['nho1', 'nho2'], inplace=True)
    error_calc.to_excel('output/mae_all.xlsx', index=False)
    print("Processing complete. Results saved to output directory.")


if __name__ == '__main__':
    main()

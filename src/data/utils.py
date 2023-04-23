"""Utility script for abstracting the data view from the actual Gridification of the data."""

from collections import Counter
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from src.data.grid import Grid


def calculate_sample_weights(data: tuple,
                             scale: bool,
                             transform_labels: bool = True) -> np.ndarray:
    """
    Calculates a weight for a list of training samples based on the frequency of the label 
    occurrence depending on the target cell. In contrast to calculating label weights (where weights
    are calculated only based on the absolute occurrence in the dataset) this approach may yield
    different weights for a sample with the same label but different target cells.

    Parameters
    ----------
        data : tuple[list, np.ndarray]
            A tuple consisting of the input data list (which is itself a list of np.ndarrays) and
            the np.ndarray of labels.
        scale : bool
            Scales the values between 0 and 1 if True. Returns values of varying scale otherwise.
        transform_labels : bool
            Transforms one-hot encoded labels back into real-value representation if True. True by
            default.

    Returns
    -------
        An np.ndarray of the same length as data[0] and data[1] where the i-th value in the returned
        array constitutes the weight for the i-th sample in the training data.
    """
    # Transform one-hot encoded representation back into real-value representation for labels.
    if transform_labels:
        labels = np.where(data[1] == 1)[1]
    # Transform array of array of values into a format that's usable for the Counter object.
    else:
        labels = data[1].flatten()
    target_cell_indicators = data[0][-1]
    if len(labels) != len(target_cell_indicators):
        raise ValueError(
            "Target cell indicator array and label array are not of the same length!"
        )
    counters = [Counter() for _ in range(target_cell_indicators.shape[-1])]
    target_cells = np.where(target_cell_indicators == 1)
    for sample_idx, cell_idx in zip(target_cells[0], target_cells[1]):
        counters[cell_idx].update([labels[sample_idx]])
    cell_label_weights = []
    for i in range(target_cell_indicators.shape[-1]):
        cell_label_weights.append(_calculate_label_weights(counters[i], scale))
    return _match_sample_with_label_weight(labels, target_cells[1],
                                           cell_label_weights)


def _calculate_label_weights(counter: Counter, scale: bool) -> dict:
    """
    Calculates the weight of a label based on its frequency. Scales between 0 and 1 if 'scale' is
    True.
    """
    total_values = sum(counter.values())
    l = []
    w = []
    for label, counts in counter.items():
        l.append(label)
        weight = total_values / counts
        w.append(weight)
    if scale:
        max_weight = np.array(w).max()
        for i, weights in enumerate(w):
            w[i] = weights / max_weight
    return {label: weight for label, weight in zip(l, w)}


def _match_sample_with_label_weight(labels: np.ndarray,
                                    target_cell_idx: np.ndarray,
                                    label_weights: list) -> np.ndarray:
    """
    Generates a list of weights per sample, given a training data and a label weights per target
    cell.
    """
    sample_weights = np.empty(labels.shape[0], dtype=np.float64)
    for i, target in enumerate(target_cell_idx):
        label = labels[i]
        sample_weights[i] = label_weights[target][label]
    return sample_weights


def scale_breakin_values(grid: pd.DataFrame,
                         global_max: int = None) -> pd.DataFrame:
    """
    Takes a Grid dataframe and min-max-normalizes break-in values globally. This means that every
    value is divided by the largest value of all cells, rather than the largest value of a given
    cell.
    """
    breakin_columns = [
        colname for colname in grid.columns if "cell" in colname
    ]

    if not global_max:
        global_max = determine_global_max(grid)

    # Normalize all columns.
    for col in breakin_columns:
        grid[col] = grid[col] / global_max

    return grid


def scale_weather_values(train_grid: pd.DataFrame,
                         test_grid: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a Grid dataframe and min-max-scales every feature of the 'weather' group of variables.
    """
    weather_columns = [
        colname for colname in train_grid.columns if "weather" in colname
    ]
    scaler = MinMaxScaler()
    for col in weather_columns:
        scaler.fit(train_grid[[col]])
        train_grid[col] = scaler.transform(train_grid[[col]])
        test_grid[col] = scaler.transform(test_grid[[col]])
    return train_grid, test_grid


def determine_global_max(grid: pd.DataFrame):
    """
    Determines the largest break-in value in all cells at all times and returns that value.
    """
    breakin_columns = [
        colname for colname in grid.columns if "cell" in colname
    ]
    global_max = 0
    for col in breakin_columns:
        local_max = grid[col].max()
        if local_max > global_max:
            global_max = local_max
    return global_max


def generate_data_windows(grid: pd.DataFrame,
                          label_grid: pd.DataFrame = None,
                          input_timesteps: int = 7,
                          merge_breakins_and_dates: bool = False) -> tuple:
    """
    Takes a Grid dataframe and turns it into a tuple of input and label np.ndarrays.

    Parameters
    ----------
        grid : pd.DataFrame
            A dataframe describing the Grid data.
        label_grid : pd.DataFrame, optional
            If normalization techniques were applied to the original grid, an unnormalized version
            of the grid should be provided to gather real-value labels. If no unnormalized grid is
            provided, this function will generate labels based on break-in values of the input grid.
        input_timesteps : int, optional
            The amount of dates included in a single input array.
        merge_breakins_and_dates :  bool, optional
            Whether or not to include date information in the same array as break-in information. 

    Returns
    -------
        A tuple of length 2 containing:
            [0] A list of input data ndarrays.
            [1] An np.ndarray encoding label data.
    """
    cell_count = len([col for col in grid.columns if "cell" in col])
    variable_mapping = _get_variable_mapping(grid)
    grid_array = grid.to_numpy()
    if isinstance(label_grid, pd.DataFrame):
        label_grid_array = label_grid.to_numpy()
    else:
        label_grid_array = grid_array

    dataset_length = (len(grid_array) - input_timesteps) * cell_count
    if merge_breakins_and_dates:
        variable_mapping = _merge_breakins_and_dates(variable_mapping)
    inputs = []
    for input_category, variable_range in variable_mapping.items():
        variable_length = variable_range[1] - variable_range[0] + 1
        if input_category == "breakins":
            input_arr = np.empty(
                (dataset_length, input_timesteps, variable_length),
                dtype=np.float32)
        else:
            input_arr = np.empty((dataset_length, variable_length),
                                 dtype=np.float32)
        inputs.append(input_arr)
    labels = np.empty((dataset_length, 1), dtype=np.float32)
    targets = np.zeros((dataset_length, cell_count), dtype=np.int8)

    # Define index-tracking variable outside of loop as it is tied exclusively to the inner loop.
    i = 0
    for label_index in range(input_timesteps, len(grid_array)):
        input_templates = []
        for input_category, variable_range in variable_mapping.items():
            if input_category == "breakins":
                template = grid_array[label_index -
                                      input_timesteps:label_index,
                                      variable_range[0]:variable_range[1] + 1]
            else:
                template = grid_array[label_index,
                                      variable_range[0]:variable_range[1] + 1]
            input_templates.append(template)
        for j in range(cell_count):
            for variable_type_idx, template in enumerate(input_templates):
                inputs[variable_type_idx][i] = template
            labels[i] = label_grid_array[label_index, j]
            targets[i, j] = 1
            i += 1
    inputs.append(targets)
    return inputs, labels


def _merge_breakins_and_dates(variable_mapping: dict) -> dict:
    """
    Merges the entries for breakins and dates into a single entry in the variable mapping.
    Because the classifiers are sensitive to the order of the 'variable_mapping' keys, we have to
    make sure the order remains intact otherwise.
    """
    key_list = list(variable_mapping.keys())
    if not (key_list.index("breakins") == 0 and key_list.index("date") == 1):
        raise ValueError(
            "Variable mapping is not of the right order. Consult source code for more information."
        )
    # Let us assume if the above condition is not violated, the dict is formatted correctly.
    start_index = variable_mapping["breakins"][0]
    end_index = variable_mapping["date"][1]
    new_mapping = {"breakins": [start_index, end_index]}
    for input_category, variable_range in variable_mapping.items():
        if input_category in ["breakins", "date"]:
            continue
        new_mapping[input_category] = variable_range
    return new_mapping


def _get_variable_mapping(data: pd.DataFrame, ) -> dict:
    """
    Creates a mapping dict that contains beginning and end endices for different variable types
    contained in the dataset.
    """
    var_map = {}
    for i, var_name in enumerate(data.columns):
        var_type = var_name.split("_")[0]
        if var_type not in var_map:
            var_map[var_type] = [i, i]
            continue
        var_map[var_type][1] = i
    return var_map


def get_dataset(
    date_range: pd.date_range,
    grid_size_vertical: int = 5,
    grid_size_horizontal: int = 5,
    auxiliary_data: list = None,
    encode_event_data: bool = False,
) -> pd.DataFrame:
    """
    Get a dataframe of crime data according to the specified configuration. Checks if CSV with
    required configuration exists already. If it does, the CSV is loaded into a dataframe. If not,
    generate new CSV and return the resulting dataframe.
    """
    filename = _generate_name_for_csv(date_range, grid_size_vertical,
                                      grid_size_horizontal, auxiliary_data)
    try:
        data = pd.read_csv(filename)
        print("Dataset loaded from CSV.")
    except FileNotFoundError:
        print("Building dataset from scratch.")
        generate_dataset_csv(date_range, grid_size_vertical,
                             grid_size_horizontal, auxiliary_data)
        data = pd.read_csv(filename)
    if "events" in auxiliary_data and encode_event_data:
        data = _multihot_encode_event_data(data)
    return data


def generate_dataset_csv(
    date_range: pd.date_range,
    grid_size_vertical: int = 5,
    grid_size_horizontal: int = 5,
    auxiliary_data: list = None,
) -> pd.DataFrame:
    """
    Populates a Grid object with crime data and required auxiliary data and writes the resulting
    Grid objects into a persistent CSV file.
    Returns Grid object.
    """
    # Load WED data and structure it into Grid.
    wed = pd.read_excel("data/WED_data.xlsx")
    wed = wed[wed["TZ_DATUM"].isin(date_range)]
    wed.dropna(inplace=True)
    lat_col_name = "Breitengrad_reduziert"
    long_col_name = "Längengrad_reduziert"
    wed[long_col_name] = wed[long_col_name].str.replace(",", ".", regex=False)
    wed[long_col_name] = wed[long_col_name].astype(float)
    wed[lat_col_name] = wed[lat_col_name].str.replace(",", ".", regex=False)
    wed[lat_col_name] = wed[lat_col_name].astype(float)
    wed.dropna(inplace=True)
    grid = Grid(
        latitudes=wed.Breitengrad_reduziert,
        longitudes=wed.Längengrad_reduziert,
        grid_size_horizontal=grid_size_horizontal,
        grid_size_vertical=grid_size_vertical,
    )
    grid.populate(wed)

    if isinstance(auxiliary_data, list):
        if "weather" in auxiliary_data:
            _add_weather_data(grid)
        if "events" in auxiliary_data:
            _add_event_data(grid)
        if "social" in auxiliary_data:
            _add_social_data(grid)

    filename = _generate_name_for_csv(date_range, grid_size_vertical,
                                      grid_size_horizontal, auxiliary_data)
    grid.to_csv(filename)

    return grid


def split_input_into_tuples(raw_inputs: np.ndarray,
                            variable_map: dict) -> list:
    """
    Takes the potential training data input of 'generate_data_windows'[0] and the variable map
    generated by 'get_dataset_with_variable_mapping' and splits the input data into n different
    np.ndarrays where n is the length of the variable_map.
    """
    inputs = []
    for var_range in variable_map.values():
        variable_array = raw_inputs[:, :, var_range[0]:var_range[1] + 1]
        inputs.append(variable_array)
    return inputs


def _multihot_encode_event_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Removes the original columns encoding event data and replaces each of them with columns
    one-hot encoding the original values in the cell.
    Important: this function assumes that events columns exist in the DataFrame and does not handle
    cases where that assertion fails.
    """
    events_cols = [col for col in data.columns if "events" in col]
    for col in events_cols:
        feature_col = data[col]
        for value in set(feature_col.values):
            data[f"{col}_{value}"] = (feature_col == value).replace({
                True: 1,
                False: 0
            })
        del data[col]
    return data


def _generate_name_for_csv(
    date_range: pd.date_range,
    grid_size_vertical: int = 5,
    grid_size_horizontal: int = 5,
    auxiliary_data: list = None,
) -> str:
    """
    Generates a name for the CSV file for a Grid of specified configuration.
    """
    covered_dates = f"{date_range[0].year}-{date_range[-1].year}"
    grid_format = f"{grid_size_vertical}x{grid_size_horizontal}"
    data_types = "_".join(["crimes"] + auxiliary_data)
    filename = f"data/grids/{data_types}_{covered_dates}_{grid_format}.csv"
    return filename


def _add_weather_data(grid: Grid, normalize: bool = False):
    """
    Loads weather data from spreadsheet, normalizes values, and adds them to the Grid object. During
    normalization, wind direction and wind speed are converted into vectors. This transformation is
    needed because wind direction is in units of degrees by default. This is a bad model input
    because 360° and 0° should be close to one another in the model.
    """
    weather = pd.read_excel("data/weather_data.xlsx")
    # Remove data for the year 2020.
    weather = weather[weather["date"].isin(grid.date_range)]
    weather.fillna(method="ffill", inplace=True)

    # Transform wind data.
    direction = weather.pop("wdir")
    velocity = weather.pop("wspd")
    max_velocity = weather.pop("wpgt")
    rad_direction = direction * np.pi / 180
    weather["wind_x"] = velocity * np.cos(rad_direction)
    weather["wind_y"] = velocity * np.sin(rad_direction)
    weather["max_wind_x"] = max_velocity * np.cos(rad_direction)
    weather["max_wind_y"] = max_velocity * np.sin(rad_direction)

    if normalize:
        weather = _normalize_dataframe(weather)

    # Add data to grid.
    grid.add_auxiliary_data(data_category="weather", data=weather)


def _add_event_data(grid: Grid, normalize: bool = False):
    """
    Load event data into Grid object.
    """
    events = pd.read_csv("data/event_data.csv")
    events["date"] = pd.to_datetime(events["date"])
    events = events[events["date"].isin(grid.date_range)]
    # Remove location specificity.
    if "location" in events.columns:
        events = events.drop(["location"], axis=1).drop_duplicates()

    if normalize:
        events = _normalize_dataframe(events)

    grid.add_auxiliary_data(data_category="events", data=events)


def _add_social_data(grid: Grid, normalize: bool = False):
    """
    Load social data into Grid object.
    """
    social_data = pd.read_csv("data/social_data.csv")
    social_data["date"] = pd.to_datetime(social_data["date"])
    social_data = social_data[social_data["date"].isin(grid.date_range)]

    if normalize:
        social_data = _normalize_dataframe(social_data)

    grid.add_auxiliary_data(data_category="socio-economic", data=social_data)


def _normalize_dataframe(data: pd.DataFrame, strategy: str = "minmaxscaler"):
    """
    Normalizes data contained in the provided dataframe.
    """
    date = data.pop("date")
    if strategy == "standard":
        data = (data - data.mean()) / data.std()
    elif strategy == "minmaxscaler":
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(data)
        for i, col in enumerate(data.columns):
            data[col] = scaled[:, i]
    else:
        raise ValueError("Unsupported strategy type.")
    data["date"] = date
    return data.copy()

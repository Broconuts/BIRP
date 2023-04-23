"""Script for the Grid class."""

import logging
import pandas as pd
import numpy as np
from tqdm import tqdm
from src.data.cell import Cell, SAMPLING_WINDOWS


class Grid:
    """
    Class for mapping a set of geo-spatial coordinates onto a grid of square areas. The class
    calculates the size of individual cells based on provided parameters.
    """

    def __init__(self,
                 latitudes: pd.Series,
                 longitudes: pd.Series,
                 grid_size_horizontal: int,
                 grid_size_vertical: int,
                 normalize: bool = False):
        """
        Initialize the grid.
        """
        if len(latitudes) != len(longitudes):
            raise ValueError(
                "List of latitudes and list of longitudes must be of the same length."
            )
        # Variable to check whether Grid is empty to avoid accidentally iterating over empty Grid.
        self.contains_data = False
        # Determine horizontal scale.
        self.grid_size_horizontal = grid_size_horizontal
        self.long_min = longitudes.min()
        self.long_max = longitudes.max()
        self.single_cell_length = (self.long_max -
                                   self.long_min) / (grid_size_horizontal - 1)

        # Determine vertical scale.
        self.grid_size_vertical = grid_size_vertical
        self.lat_min = latitudes.min()
        self.lat_max = latitudes.max()
        self.single_cell_height = (self.lat_max -
                                   self.lat_min) / (grid_size_vertical - 1)

        # Create variable for holding cells.
        self.cells = np.empty(
            shape=(self.grid_size_vertical, self.grid_size_horizontal),
            dtype=Cell,
        )
        self.normalize = normalize

        # Lazily initialized variables.
        self.first_date = None
        self.last_date = None
        self.date_range = None
        self.number_of_crime_instances = None
        self.auxiliary_data = {}
        self.grid_timeseries = None
        self._raw_grid_timeseries = None

    def __iter__(self):
        """
        Make Grid object iterable. This function returns an iterator that flatly iterates over
        call cells line by line, column by column.
        """
        if not self.contains_data:
            raise RuntimeError(
                "Cannot iterate over an empty Grid. Please populate Grid with data first."
            )
        return (cell for cell in self.cells.flatten())

    def __repr__(self):
        """
        Summarizes data contained in the Grid object; provides information regarding timespan of
        the contained data, spatial binning, and how many data points are contained in it.
        """
        summary = ""
        if self.contains_data:
            summary += "\n".join([
                f"First recorded date: {self.first_date}",
                f"Last recorded date: {self.last_date}",
                f"Total recorded crime occurrences: {self.number_of_crime_instances}\n"
            ])
        else:
            summary += "Grid currently contains no data points.\n"
        summary += "\n".join([
            f"Total cells: {self.grid_size_vertical * self.grid_size_horizontal}",
            f"{self.grid_size_horizontal} columns and {self.grid_size_vertical} rows."
        ])
        return summary

    def populate(self, data: pd.DataFrame):
        """
        Create individual Cell objects and write crime time series into them.
        """
        if self.contains_data:
            raise RuntimeError(
                "Trying to populate a Grid that was already populated with data. Please flush Grid "
                "before attempting to populate again.")
        self.contains_data = True
        self.number_of_crime_instances = len(data)
        self.first_date = data["TZ_DATUM"].min()
        self.last_date = data["TZ_DATUM"].max()
        self.date_range = pd.date_range(start=self.first_date,
                                        end=self.last_date)
        for _, row in data.iterrows():
            row_idx, col_idx = self.coords_to_cell_idx(
                latitude=row["Breitengrad_reduziert"],
                longitude=row["Längengrad_reduziert"],
            )
            # Create a new Cell object in the grid if this is the first occurrence.
            try:
                if not self.cells[row_idx, col_idx]:
                    self.cells[row_idx, col_idx] = Cell(row_idx, col_idx)
            except IndexError:
                print(f"Row: {row_idx}")
                print(f"Col: {col_idx}")
                print("Raw row data:")
                print(row)
            self.cells[row_idx, col_idx].update(row["TO_ORTSTEIL"],
                                                row["TZ_DATUM"])

    def add_auxiliary_data(self, data_category: str, data: pd.DataFrame):
        """
        Add auxiliary data to the Grid object. This function provides the logic for adding all three
        supported types of additional data, namely 'weather', 'events', and 'socio-economic' data.
        """
        data.sort_values(by="date", inplace=True, ignore_index=True)
        if self.contains_data:
            covered_dates = pd.Series(data["date"].unique())
            if not pd.Series(self.date_range).equals(covered_dates):
                logging.warning(
                    "%s data does not cover entire date range of crime data. Grid contains %s data "
                    "points, added data contains %s data points.",
                    data_category, len(self.date_range), len(covered_dates))
        else:
            logging.warning(
                "{data_category} data was added without checks for date coverage."
            )
        # Handle city-wide events first. In these cases, you do not need dates as indices.
        if data_category in ["weather", "events"]:
            self.auxiliary_data[data_category] = data
        # Hanndle socio-economic data. These will be split into individual dicts per part-of-city.
        elif data_category == "socio-economic":
            logging.warning(
                "Using socio-economic data is deprecated and can cause problems in other functions."
            )
            self.auxiliary_data[data_category] = {}
            for location in data["location"].unique():
                temp_df = data[data["location"] == location].drop("location",
                                                                  axis=1)
                self.auxiliary_data[data_category][location] = temp_df.drop(
                    "date", axis=1).to_numpy()
        else:
            raise ValueError(
                "Data must be of category 'weather', 'socio-economic', or 'events'."
            )

    def flush(self):
        """
        Empties the cells of the Grid. This leaves the Grid in a state similar to the one after
        initialization. Essentially, this function reverts all uses of the populate function on the
        Grid.
        """
        if self.contains_data:
            self.cells = np.empty(
                shape=(self.grid_size_vertical, self.grid_size_horizontal),
                dtype=Cell,
            )

    def coords_to_cell_idx(self, latitude: float, longitude: float) -> tuple:
        """
        Uses coordinates to find the indices of the cell that covers that area of the grid.
        """
        if (latitude < self.lat_min
                or latitude > self.lat_max) or (longitude < self.long_min
                                                or longitude > self.long_max):
            raise ValueError("Coordinates are outside of the defined grid!")
        # Turn latitude and longitude into relative values.
        longitude = longitude - self.long_min
        latitude = latitude - self.lat_min
        row_idx = ((self.grid_size_vertical - 1) -
                   round(latitude / self.single_cell_height))
        col_idx = round(longitude / self.single_cell_length)
        return row_idx, col_idx

    def count_dead_cells(self, absolute_values: bool = False) -> float:
        """
        Identifies how many cells do not contain any occurrences. This is a relevant metric for
        determining whether cell size is too small. Function returns percentage of dead cells by
        default, but can return absolute values if specified.
        """
        if not self.contains_data:
            raise RuntimeError(
                "Cannot count dead cells for an unpopulated Grid. Please populate Grid first."
            )
        dead_cells = 0
        for cell in self.cells.flatten():
            if cell is None:
                dead_cells += 1
        if absolute_values:
            return dead_cells
        return round(
            dead_cells / (self.grid_size_vertical * self.grid_size_horizontal),
            4)

    def count_inactive_cells_by_time_interval(self, interval: str) -> float:
        """
        Identifies how many times any cell exhibit no activity within a specified time interval.
        Always returns the percentage (as a value between 0 and 1) because absolute numbers convey
        little meaning given the large number of total instances.
        This function only considers globally alive cells, as having no globally dead cells should
        be a criterion that is fulfilled in any case.
        """
        if not self.contains_data:
            raise RuntimeError(
                "Cannot count inactive cells for an unpopulated Grid. Please populate Grid first."
            )
        if interval not in SAMPLING_WINDOWS:
            raise ValueError(
                f"Parameter 'interval' must be in {SAMPLING_WINDOWS.keys()}.")
        inactive_cell_counter = 0
        for cell in tqdm(self.get_alive_cells()):
            activity = cell.aggregate_occurrences(time_window=interval)
            inactive_cell_counter += activity[
                "crime_occurrences"].value_counts()[0]
        return inactive_cell_counter

    def get_alive_cells(self, return_indices: bool = False) -> list:
        """
        Returns a flat list of all cells that recorded criminal activity. List contains actual Cell
        objects by default, or the index tuples if specified.
        """
        if not self.contains_data:
            raise RuntimeError(
                "Cannot count alive cells for an unpopulated Grid. Please populate Grid first."
            )
        alive_cells = []
        row_len = self.cells.shape[1]
        col_len = self.cells.shape[0]
        for i in range(col_len):
            for j in range(row_len):
                if self.cells[i][j]:
                    if return_indices:
                        alive_cells.append((i, j))
                    else:
                        alive_cells.append(self.cells[i][j])
        return alive_cells

    def _initialize_grid_timeseries(self):
        """
        Generates an ndarray that contains a flat list of crime occurrence datapoints for each of
        the cells in the Grid. This list can be appended with additional data if specified in the
        function parameters.
        """
        grid_series = []
        for date in self.date_range:
            day_list = []
            for cell in self.cells.flatten():
                try:
                    occurrence = cell.get_crime_occurrence_for_date(date)
                # Handle case of dead (i.e. never initialized) cell.
                except AttributeError:
                    occurrence = 0
                day_list.append(occurrence)
            grid_series.append(np.array(day_list, dtype=np.float32))
        self.grid_timeseries = np.array(grid_series, dtype=np.int16)
        if self.normalize:
            self._raw_grid_timeseries = np.copy(self.grid_timeseries)
            self.grid_timeseries = np.asarray(
                list(map(lambda a: a / 14, self.grid_timeseries)))

    def to_csv(self, path: str):
        """
        Creates a CSV of the timeseries. The CSV contains a row for every date in 'Grid.date_range'.
        Every row contains a value for each cell in the Grid, describing the number of break-ins in
        that cell on that date. In addition to that, each row contains a two values encoding time.
        Finally, each row contains the values of the auxiliary data for a given day.

        Parameters
        ----------
            path : str
                The path where the CSV file should be stored.
        """
        if self.grid_timeseries is None:
            self._initialize_grid_timeseries()
        rows = []
        for i, date in enumerate(self.date_range):
            rows.append(self._get_date_row(date, i))
        pd.DataFrame(rows).to_csv(path, index=False)

    def _get_date_row(self, date: pd.DatetimeIndex, date_index: int) -> dict:
        """
        Compiles all relevant data points for a given date into a row dict.

        Parameters
        ----------
            date : pd.DatetimeIndex
                The date for which the row is being built.
            date_index : int
                The index of the date in the date range of the Grid object. This is needed because
                the grid_timeseries is integer-indexed, not date-indexed; we need an integer to
                retrieve the break-ins for a given date.
        """
        row = {}

        # Compile break-in data.
        break_ins = self.grid_timeseries[date_index].flatten()
        for i, occurrences in enumerate(break_ins):
            row[f"breakins_cell{i}"] = occurrences

        # Compile time-related data.
        row["date_day-of-week"] = date.dayofweek
        row["date_week-of-year"] = date.weekofyear

        # Compile auxiliary data - if present.
        for data_type, data in self.auxiliary_data.items():
            if data_type == "socio-economic":
                raise NotImplementedError(
                    "Adding socio-economic data is currently not supported.")
            for variable_name, value in data.loc[data["date"] == date].items():
                if variable_name == "date":
                    continue
                # Value is a list of length 1, therefore we need to supply an index.
                row[f"{data_type}_{variable_name}"] = value.values[0]

        return row

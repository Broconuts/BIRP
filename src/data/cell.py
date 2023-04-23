"""Script for the Cell class."""

import logging
import pandas as pd

SAMPLING_WINDOWS = {"weekly": "W", "monthly": "M", "annual": "Y"}
# Store start and end dates for the WED dataset in globals for now.
START_DATE = "1/1/2016"
END_DATE = "31/12/2020"


class Cell:
    """
    Helper class for managing the state of Cells in a Grid.
    """

    def __init__(self, row: int, col: int):
        self.row = row
        self.col = col
        self.part_of_city_counter = {}
        self.crime_occurrences = {}

    def get_crime_occurrence_for_date(self, date: pd.Timestamp) -> int:
        """
        Returns number of crime occurrences for a specified date. If date is not in the object dict
        'crime_occurrences', we know that 0 crimes occurred during that date.
        """
        try:
            return self.crime_occurrences[date]
        except KeyError:
            return 0

    def _update_part_of_city_counter(self, poc: str):
        """
        Checks if part of city was already added to cell object. If not, adds it to object and sets
        counter to 1. Else, it increases the counter of the existing element by 1.
        """
        try:
            self.part_of_city_counter[poc] += 1
        except KeyError:
            self.part_of_city_counter[poc] = 1

    def _update_crime_occurrences(self, date: pd.Timestamp):
        """
        Checks if a crime has already been registered for the provided timestamp. If not, it creates
        an entry for the given timestamp and sets the counter to 1. Else, it increases the counter
        for the existing timestamp by 1.
        """
        try:
            self.crime_occurrences[date] += 1
        except KeyError:
            self.crime_occurrences[date] = 1

    def update(self, poc: str, date: pd.Timestamp):
        """
        Public method for updating every aspect of a cell with a single function call. Hands down
        parameters to relevant private updating functions.
        """
        self._update_part_of_city_counter(poc)
        self._update_crime_occurrences(date)

    def to_timeseries(self,
                      start: str = START_DATE,
                      end: str = END_DATE,
                      add_time_variables: bool = True) -> pd.DataFrame:
        """
        Converts the dense crime_occurrence dictionary into a sparse time series with additional
        time-based features to make time series prediction easier.
        """
        date_range = pd.date_range(start=start, end=end)
        crimes = []
        for date in date_range:
            try:
                count = self.crime_occurrences[date]
            except KeyError:
                count = 0
            crimes.append(count)
        data = {"date": date_range, "crime_occurrences": crimes}
        time_series = pd.DataFrame(data=data)
        # Make sure that values are sorted, even though they technically should already.
        time_series.sort_values(by="date", inplace=True)
        if add_time_variables:
            time_series["day_of_month"] = time_series["date"].dt.day
            time_series["day_of_week"] = time_series["date"].dt.dayofweek
            time_series["month"] = time_series["date"].dt.month
            time_series["week"] = time_series["date"].dt.isocalendar().week
        return time_series.set_index("date")

    def aggregate_occurrences(self,
                              time_window: str = "weekly",
                              start: str = START_DATE,
                              end: str = END_DATE,
                              add_time_variables: bool = True) -> pd.Series:
        """
        Aggregate data into one of the three pre-defined time windows.
        """
        time_window = time_window.lower()
        if time_window not in SAMPLING_WINDOWS:
            raise ValueError(
                f"Invalid time_window value. Must be in {SAMPLING_WINDOWS.keys()}."
            )
        time_series = self.to_timeseries(start, end, add_time_variables=False)
        sampled_time_series = time_series.resample(
            SAMPLING_WINDOWS[time_window]).sum()
        if add_time_variables:
            if time_window in ["weekly", "monthly"]:
                sampled_time_series["month"] = sampled_time_series.index.month
            if time_window == "weekly":
                sampled_time_series[
                    "week"] = sampled_time_series.index.isocalendar().week
        return sampled_time_series

    def get_dominant_part_of_city(self) -> str:
        """
        Returns the part(s) of city that is most prevalent in this cell. If one part of city has a
        majority of instances, the returned list will be of length 1. Otherwise the list will be of
        length n where n is the number of parts of city that contributed the same amount of cases.
        """
        current_max = {"part_of_city": [], "amount": 0}
        for poc, amount in self.part_of_city_counter.items():
            if amount > current_max["amount"]:
                current_max["part_of_city"] = [poc]
            elif amount == current_max["amount"]:
                current_max["part_of_city"].append(poc)
        if len(current_max["part_of_city"]) != 1:
            logging.warning(
                "Cell [%s, %s] has two equally dominant parts-of-city.",
                self.row, self.col)
        return current_max["part_of_city"][0]

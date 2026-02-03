# From: gluonts/src/gluonts/time_feature/_base.py
# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

from typing import List, Tuple, Any

import numpy as np
import pandas as pd
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset


class TimeFeature:
    def __init__(self):
        pass

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        pass

    def __repr__(self):
        return self.__class__.__name__ + "()"


class SecondOfMinute(TimeFeature):
    """Minute of hour encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.second / 59.0


class MinuteOfHour(TimeFeature):
    """Minute of hour encoded as value between [0.5, 1.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        moh = index.minute / 59.0
        return moh + 0.5

class HourOfDay(TimeFeature):
    """Hour of day encoded as value between [0.5, 1.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        ori_val = index.hour / 23.0
        return ori_val + 0.5


class DayOfWeek(TimeFeature):
    """Hour of day encoded as value between [0.5, 1.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        dow = index.dayofweek / 6.0
        return dow + 0.5


class DayOfMonth(TimeFeature):
    """Day of month encoded as value between [0.5, 1.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        dom = (index.day - 1) / 30.0
        return dom + 0.5


class DayOfYear(TimeFeature):
    """Day of year encoded as value between [0.5, 1.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        doy = (index.dayofyear - 1) / 365.0
        return doy + 0.5


class MonthOfYear(TimeFeature):
    """Month of year encoded as value between [0.5, 1.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        moy = (index.month - 1) / 11.0
        return moy + 0.5


class WeekOfYear(TimeFeature):
    """Week of year encoded as value between [0.5, 1.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.isocalendar().week - 1) / 52.0 + 0.5


def time_features_from_frequency_str(freq_str: str) -> List[TimeFeature]:
    """
    Returns a list of time features that will be appropriate for the given frequency string.
    Parameters
    ----------
    freq_str
        Frequency string of the form [multiple][granularity] such as "12H", "5min", "1D" etc.
    """

    features_by_offsets = {
        offsets.YearEnd: [],
        offsets.QuarterEnd: [MonthOfYear],
        offsets.MonthEnd: [MonthOfYear],
        offsets.Week: [DayOfMonth, WeekOfYear],
        offsets.Day: [DayOfWeek, DayOfMonth, DayOfYear, MonthOfYear],
        offsets.BusinessDay: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Hour: [HourOfDay, DayOfWeek, DayOfMonth, DayOfYear, MonthOfYear],
        offsets.Minute: [
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
            MonthOfYear
        ],
        offsets.Second: [
            SecondOfMinute,
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
    }

    offset = to_offset(freq_str)

    for offset_type, feature_classes in features_by_offsets.items():
        if isinstance(offset, offset_type):
            return [cls() for cls in feature_classes]

    supported_freq_msg = f"""
    Unsupported frequency {freq_str}
    The following frequencies are supported:
        Y   - yearly
            alias: A
        M   - monthly
        W   - weekly
        D   - daily
        B   - business days
        H   - hourly
        T   - minutely
            alias: min
        S   - secondly
    """
    raise RuntimeError(supported_freq_msg)


def time_features(dates, freq='h'):
    return np.vstack([feat(dates) for feat in time_features_from_frequency_str(freq)])

if __name__ == '__main__':
    print(time_features("2017", freq='h'))
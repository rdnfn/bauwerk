"""Module with standard environment configs."""


EXP001_SINGLE_DATA = {
    "general": {
        "type": "solar_battery_house.SolarBatteryHouseEnv",
        "infeasible_control_penalty": True,
        "grid_charging": True,
        "logging_level": "WARNING",  # if using RLlib, set to 'RAY'
    },
    "components": {
        "battery": {
            "type": "LithiumIonBattery",
            "size": 10,
            "chemistry": "NMC",
            "time_step_len": 1,
        },
        "solar": {
            "type": "DataPV",
            "data_path": None,
            "fixed_sample_num": 12,
        },
        "load": {
            "type": "DataLoad",
            "data_path": None,
            "fixed_sample_num": 12,
        },
        "grid": {
            "type": "PeakGrid",
            "peak_threshold": 1.0,
        },
    },
}


DEFAULT_ENV_CONFIG = EXP001_SINGLE_DATA

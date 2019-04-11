import glob
import os
import configparser
from pathlib import Path


class DataManager:

    def __init__(self, assets_path_str):
        self.assets_path = Path(assets_path_str)
        self.input_path = self.assets_path / "input"
        self.cells_path = self.assets_path / "cells"
        self.out_path = self.assets_path / "out"
        self._allowed_input_extensions = [".png", ".jpeg", ".jpg"]
        self._allowed_output_extensions = [".png"]

        self._classes = {0: "epiteliali",
                         1: "neutrofili",
                         2: "eosinofili",
                         3: "mastcellule",
                         4: "linfociti",
                         5: "mucipare",
                         6: "macchie",
                         7: "nuclei_nudi"}

        self.classes_path = [self.out_path / class_name for class_name in self._classes.values()]

    @classmethod
    def from_file(cls, config_file="config.ini"):
        config = configparser.ConfigParser()
        config.read(config_file)

        data_manager = cls(config["Paths"]["assets"])
        data_manager.input_path = Path(config["Paths"]["input_path"])
        data_manager.cells_path = Path(config["Paths"]["cells_path"])
        data_manager.out_path = Path(config["Paths"]["output_path"])

        data_manager._allowed_input_extensions = config["Misc"]["input_img_extensions"].split(";")
        data_manager._allowed_output_extensions = config["Misc"]["export_img_extension"].split(";")

        return data_manager

    @staticmethod
    def get_file_by_extensions(path, allowed_extensions):
        files = []
        for extension in allowed_extensions:
            files.extend([str(path) for path in path.glob("*" + extension)])
        return files

    def get_input_images(self):
        return DataManager.get_file_by_extensions(self.input_path, self._allowed_input_extensions)

    def get_cells_images(self):
        return DataManager.get_file_by_extensions(self.cells_path, self._allowed_input_extensions)

    def get_output_extension(self):
        return self._allowed_output_extensions[0]

    def get_assets_path(self):
        return str(self.assets_path)

    def get_input_path(self):
        return str(self.input_path)

    def get_cells_path(self):
        return str(self.cells_path)

    def get_output_path(self):
        return str(self.out_path)

    def get_cell_class_path(self, class_index):
        return str(self.out_path / self._classes[class_index])

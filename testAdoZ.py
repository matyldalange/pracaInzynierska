import unittest
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel
from unittest.mock import patch, MagicMock
import os
from interface import Aplikacja


class TestAplikacja(unittest.TestCase):
    def setUp(self):
        self.app = QApplication([])
        self.window = Aplikacja()

    def test_full_process(self):
        with patch("builtins.open", MagicMock()):
            self.window.choose_image()

        with patch("cv2.imread", return_value=MagicMock()):
            with patch("cv2.imwrite", return_value=True):
                self.window.apply_mask()

        output_folder = "/Users/matyldalange/Desktop/obrazy_z_maseczkami/"
        files = os.listdir(output_folder)
        self.assertTrue(len(files) > 0, "Nie zapisano pliku do odpowiedniego folderu.")
    def tearDown(self):
        self.app.quit()


if __name__ == "__main__":
    unittest.main()

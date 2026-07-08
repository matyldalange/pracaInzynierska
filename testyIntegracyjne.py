import unittest
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from PyQt5.QtTest import QTest
from interface import Aplikacja
import sys
import os
import time

class TestAplikacji(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication(sys.argv)
        cls.window = Aplikacja()
        cls.window.show()

    def test_apply_mask_button(self):
        test_image_path = '/Users/matyldalange/Desktop/003397.jpg'
        self.window.selected_image_path = test_image_path

        QTest.mouseClick(self.window.button_mask, Qt.LeftButton)
        time.sleep(1)
        self.assertTrue(self.window.label.text().startswith("Przetworzyliśmy już Twój obraz!"))

    def test_apply_mask_rgb(self):
        test_image_path = '/Users/matyldalange/Desktop/003397.jpg'
        self.window.selected_image_path = test_image_path
        self.window.apply_mask_rgb()

        masked_image_path = '/Users/matyldalange/Desktop/obrazy_z_maseczkami/maseczka_003397.jpg'
        self.assertTrue(os.path.exists(masked_image_path), "Maska nie została zapisana poprawnie!")

    def test_apply_mask_thermal(self):
        test_image_path = '/Users/matyldalange/Desktop/N11186.jpg'
        self.window.selected_image_path = test_image_path
        self.window.apply_mask_thermal(None, '/Users/matyldalange/Desktop/obrazy_z_maseczkami/')

        masked_image_path = '/Users/matyldalange/Desktop/obrazy_z_maseczkami/N11186_masked.jpg'
        self.assertTrue(os.path.exists(masked_image_path), "Plik z maską termograficzną nie został zapisany poprawnie!")

    @classmethod
    def tearDownClass(cls):
        cls.app.quit()

if __name__ == '__main__':
    unittest.main()





import unittest
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication
from interface import Aplikacja
import os


class TestAplikacja(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication([])
        cls.window = Aplikacja()
    def setUp(self):
        self.window.show()
        self.window.label.setText("")
        self.window.image_type = None
    def test_check_image_type_thermal(self):
        image_path = '/Users/matyldalange/Desktop/N11186.jpg'
        self.window.selected_image_path = image_path
        self.window.check_image_type()
        self.assertEqual(self.window.image_type, 'thermal', "Obraz nie został rozpoznany jako termograficzny!")
    def test_check_image_type_rgb(self):
        image_path = '/Users/matyldalange/Desktop/003397.jpg'
        self.window.selected_image_path = image_path
        self.window.check_image_type()
        self.assertEqual(self.window.image_type, 'rgb', "Obraz nie został rozpoznany jako RGB!")
    def test_apply_mask_thermal(self):
        image_path = '/Users/matyldalange/Desktop/N11186.jpg'
        self.window.selected_image_path = image_path
        self.window.check_image_type()
        output_folder = '/Users/matyldalange/Desktop/obrazy_z_maseczkami/'
        output_path = os.path.join(output_folder, 'N11186_masked.jpg')
        self.window.apply_mask_thermal(image_path, output_folder)
        self.assertTrue(os.path.exists(output_path), "Obraz termograficzny nie został zapisany poprawnie!")
    def test_apply_mask_rgb(self):
        image_path = '/Users/matyldalange/Desktop/003397.jpg'
        self.window.selected_image_path = image_path
        self.window.check_image_type()
        output_folder = "/Users/matyldalange/Desktop/obrazy_z_maseczkami/"
        output_path = os.path.join(output_folder, f"maseczka_{os.path.basename(image_path)}")
        self.window.apply_mask_rgb()
        self.assertTrue(os.path.exists(output_path), "Obraz RGB nie został zapisany poprawnie!")
    def tearDown(self):
        self.window.close()
    @classmethod
    def tearDownClass(cls):
        cls.app.quit()


if __name__ == '__main__':
    unittest.main()




import unittest
import os
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer
from PyQt5.QtTest import QTest
from interface import Aplikacja


class TestStyledApp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Inicjalizacja aplikacji PyQt5 dla testów GUI"""
        cls.app = QApplication([])

    def setUp(self):
        """Tworzenie instancji StyledApp przed każdym testem"""
        self.window = Aplikacja()
        self.window.show()

        # Tworzenie testowego obrazu
        self.test_image_path = "/Users/matyldalange/Desktop/test_image.jpeg"
        with open(self.test_image_path, "w") as f:
            f.write("")  # Tworzymy pusty plik jako placeholder

    def tearDown(self):
        """Zamykanie okna i usuwanie testowego obrazu"""
        self.window.close()
        if os.path.exists(self.test_image_path):
            os.remove(self.test_image_path)

    def test_choose_image(self):
        self.window.selected_image_path = self.test_image_path
        self.assertTrue(os.path.exists(self.test_image_path), "Wybrany obraz nie istnieje.")

    def test_fun_fact_timer(self):
        self.window.show_fun_fact()
        self.assertTrue(self.window.fun_fact_label.isVisible(), "Ciekawostka nie została wyświetlona.")

        QTest.qWait(11000)
        self.assertFalse(self.window.fun_fact_label.isVisible(), "Ciekawostka nie została ukryta po 10 sekundach.")

    def test_doctor_animation(self):
        initial_x = self.window.doctor_label.x()
        self.window.move_doctor()
        updated_x = self.window.doctor_label.x()
        self.assertNotEqual(initial_x, updated_x, "Doktor nie porusza się.")
    def test_button_visibility_and_activity(self):
        self.assertTrue(self.window.button_play_game.isVisible(), "'Zagraj w grę' nie jest widoczny.")
        self.assertTrue(self.window.button_folder.isVisible(), "'Wybierz zdjęcie z folderu' nie jest widoczny.")
        self.assertTrue(self.window.button_mask.isVisible(), "'Nałóż maseczkę' nie jest widoczny.")
        self.assertTrue(self.window.button_exit.isVisible(), "'Wyjdź z aplikacji' nie jest widoczny.")

        self.assertTrue(self.window.button_play_game.isEnabled(), "'Zagraj w grę' nie jest aktywny.")
        self.assertTrue(self.window.button_folder.isEnabled(), "'Wybierz zdjęcie z folderu' nie jest aktywny.")
        self.assertTrue(self.window.button_mask.isEnabled(), "'Nałóż maseczkę' nie jest aktywny.")
        self.assertTrue(self.window.button_exit.isEnabled(), "'Wyjdź z aplikacji' nie jest aktywny.")

    @classmethod
    def tearDownClass(cls):
        """Zamykanie aplikacji PyQt5 po zakończeniu testów"""
        cls.app.quit()


if __name__ == "__main__":
    unittest.main()

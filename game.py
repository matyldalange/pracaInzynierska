import arcade
import random

# Klasa ptaka (wirusa)
class Virus(arcade.Sprite):
    def __init__(self, x, y, ground_height):
        super().__init__("/Users/matyldalange/Desktop/wirus.png", 0.025)  # Wczytanie obrazu
        self.center_x = x
        self.center_y = y
        self.ground_height = ground_height
        self.change_y = 0
        self.flap_power = 13

    def update(self):
        self.change_y -= 1  # Grawitacja
        self.center_y += self.change_y

        if self.top > 512:
            self.top = 512  # Ograniczamy ptaka do ekranu
        if self.bottom < self.ground_height:
            self.bottom = self.ground_height  # Ograniczamy ptaka do podłoża
            self.change_y = 0

    def flap(self):
        self.change_y = self.flap_power  # Nadajemy ptakowi "siłę" do lotu

# Klasa rury (przeszkody)
class Pipes:
    def __init__(self, x, gap_start, gap_size):
        self.width = 50
        self.gap_start = gap_start
        self.gap_size = gap_size

        self.top_pipe = arcade.SpriteSolidColor(self.width, 512 - (gap_start + gap_size), (0, 121, 107))
        self.bottom_pipe = arcade.SpriteSolidColor(self.width, gap_start, (0, 121, 107))

        self.top_pipe.center_x = x
        self.top_pipe.center_y = 512 - (self.top_pipe.height / 2)
        self.bottom_pipe.center_x = x
        self.bottom_pipe.center_y = self.bottom_pipe.height / 2

    def update(self):
        self.top_pipe.center_x -= 5
        self.bottom_pipe.center_x -= 5

    def draw(self):
        self.top_pipe.draw()
        self.bottom_pipe.draw()

    def check_collision(self, virus):
        return arcade.check_for_collision(virus, self.top_pipe) or arcade.check_for_collision(virus, self.bottom_pipe)

class Game(arcade.Window):
    def __init__(self, width, height):
        super().__init__(width, height, "Ucieknij od wirusa")
        self.virus = None
        self.pipes = None
        self.score = 0
        self.state = "MainMenu"
        self.pipe_spacing = 300
        self.last_pipe_x = 600
        self.background_color = (232, 245, 233)

    def setup(self):
        self.score = 0
        self.virus = Virus(50, 256, 50)
        self.pipes = []
        self.last_pipe_x = 600

    def on_draw(self):
        arcade.start_render()
        self.virus.draw()
        for pipe in self.pipes:
            pipe.draw()

        if self.state == "MainMenu":
            arcade.draw_text(
                "NACIŚNIJ SPACJĘ, ABY ZACZĄĆ",
                300, 350,
                (0, 77, 64),
                24,
                font_name="Trebuchet MS",
                align="center",
                anchor_x="center",
                anchor_y="center",
                width=self.width - 40,  # Dodany parametr 'width'
            )

        elif self.state == "Playing":
            arcade.draw_text(
                f"PUNKTY: {self.score}",
                10, 470,
                (0, 77, 64),
                18,
                font_name="Trebuchet MS",
            )

        elif self.state == "GameOver":
            arcade.draw_text(
                f"PRZEGRAŁEŚ! WYNIK: {self.score}",
                300, 280,
                (183, 28, 28),
                22,
                font_name="Trebuchet MS",
                align="center",
                anchor_x="center",
                anchor_y="center",
                width=self.width - 40,  # Dodany parametr 'width'
            )
            arcade.draw_text(
                "NAŁÓŻ MASECZKĘ, ABY UNIKNĄĆ WIRUSA!",
                300, 240,
                (183, 28, 28),
                18,
                font_name="Trebuchet MS",
                align="center",
                anchor_x="center",
                anchor_y="center",
                width=self.width - 40,  # Dodany parametr 'width'
            )

    def on_key_press(self, symbol, modifiers):
        if symbol == arcade.key.SPACE:
            if self.state == "MainMenu":
                self.state = "Playing"
                self.setup()
            elif self.state == "Playing":
                self.virus.flap()
            elif self.state == "GameOver":
                self.state = "Playing"  # Zmieniono, by kontynuować grę po przegranej
                self.setup()  # Resetujemy wszystkie elementy (np. ptaka i rury)

    def on_update(self, delta_time):
        #Jeśli gra jest w stanie "Playing" to kontynuujemy aktualizację rozgrywki
        if self.state == "Playing":
            self.virus.update()
            #Sprawdzenie, czy ostatnia przeszkoda jest odpowiednio daleko, aby dodać nową
            if self.last_pipe_x <= 600:
                gap_start = random.randint(100, 300)
                gap_size = 200
                #Tworzenie nowej rury
                pipe = Pipes(self.last_pipe_x, gap_start, gap_size)
                self.pipes.append(pipe)
                self.last_pipe_x += self.pipe_spacing
            #Aktualizacja stanu wszystkich przeszkód
            for pipe in self.pipes:
                pipe.update()
            #Sprawdzenie kolizji wirusa z przeszkodami
            for pipe in self.pipes:
                if pipe.check_collision(self.virus):
                    self.state = "GameOver"
            #Sprawdzenie, czy zdobyto punkt, czyli wirus przeszedł przez przerwę między rurami
            for pipe in self.pipes:
                if pipe.top_pipe.center_x < self.virus.center_x and not hasattr(pipe, "scored"):
                    self.score += 1
                    pipe.scored = True

            self.pipes = [pipe for pipe in self.pipes if pipe.top_pipe.center_x > 0]
            self.last_pipe_x -= 5


def main():
    game = Game(600, 512)
    game.setup()
    arcade.run()


if __name__ == "__main__":
    main()

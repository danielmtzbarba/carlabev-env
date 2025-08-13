from dataclasses import dataclass

@dataclass
class Settings:
    width: int = 1200
    height: int = 900
    offx: int = 50
    offy: int = -250
    #
    white: tuple = (255, 255, 255)
    black: tuple = (0, 0, 0)
    grey: tuple = (200, 200, 200)
    button_color: tuple = (50, 150, 50)
    blue: tuple = (0, 120, 215)
    green: tuple = (0, 215, 120)
    red: tuple = (215, 0, 120)

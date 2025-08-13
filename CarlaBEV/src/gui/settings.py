from dataclasses import dataclass

@dataclass
class Settings:
    width: int = 1280
    height: int = 1000
    #
    left_panel_w: int = 250
    right_panel_w: int = 250
    #
    offx: int = 120
    offy: int = -200
    margin_x: int = 30
    margin_y: int = 35
    #
    white: tuple = (255, 255, 255)
    black: tuple = (0, 0, 0)
    grey: tuple = (200, 200, 200)
    button_color: tuple = (50, 150, 50)
    blue: tuple = (0, 120, 215)
    green: tuple = (0, 215, 120)
    red: tuple = (215, 0, 120)

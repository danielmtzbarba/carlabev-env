import pygame 

from CarlaBEV.src.gui.settings import Settings as cfg

class TextBox:
    def __init__(self, rect, font, text=""):
        self.rect = pygame.Rect(rect)
        self.color = cfg.white 
        self.text = text
        self.font = font
        self.active = False
        self.select_all_on_next_input = False
        self.bg_color = (255, 255, 255)
        self.border_color = (192, 198, 206)
        self.active_border_color = (29, 110, 235)
        self.text_color = (28, 31, 36)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            self.active = self.rect.collidepoint(event.pos)
            self.select_all_on_next_input = self.active
        if event.type == pygame.KEYDOWN and self.active:
            if event.key == pygame.K_RETURN:
                self.active = False
                self.select_all_on_next_input = False
            elif event.key == pygame.K_ESCAPE:
                self.active = False
                self.select_all_on_next_input = False
            elif event.key == pygame.K_BACKSPACE:
                if self.select_all_on_next_input:
                    self.text = ""
                    self.select_all_on_next_input = False
                else:
                    self.text = self.text[:-1]
            elif event.key == pygame.K_DELETE:
                self.text = ""
                self.select_all_on_next_input = False
            else:
                if event.unicode:
                    if self.select_all_on_next_input:
                        self.text = event.unicode
                        self.select_all_on_next_input = False
                    else:
                        self.text += event.unicode

    def draw(self, screen):
        pygame.draw.rect(screen, self.bg_color, self.rect, border_radius=12)
        border = self.active_border_color if self.active else self.border_color
        pygame.draw.rect(screen, border, self.rect, 2, border_radius=12)
        txt_surface = self.font.render(self.text, True, self.text_color)
        screen.blit(txt_surface, (self.rect.x + 12, self.rect.y + (self.rect.height - txt_surface.get_height()) // 2))

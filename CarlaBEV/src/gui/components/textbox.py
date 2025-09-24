import pygame 

from CarlaBEV.src.gui.settings import Settings as cfg

class TextBox:
    def __init__(self, rect, font, text=""):
        self.rect = pygame.Rect(rect)
        self.color = cfg.white 
        self.text = text
        self.font = font
        self.active = False

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            self.active = self.rect.collidepoint(event.pos)
        if event.type == pygame.KEYDOWN and self.active:
            if event.key == pygame.K_RETURN:
                self.active = False
            elif event.key == pygame.K_BACKSPACE:
                self.text = self.text[:-1]
            else:
                self.text += event.unicode

    def draw(self, screen):
        pygame.draw.rect(screen, self.color, self.rect, 0)
        pygame.draw.rect(screen, cfg.black, self.rect, 2)
        txt_surface = self.font.render(self.text, True, cfg.black)
        screen.blit(txt_surface, (self.rect.x + 5, self.rect.y + 5))


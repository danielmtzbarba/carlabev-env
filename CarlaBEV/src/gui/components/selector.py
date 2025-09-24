import pygame 

from CarlaBEV.src.gui.settings import Settings as cfg

class Selector:
    def __init__(self, rect, font, options, selected=0):
        self.rect = pygame.Rect(rect)
        self.font = font
        self.options = options
        self.selected = selected
        self.option_rects = []
        self._build_option_rects()

    def _build_option_rects(self):
        self.option_rects = []
        y = self.rect.y
        for i, opt in enumerate(self.options):
            r = pygame.Rect(self.rect.x + 10, y + i * 30, 20, 20)
            self.option_rects.append(r)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            for i, r in enumerate(self.option_rects):
                if r.collidepoint(event.pos):
                    self.selected = i

    def draw(self, screen):
        for i, r in enumerate(self.option_rects):
            pygame.draw.rect(screen, cfg.black, r, 2)
            if i == self.selected:
                pygame.draw.circle(screen, cfg.blue, r.center, 7)
            text_surf = self.font.render(self.options[i], True, cfg.black)
            screen.blit(text_surf, (r.right + 5, r.y - 2))
    
    @property
    def selection(self):
        return self.options[self.selected]

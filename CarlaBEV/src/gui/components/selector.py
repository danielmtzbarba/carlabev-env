import pygame 

from CarlaBEV.src.gui.settings import Settings as cfg

class Selector:
    def __init__(self, rect, font, options, selected=0):
        self.rect = pygame.Rect(rect)
        self.font = font
        self.options = options
        self.selected = selected
        self.option_rects = []
        self.option_height = 30
        self.option_gap = 4
        self._build_option_rects()

    def _build_option_rects(self):
        self.option_rects = []
        y = self.rect.y
        for i, opt in enumerate(self.options):
            r = pygame.Rect(
                self.rect.x,
                y + i * (self.option_height + self.option_gap),
                self.rect.width,
                self.option_height,
            )
            self.option_rects.append(r)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            for i, r in enumerate(self.option_rects):
                if r.collidepoint(event.pos):
                    self.selected = i

    def draw(self, screen):
        for i, r in enumerate(self.option_rects):
            if i == self.selected:
                pygame.draw.rect(screen, (227, 238, 255), r, border_radius=12)
                pygame.draw.rect(screen, (29, 110, 235), r, 2, border_radius=12)
            else:
                pygame.draw.rect(screen, (246, 247, 249), r, border_radius=12)
                pygame.draw.rect(screen, (204, 209, 216), r, 1, border_radius=12)

            text_color = (29, 31, 36) if i != self.selected else (18, 73, 170)
            text_surf = self.font.render(self.options[i], True, text_color)
            screen.blit(text_surf, (r.x + 14, r.y + (r.height - text_surf.get_height()) // 2))
    
    @property
    def selection(self):
        return self.options[self.selected]

    def update_options(self, options):
        """Update options dynamically and reset selection."""
        if self.options != options:
            self.options = options
            if self.selected >= len(options):
                self.selected = 0
            self._build_option_rects()

    def set_metrics(self, option_height, option_gap):
        self.option_height = int(option_height)
        self.option_gap = int(option_gap)
        self._build_option_rects()

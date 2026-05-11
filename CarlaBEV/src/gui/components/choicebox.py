import pygame


class ChoiceBox:
    def __init__(self, rect, font, options=None, selected=0, labels=None):
        self.rect = pygame.Rect(rect)
        self.font = font
        self.options = list(options or [])
        self.selected = selected if self.options else None
        self.labels = dict(labels or {})
        self.bg_color = (255, 255, 255)
        self.border_color = (192, 198, 206)
        self.active_border_color = (29, 110, 235)
        self.text_color = (28, 31, 36)
        self.active = False

    def update_options(self, options, labels=None):
        previous = self.selection if self.selected is not None and self.options else None
        self.options = list(options)
        if labels is not None:
            self.labels = dict(labels)
        if not self.options:
            self.selected = None
        elif previous in self.options:
            self.selected = self.options.index(previous)
        else:
            self.selected = 0

    def set_selected_by_value(self, value):
        if value in self.options:
            self.selected = self.options.index(value)
        elif not self.options:
            self.selected = None

    def handle_event(self, event):
        if not self.options:
            return False
        if event.type == pygame.MOUSEBUTTONDOWN:
            self.active = self.rect.collidepoint(event.pos)
        if event.type == pygame.MOUSEBUTTONUP and self.rect.collidepoint(event.pos):
            if event.button == 1:
                self.selected = (self.selected + 1) % len(self.options)
                return True
            if event.button == 3:
                self.selected = (self.selected - 1) % len(self.options)
                return True
        return False

    def draw(self, screen):
        pygame.draw.rect(screen, self.bg_color, self.rect, border_radius=12)
        border = self.active_border_color if self.active else self.border_color
        pygame.draw.rect(screen, border, self.rect, 2, border_radius=12)
        label = self.display_text
        txt_surface = self.font.render(label, True, self.text_color)
        screen.blit(
            txt_surface,
            (self.rect.x + 12, self.rect.y + (self.rect.height - txt_surface.get_height()) // 2),
        )
        hint = self.font.render("< >", True, (96, 104, 117))
        screen.blit(
            hint,
            (self.rect.right - hint.get_width() - 10, self.rect.y + (self.rect.height - hint.get_height()) // 2),
        )

    @property
    def selection(self):
        if self.selected is None or not self.options:
            return None
        return self.options[self.selected]

    @property
    def display_text(self):
        value = self.selection
        if value is None:
            return ""
        return self.labels.get(value, str(value))

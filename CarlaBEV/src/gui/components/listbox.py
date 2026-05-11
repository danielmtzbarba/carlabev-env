import pygame


class ListBox:
    def __init__(self, rect, font, items=None):
        self.rect = pygame.Rect(rect)
        self.font = font
        self.items = list(items or [])
        self.selected = 0 if self.items else None
        self.hovered = None
        self.row_height = 24
        self.row_gap = 4
        self.padding = 8
        self.item_rects = []
        self._build_item_rects()

        self.bg_color = (255, 255, 255)
        self.border_color = (192, 198, 206)
        self.row_color = (246, 247, 249)
        self.hover_color = (235, 240, 247)
        self.selected_bg_color = (227, 238, 255)
        self.selected_border_color = (29, 110, 235)
        self.text_color = (28, 31, 36)
        self.selected_text_color = (18, 73, 170)

    def _build_item_rects(self):
        self.item_rects = []
        y = self.rect.y + self.padding
        width = max(0, self.rect.width - 2 * self.padding)
        for idx, _ in enumerate(self.items):
            self.item_rects.append(
                pygame.Rect(self.rect.x + self.padding, y, width, self.row_height)
            )
            y += self.row_height + self.row_gap

    def set_items(self, items):
        previous_item = self.selected_item
        self.items = list(items)
        if not self.items:
            self.selected = None
        elif previous_item in self.items:
            self.selected = self.items.index(previous_item)
        else:
            self.selected = 0
        self._build_item_rects()

    def set_metrics(self, row_height, row_gap, padding=8):
        self.row_height = int(row_height)
        self.row_gap = int(row_gap)
        self.padding = int(padding)
        self._build_item_rects()

    def set_selected_by_value(self, value):
        if value in self.items:
            self.selected = self.items.index(value)
        elif not self.items:
            self.selected = None

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.hovered = None
            for idx, item_rect in enumerate(self.item_rects):
                if item_rect.collidepoint(event.pos):
                    self.hovered = idx
                    break

        if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            for idx, item_rect in enumerate(self.item_rects):
                if item_rect.collidepoint(event.pos):
                    self.selected = idx
                    return self.items[idx]
        return None

    def draw(self, screen):
        pygame.draw.rect(screen, self.bg_color, self.rect, border_radius=12)
        pygame.draw.rect(screen, self.border_color, self.rect, 1, border_radius=12)

        for idx, item_rect in enumerate(self.item_rects):
            if idx == self.selected:
                bg = self.selected_bg_color
                border = self.selected_border_color
                text_color = self.selected_text_color
                border_width = 2
            elif idx == self.hovered:
                bg = self.hover_color
                border = self.border_color
                text_color = self.text_color
                border_width = 1
            else:
                bg = self.row_color
                border = self.border_color
                text_color = self.text_color
                border_width = 1

            pygame.draw.rect(screen, bg, item_rect, border_radius=10)
            pygame.draw.rect(screen, border, item_rect, border_width, border_radius=10)

            text_surf = self.font.render(self.items[idx], True, text_color)
            screen.blit(
                text_surf,
                (
                    item_rect.x + 12,
                    item_rect.y + (item_rect.height - text_surf.get_height()) // 2,
                ),
            )

    @property
    def selected_item(self):
        if self.selected is None or self.selected >= len(self.items):
            return None
        return self.items[self.selected]

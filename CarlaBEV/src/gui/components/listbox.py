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
        self.scroll_offset = 0
        self.max_visible_rows = 6
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
        start = self.scroll_offset
        end = min(len(self.items), start + self.visible_capacity)
        for idx in range(start, end):
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
        self.scroll_offset = min(self.scroll_offset, self.max_scroll_offset)
        self._ensure_selection_visible()
        self._build_item_rects()

    def set_metrics(self, row_height, row_gap, padding=8, max_visible_rows=6):
        self.row_height = int(row_height)
        self.row_gap = int(row_gap)
        self.padding = int(padding)
        self.max_visible_rows = int(max_visible_rows)
        self.scroll_offset = min(self.scroll_offset, self.max_scroll_offset)
        self._build_item_rects()

    def set_selected_by_value(self, value):
        if value in self.items:
            self.selected = self.items.index(value)
            self._ensure_selection_visible()
            self._build_item_rects()
        elif not self.items:
            self.selected = None

    def _ensure_selection_visible(self):
        if self.selected is None:
            return
        if self.selected < self.scroll_offset:
            self.scroll_offset = self.selected
        elif self.selected >= self.scroll_offset + self.visible_capacity:
            self.scroll_offset = self.selected - self.visible_capacity + 1

    def scroll(self, delta):
        self.scroll_offset = max(0, min(self.max_scroll_offset, self.scroll_offset + delta))
        self._build_item_rects()

    def handle_event(self, event):
        mouse_pos = pygame.mouse.get_pos()
        if event.type == pygame.MOUSEBUTTONDOWN and self.rect.collidepoint(mouse_pos):
            if event.button == 4:
                self.scroll(-1)
                return None
            if event.button == 5:
                self.scroll(1)
                return None

        if event.type == pygame.MOUSEWHEEL and self.rect.collidepoint(mouse_pos):
            self.scroll(-int(event.y))
            return None

        if event.type == pygame.MOUSEMOTION:
            self.hovered = None
            for visible_idx, item_rect in enumerate(self.item_rects):
                if item_rect.collidepoint(event.pos):
                    self.hovered = self.scroll_offset + visible_idx
                    break

        if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            for visible_idx, item_rect in enumerate(self.item_rects):
                if item_rect.collidepoint(event.pos):
                    item_idx = self.scroll_offset + visible_idx
                    self.selected = item_idx
                    self._ensure_selection_visible()
                    self._build_item_rects()
                    return self.items[item_idx]
        return None

    def draw(self, screen):
        pygame.draw.rect(screen, self.bg_color, self.rect, border_radius=12)
        pygame.draw.rect(screen, self.border_color, self.rect, 1, border_radius=12)

        previous_clip = screen.get_clip()
        screen.set_clip(self.rect)
        for visible_idx, item_rect in enumerate(self.item_rects):
            item_idx = self.scroll_offset + visible_idx
            if item_idx == self.selected:
                bg = self.selected_bg_color
                border = self.selected_border_color
                text_color = self.selected_text_color
                border_width = 2
            elif item_idx == self.hovered:
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

            text_surf = self.font.render(self.items[item_idx], True, text_color)
            screen.blit(
                text_surf,
                (
                    item_rect.x + 12,
                    item_rect.y + (item_rect.height - text_surf.get_height()) // 2,
                ),
            )
        screen.set_clip(previous_clip)

        if self.max_scroll_offset > 0:
            track_w = 4
            track_rect = pygame.Rect(
                self.rect.right - self.padding,
                self.rect.y + self.padding,
                track_w,
                self.rect.height - 2 * self.padding,
            )
            pygame.draw.rect(screen, (220, 225, 232), track_rect, border_radius=track_w // 2)
            thumb_h = max(20, int(track_rect.height * (self.visible_capacity / max(1, len(self.items)))))
            thumb_y = track_rect.y + int(
                (track_rect.height - thumb_h) * (self.scroll_offset / max(1, self.max_scroll_offset))
            )
            thumb_rect = pygame.Rect(track_rect.x, thumb_y, track_rect.width, thumb_h)
            pygame.draw.rect(screen, (178, 186, 197), thumb_rect, border_radius=track_w // 2)

    @property
    def selected_item(self):
        if self.selected is None or self.selected >= len(self.items):
            return None
        return self.items[self.selected]

    @property
    def visible_capacity(self):
        available_h = max(0, self.rect.height - 2 * self.padding)
        per_row = max(1, self.row_height + self.row_gap)
        capacity = max(1, (available_h + self.row_gap) // per_row)
        return min(self.max_visible_rows, capacity)

    @property
    def max_scroll_offset(self):
        return max(0, len(self.items) - self.visible_capacity)

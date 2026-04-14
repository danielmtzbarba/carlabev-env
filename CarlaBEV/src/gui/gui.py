import math
import pygame
import numpy as np

from CarlaBEV.src.gui.components import Button, Selector, TextBox, ListBox
from CarlaBEV.src.gui.settings import Settings as cfg
from CarlaBEV.src.scenes.scenarios.specs import get_scenario_spec, list_scenario_ids


class GUI:
    def __init__(self):
        pygame.display.set_caption("Traffic Scenario Designer")
        display = pygame.display.Info()
        window_width = min(cfg.width, max(1080, display.current_w - 80))
        window_height = min(cfg.height, max(680, display.current_h - 80))
        self.screen = pygame.display.set_mode((window_width, window_height))

        self.font = pygame.font.SysFont(None, 22)
        self.font_small = pygame.font.SysFont(None, 18)
        self.font_title = pygame.font.SysFont(None, 34)
        self.font_section = pygame.font.SysFont(None, 24)

        self.colors = {
            "app_bg": (232, 235, 239),
            "panel_bg": (244, 246, 248),
            "card_bg": (255, 255, 255),
            "canvas_bg": (210, 214, 219),
            "canvas_frame": (185, 190, 197),
            "border": (182, 188, 196),
            "text": (28, 31, 36),
            "muted": (96, 104, 117),
            "accent": (29, 110, 235),
            "accent_soft": (227, 238, 255),
            "success": (41, 128, 64),
            "success_soft": (229, 242, 232),
        }

        self.left_panel_rect = pygame.Rect(0, 0, 320, self.screen.get_height())
        self.right_panel_rect = pygame.Rect(self.screen.get_width() - 260, 0, 260, self.screen.get_height())
        self.map_rect = pygame.Rect(0, 0, 100, 100)
        self._map_surface_size = (128, 128)
        self._map_crop_rect = pygame.Rect(0, 0, 128, 128)
        self._layout_ready = False

        self.current_frame = 0
        self.total_frames = 200
        self.status_lines = ["Click the map to place the scenario anchor."]

        self.scene_name = TextBox((0, 0, 100, 36), self.font, "Scene1")
        self.scenario_selector = Selector((0, 0, 100, 100), self.font, list_scenario_ids())
        self.level_selector = Selector((0, 0, 100, 100), self.font, ["Level 1", "Level 2", "Level 3", "Level 4"])
        self.field_boxes = {}
        self.active_fields = []
        self.sync_scenario_form()

        self.anchor_btn = Button((0, 0, 120, 34), self.font, "Anchor On Map")
        self.del_btn = Button((0, 0, 120, 34), self.font, "Delete Actor")
        self.listbox = ListBox((0, 0, 100, 100), self.font)
        self.save_btn = Button((0, 0, 120, 34), self.font, "Save Config")
        self.add_rdm_actor_btn = Button((0, 0, 120, 34), self.font, "Random Scene")
        self.play_btn = Button((0, 0, 120, 34), self.font, "Play Preview")

        self.fov_rect = pygame.Rect(0, 0, 180, 180)
        self.timeline_rect = pygame.Rect(0, 0, 100, 14)
        self.section_rects = {}

    def handle_timeline_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and self.timeline_rect.collidepoint(event.pos):
            self.update_frame_from_mouse(event.pos[0])
        elif event.type == pygame.MOUSEMOTION and event.buttons[0]:
            if self.timeline_rect.collidepoint(event.pos):
                self.update_frame_from_mouse(event.pos[0])

    def handle_event(self, event):
        self.scene_name.handle_event(event)

        prev_scenario = self.scenario_selector.selected
        self.scenario_selector.handle_event(event)
        if self.scenario_selector.selected != prev_scenario:
            self.sync_scenario_form()

        self.level_selector.handle_event(event)
        for box in self.field_boxes.values():
            box.handle_event(event)
        self.handle_timeline_event(event)

        ignored_click = False
        if event.type == pygame.MOUSEBUTTONUP:
            ignored_click = any(
                rect.collidepoint(event.pos)
                for rect in [
                    self.scene_name.rect,
                    self.scenario_selector.rect,
                    self.level_selector.rect,
                    self.save_btn.rect,
                    self.add_rdm_actor_btn.rect,
                    self.play_btn.rect,
                ]
            ) or any(box.rect.collidepoint(event.pos) for box in self.field_boxes.values())

        if event.type == pygame.MOUSEBUTTONUP and not ignored_click and self.map_rect.collidepoint(event.pos):
            return {"action": "anchor", "pos": event.pos}

        if self.add_rdm_actor_btn.handle_event(event):
            self.env.reset(options={"scene": "rdm"})
            self.set_status("Loaded a random traffic scene.", "Choose a scenario and click the map to return to authored mode.")
            return "rdm"

        if self.play_btn.handle_event(event):
            if not self.play_mode:
                self.play_btn.text = "Stop Preview"
                self.play_scene()
                return True
            self.play_btn.text = "Play Preview"
            self.toggle_play_mode()

        if self.save_btn.handle_event(event):
            self.save_scene(self.scene_name.text)

        return None

    def sync_scenario_form(self):
        spec = self.scenario_spec
        self.level_selector.update_options(spec.level_options())
        self.active_fields = list(spec.fields)

        new_boxes = {}
        for field in self.active_fields:
            existing = self.field_boxes.get(field.key)
            text = existing.text if existing is not None else field.default_text()
            new_boxes[field.key] = TextBox((0, 0, 100, 34), self.font, text)
        self.field_boxes = new_boxes

        if self._layout_ready:
            self.update_layout(self._map_surface_size)

    def get_form_values(self):
        values = {}
        for field in self.active_fields:
            values[field.key] = field.parse(self.field_boxes[field.key].text)
        return values

    def update_layout(self, map_surface_size):
        self._map_surface_size = map_surface_size
        screen_w = self.screen.get_width()
        screen_h = self.screen.get_height()

        left_w = min(360, max(300, screen_w // 3))
        right_w = min(250, max(220, screen_w // 5))
        center_min_w = 360
        if left_w + right_w + center_min_w > screen_w:
            overflow = left_w + right_w + center_min_w - screen_w
            left_w -= min(overflow, max(0, left_w - 280))
            overflow = max(0, left_w + right_w + center_min_w - screen_w)
            right_w -= min(overflow, max(0, right_w - 200))

        self.left_panel_rect = pygame.Rect(0, 0, left_w, screen_h)
        self.right_panel_rect = pygame.Rect(screen_w - right_w, 0, right_w, screen_h)

        center_x = self.left_panel_rect.right + 18
        center_w = self.right_panel_rect.left - center_x - 18
        center_y = 22
        center_h = screen_h - 110

        crop_w, crop_h = self._map_crop_rect.size
        map_w, map_h = crop_w, crop_h
        scale = min(center_w / map_w, center_h / map_h)
        draw_w = max(1, int(map_w * scale))
        draw_h = max(1, int(map_h * scale))
        map_x = center_x + max(0, (center_w - draw_w) // 2)
        map_y = center_y + max(0, (center_h - draw_h) // 2)
        self.map_rect = pygame.Rect(map_x, map_y, draw_w, draw_h)

        fov_size = min(self.right_panel_rect.width - 36, 190)
        self.fov_rect = pygame.Rect(
            self.right_panel_rect.x + (self.right_panel_rect.width - fov_size) // 2,
            96,
            fov_size,
            fov_size,
        )
        self.timeline_rect = pygame.Rect(self.map_rect.x, self.map_rect.bottom + 22, self.map_rect.width, 12)

        self.layout_controls()
        self._layout_ready = True

    def layout_controls(self):
        left_x = self.left_panel_rect.x + 24
        left_w = self.left_panel_rect.width - 48
        y = 96

        self.scene_name.rect = pygame.Rect(left_x, y, left_w, 40)
        self.section_rects["scene_name"] = pygame.Rect(left_x - 12, y - 28, left_w + 24, 86)
        y += 66

        self.scenario_selector.rect = pygame.Rect(left_x, y, left_w, 102)
        self.scenario_selector._build_option_rects()
        self.section_rects["scenario"] = pygame.Rect(left_x - 12, y - 28, left_w + 24, 148)
        y += 142

        self.level_selector.rect = pygame.Rect(left_x, y, left_w, 102)
        self.level_selector._build_option_rects()
        self.section_rects["difficulty"] = pygame.Rect(left_x - 12, y - 28, left_w + 24, 182)
        y += 142

        gap = 14
        col_count = 2 if len(self.active_fields) > 3 else 1
        field_w = left_w if col_count == 1 else (left_w - gap) // 2
        row_h = 52
        for idx, field in enumerate(self.active_fields):
            col = idx % col_count
            row = idx // col_count
            box_x = left_x + col * (field_w + gap)
            box_y = y + row * row_h + 18
            self.field_boxes[field.key].rect = pygame.Rect(box_x, box_y, field_w, 34)

        rows = math.ceil(len(self.active_fields) / col_count) if self.active_fields else 0
        params_h = max(92, rows * row_h + 46)
        self.section_rects["parameters"] = pygame.Rect(left_x - 12, y - 28, left_w + 24, params_h)
        y += rows * row_h + 28

        button_w = (left_w - 12) // 2
        self.save_btn.rect = pygame.Rect(left_x, y, button_w, 38)
        self.anchor_btn.rect = pygame.Rect(left_x + button_w + 12, y, button_w, 38)
        self.section_rects["actions"] = pygame.Rect(left_x - 12, y - 18, left_w + 24, 74)

        right_x = self.right_panel_rect.x + 18
        right_w = self.right_panel_rect.width - 36
        action_y = self.fov_rect.bottom + 26
        self.add_rdm_actor_btn.rect = pygame.Rect(right_x, action_y, right_w, 38)
        self.play_btn.rect = pygame.Rect(right_x, action_y + 50, right_w, 38)

    def screen_to_map_pos(self, pos):
        if self.map_rect.width <= 0 or self.map_rect.height <= 0:
            return None
        rel_x = (pos[0] - self.map_rect.x) / self.map_rect.width
        rel_y = (pos[1] - self.map_rect.y) / self.map_rect.height
        map_x = int(self._map_crop_rect.x + rel_x * self._map_crop_rect.width)
        map_y = int(self._map_crop_rect.y + rel_y * self._map_crop_rect.height)
        map_x = max(0, min(self._map_surface_size[0] - 1, map_x))
        map_y = max(0, min(self._map_surface_size[1] - 1, map_y))
        return map_x, map_y

    def set_status(self, *lines):
        self.status_lines = list(lines)

    def draw_fov(self):
        self._draw_card(self.right_panel_rect.x + 12, 48, self.right_panel_rect.width - 24, self.fov_rect.height + 64)
        title = self.font_section.render("FOV Preview", True, self.colors["text"])
        self.screen.blit(title, (self.right_panel_rect.x + 24, 60))

        vehicle_surface = self.env.map.canvas
        pygame.draw.rect(self.screen, self.colors["border"], self.fov_rect.inflate(2, 2), width=1, border_radius=12)
        if vehicle_surface is not None:
            scaled = pygame.transform.scale(vehicle_surface, (self.fov_rect.width, self.fov_rect.height))
            self.screen.blit(scaled, self.fov_rect.topleft)
        else:
            pygame.draw.rect(self.screen, (18, 20, 24), self.fov_rect, border_radius=12)
            text = self.font.render("No FOV", True, (220, 223, 228))
            self.screen.blit(text, text.get_rect(center=self.fov_rect.center))

    def draw_timeline(self):
        pygame.draw.rect(self.screen, (208, 214, 220), self.timeline_rect, border_radius=10)
        if self.total_frames <= 0:
            return
        handle_x = self.timeline_rect.x + int((self.current_frame / self.total_frames) * self.timeline_rect.width)
        handle_x = max(self.timeline_rect.x + 6, min(self.timeline_rect.right - 6, handle_x))
        pygame.draw.circle(self.screen, self.colors["accent"], (handle_x, self.timeline_rect.centery), 8)

    def draw_gui(self):
        map_surface = self.env.map.map_surface
        self._map_crop_rect = self._compute_map_crop_rect(map_surface)
        self.update_layout(map_surface.get_size())
        self.screen.fill(self.colors["app_bg"])

        self._draw_side_panels()
        self._draw_left_panel()
        self._draw_center_canvas()
        self.draw_fov()
        self._draw_right_panel()

    def _draw_side_panels(self):
        pygame.draw.rect(self.screen, self.colors["panel_bg"], self.left_panel_rect)
        pygame.draw.rect(self.screen, self.colors["panel_bg"], self.right_panel_rect)
        pygame.draw.line(self.screen, self.colors["border"], self.left_panel_rect.topright, self.left_panel_rect.bottomright, 1)
        pygame.draw.line(self.screen, self.colors["border"], self.right_panel_rect.topleft, self.right_panel_rect.bottomleft, 1)

    def _draw_left_panel(self):
        self.screen.blit(self.font_title.render("Scenario Designer", True, self.colors["text"]), (24, 22))
        self.screen.blit(
            self.font_small.render("Configure a scenario, then place it on the map.", True, self.colors["muted"]),
            (24, 52),
        )

        self._draw_soft_section(self.section_rects["scene_name"])
        self._draw_label("Scene Name", self.scene_name.rect.x, self.scene_name.rect.y - 18)
        self.scene_name.draw(self.screen)

        self._draw_soft_section(self.section_rects["scenario"])
        self._draw_section_header("Scenario", self.scenario_selector.rect.x, self.scenario_selector.rect.y - 18)
        self.scenario_selector.draw(self.screen)

        self._draw_soft_section(self.section_rects["difficulty"])
        self._draw_section_header("Difficulty", self.level_selector.rect.x, self.level_selector.rect.y - 18)
        self.level_selector.draw(self.screen)

        if self.active_fields:
            self._draw_soft_section(self.section_rects["parameters"])
            first_field = self.field_boxes[self.active_fields[0].key].rect
            self._draw_section_header("Parameters", first_field.x, first_field.y - 34)

        for field in self.active_fields:
            box = self.field_boxes[field.key]
            self._draw_label(field.label, box.rect.x, box.rect.y - 16, small=True)
            box.draw(self.screen)

        self._draw_soft_section(self.section_rects["actions"])
        self.save_btn.draw(self.screen)
        self._draw_disabled_button(self.anchor_btn.rect, self.anchor_btn.text)

    def _draw_center_canvas(self):
        frame_rect = self.map_rect.inflate(26, 26)
        pygame.draw.rect(self.screen, self.colors["canvas_frame"], frame_rect, border_radius=22)
        pygame.draw.rect(self.screen, self.colors["canvas_bg"], frame_rect.inflate(-2, -2), border_radius=20)

        cropped_map = self.env.map.map_surface.subsurface(self._map_crop_rect)
        scaled_map = pygame.transform.scale(cropped_map, self.map_rect.size)
        self.screen.blit(scaled_map, self.map_rect.topleft)

        caption = self.font_small.render("Map canvas", True, self.colors["muted"])
        self.screen.blit(caption, (frame_rect.x + 14, frame_rect.y + 10))

        help_text = self.font_small.render("Click anywhere on the map to place the scenario anchor.", True, self.colors["muted"])
        self.screen.blit(help_text, (self.map_rect.x, self.timeline_rect.y - 24))
        self.draw_timeline()

    def _draw_right_panel(self):
        self.screen.blit(self.font_section.render("Preview", True, self.colors["text"]), (self.right_panel_rect.x + 18, 20))
        self.screen.blit(
            self.font_small.render("Run, inspect, and save the current setup.", True, self.colors["muted"]),
            (self.right_panel_rect.x + 18, 42),
        )

        self.add_rdm_actor_btn.draw(self.screen)
        self.play_btn.draw(self.screen)

        summary_y = self.play_btn.rect.bottom + 24
        self._draw_card(self.right_panel_rect.x + 12, summary_y, self.right_panel_rect.width - 24, 156)
        self.screen.blit(self.font_section.render("Current Setup", True, self.colors["text"]), (self.right_panel_rect.x + 24, summary_y + 14))

        summary_lines = [
            f"Scenario: {self.scenario_spec.display_name}",
            f"Level: {self.level_selector.selection}",
            f"Fields: {len(self.active_fields)}",
        ]
        for idx, line in enumerate(summary_lines):
            self.screen.blit(
                self.font.render(line, True, self.colors["text"]),
                (self.right_panel_rect.x + 24, summary_y + 46 + idx * 24),
            )

        status_y = summary_y + 174
        self._draw_card(self.right_panel_rect.x + 12, status_y, self.right_panel_rect.width - 24, 126)
        self.screen.blit(self.font_section.render("Status", True, self.colors["text"]), (self.right_panel_rect.x + 24, status_y + 14))
        for idx, line in enumerate(self.status_lines[:3]):
            self._draw_wrapped_text(
                line,
                pygame.Rect(self.right_panel_rect.x + 24, status_y + 42 + idx * 24, self.right_panel_rect.width - 48, 28),
                self.font_small,
                self.colors["muted"],
            )

    def update_frame_from_mouse(self, mouse_x):
        rel_x = max(0, min(mouse_x - self.timeline_rect.x, self.timeline_rect.width))
        self.current_frame = int((rel_x / max(1, self.timeline_rect.width)) * self.total_frames)

    def _draw_card(self, x, y, width, height):
        rect = pygame.Rect(x, y, width, height)
        pygame.draw.rect(self.screen, self.colors["card_bg"], rect, border_radius=18)
        pygame.draw.rect(self.screen, self.colors["border"], rect, width=1, border_radius=18)

    def _draw_soft_section(self, rect):
        pygame.draw.rect(self.screen, (250, 251, 252), rect, border_radius=18)
        pygame.draw.rect(self.screen, (214, 219, 226), rect, width=1, border_radius=18)

    def _draw_label(self, text, x, y, small=False):
        font = self.font_small if small else self.font
        self.screen.blit(font.render(text, True, self.colors["muted"]), (x, y))

    def _draw_section_header(self, text, x, y):
        self.screen.blit(self.font_section.render(text, True, self.colors["text"]), (x, y))

    def _draw_disabled_button(self, rect, text):
        pygame.draw.rect(self.screen, self.colors["success_soft"], rect, border_radius=12)
        pygame.draw.rect(self.screen, self.colors["border"], rect, width=1, border_radius=12)
        label = self.font.render(text, True, self.colors["success"])
        self.screen.blit(label, label.get_rect(center=rect.center))

    def _draw_wrapped_text(self, text, rect, font, color):
        words = text.split()
        lines = []
        current = ""
        for word in words:
            probe = word if not current else f"{current} {word}"
            if font.size(probe)[0] <= rect.width:
                current = probe
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)

        for idx, line in enumerate(lines[:3]):
            surf = font.render(line, True, color)
            self.screen.blit(surf, (rect.x, rect.y + idx * (font.get_height() + 2)))

    def _compute_map_crop_rect(self, surface):
        arr = pygame.surfarray.array3d(surface)
        bg = arr[0, 0].astype(np.int16)
        diff = np.abs(arr.astype(np.int16) - bg).sum(axis=2)
        mask = diff > 12

        coords = np.argwhere(mask)
        if coords.size == 0:
            return pygame.Rect(0, 0, surface.get_width(), surface.get_height())

        xs = coords[:, 0]
        ys = coords[:, 1]
        pad = 12
        xmin = max(0, int(xs.min()) - pad)
        xmax = min(surface.get_width() - 1, int(xs.max()) + pad)
        ymin = max(0, int(ys.min()) - pad)
        ymax = min(surface.get_height() - 1, int(ys.max()) + pad)
        return pygame.Rect(xmin, ymin, max(1, xmax - xmin + 1), max(1, ymax - ymin + 1))

    @property
    def scenario_spec(self):
        return get_scenario_spec(self.scenario_selector.selection)


def np_to_surface(arr):
    return pygame.surfarray.make_surface(np.transpose(arr, (1, 0, 2)))

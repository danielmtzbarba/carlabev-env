import math
import pygame
import numpy as np

from CarlaBEV.src.gui.components import Button, Selector, TextBox, ListBox
from CarlaBEV.src.gui.settings import Settings
from CarlaBEV.src.scenes.scenarios.specs import get_scenario_spec, list_scenario_ids


class GUI:
    def __init__(self, settings=None):
        self.settings = settings or Settings()
        pygame.display.set_caption("Traffic Scenario Designer")
        self.layout_cfg = self.settings.designer_layout
        display = pygame.display.Info()
        self.min_window_size = (
            self.layout_cfg.min_window_width,
            self.layout_cfg.min_window_height,
        )
        window_width, window_height = self._compute_window_size(display)
        self.screen = pygame.display.set_mode(
            (window_width, window_height),
            pygame.RESIZABLE,
        )
        self._apply_responsive_metrics(window_width, window_height)

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
        self.left_body_rect = pygame.Rect(0, 0, 0, 0)
        self.map_rect = pygame.Rect(0, 0, 100, 100)
        self._map_surface_size = (128, 128)
        self._map_crop_rect = pygame.Rect(0, 0, 128, 128)
        self._layout_ready = False
        self.left_scroll_offset = 0
        self.left_content_height = 0

        self.current_frame = 0
        self.total_frames = 200
        self.status_lines = ["Click the map to place the scenario anchor."]
        self._last_window_size = self.screen.get_size()
        self.map_click_mode = None
        self.pending_actor_type = None
        self.pending_actor_start = None
        self.designed_actors = []
        self.selected_actor_index = None
        self.actor_row_rects = []

        self.scene_name = TextBox((0, 0, 100, 36), self.font, "Scene1")
        self.scenario_selector = Selector((0, 0, 100, 100), self.font, list_scenario_ids())
        self.level_selector = Selector((0, 0, 100, 100), self.font, ["Level 1", "Level 2", "Level 3", "Level 4"])
        self.field_boxes = {}
        self.active_fields = []
        self.sync_scenario_form()

        self.anchor_btn = Button((0, 0, 120, 34), self.font, "Anchor On Map")
        self.del_btn = Button((0, 0, 120, 34), self.font, "Delete Actor")
        self.place_ego_btn = Button((0, 0, 120, 34), self.font, "Place Ego")
        self.place_vehicle_btn = Button((0, 0, 120, 34), self.font, "Place Vehicle")
        self.place_ped_btn = Button((0, 0, 120, 34), self.font, "Place Ped")
        self.listbox = ListBox((0, 0, 100, 100), self.font)
        self.save_btn = Button((0, 0, 120, 34), self.font, "Save Config")
        self.add_rdm_actor_btn = Button((0, 0, 120, 34), self.font, "Random Scene")
        self.play_btn = Button((0, 0, 120, 34), self.font, "Play Preview")

        self.fov_rect = pygame.Rect(0, 0, 180, 180)
        self.timeline_rect = pygame.Rect(0, 0, 100, 14)
        self.section_rects = {}

    def _compute_window_size(self, display):
        screen_w = max(display.current_w, self.min_window_size[0])
        screen_h = max(display.current_h, self.min_window_size[1])
        window_width = int(screen_w * self.layout_cfg.window_width_ratio)
        window_height = int(screen_h * self.layout_cfg.window_height_ratio)
        window_width = max(
            self.min_window_size[0],
            min(window_width, screen_w - self.layout_cfg.window_margin_x),
        )
        window_height = max(
            self.min_window_size[1],
            min(window_height, screen_h - self.layout_cfg.window_margin_y),
        )
        return window_width, window_height

    def _refresh_window_metrics(self):
        surface = pygame.display.get_surface()
        if surface is None:
            return
        current_size = surface.get_size()
        if current_size != self._last_window_size:
            self.screen = surface
            self._last_window_size = current_size
            self._apply_responsive_metrics(*current_size)
            self.update_layout(self._map_surface_size)

    def _apply_responsive_metrics(self, screen_w, screen_h):
        scale = min(
            screen_w / self.layout_cfg.scale_ref_width,
            screen_h / self.layout_cfg.scale_ref_height,
        )
        self.ui_scale = max(self.layout_cfg.scale_min, min(scale, self.layout_cfg.scale_max))

        self.spacing_xs = max(4, int(round(self.layout_cfg.spacing_xs_base * self.ui_scale)))
        self.spacing_sm = max(8, int(round(self.layout_cfg.spacing_sm_base * self.ui_scale)))
        self.spacing_md = max(12, int(round(self.layout_cfg.spacing_md_base * self.ui_scale)))
        self.spacing_lg = max(16, int(round(self.layout_cfg.spacing_lg_base * self.ui_scale)))
        self.panel_padding = max(18, int(round(self.layout_cfg.panel_padding_base * self.ui_scale)))
        self.section_pad = max(10, int(round(self.layout_cfg.section_padding_base * self.ui_scale)))
        self.option_height = max(28, int(round(self.layout_cfg.option_height_base * self.ui_scale)))
        self.option_gap = max(4, int(round(self.layout_cfg.option_gap_base * self.ui_scale)))
        self.textbox_height = max(36, int(round(self.layout_cfg.textbox_height_base * self.ui_scale)))
        self.button_height = max(36, int(round(self.layout_cfg.button_height_base * self.ui_scale)))
        self.timeline_height = max(10, int(round(self.layout_cfg.timeline_height_base * self.ui_scale)))
        self.section_radius = max(14, int(round(self.layout_cfg.section_radius_base * self.ui_scale)))
        self.card_radius = max(14, int(round(self.layout_cfg.card_radius_base * self.ui_scale)))
        self.selector_title_gap = max(14, int(round(self.layout_cfg.section_title_gap_base * self.ui_scale)))
        self.param_label_gap = max(12, int(round(self.layout_cfg.param_label_gap_base * self.ui_scale)))
        self.frame_pad = max(16, int(round(self.layout_cfg.frame_padding_base * self.ui_scale)))
        self.header_x = self.panel_padding
        self.header_y = self.panel_padding
        self.header_gap = max(10, int(round(self.layout_cfg.header_gap_base * self.ui_scale)))
        self.timeline_handle_radius = max(6, int(round(self.layout_cfg.timeline_handle_radius_base * self.ui_scale)))
        self.button_radius = max(10, int(round(self.layout_cfg.button_radius_base * self.ui_scale)))

        self.font = pygame.font.SysFont(
            None,
            max(self.layout_cfg.font_body_min, int(round(self.layout_cfg.font_body_base * self.ui_scale))),
        )
        self.font_small = pygame.font.SysFont(
            None,
            max(self.layout_cfg.font_small_min, int(round(self.layout_cfg.font_small_base * self.ui_scale))),
        )
        self.font_title = pygame.font.SysFont(
            None,
            max(self.layout_cfg.font_title_min, int(round(self.layout_cfg.font_title_base * self.ui_scale))),
        )
        self.font_section = pygame.font.SysFont(
            None,
            max(self.layout_cfg.font_section_min, int(round(self.layout_cfg.font_section_base * self.ui_scale))),
        )

        for widget in (
            "scene_name",
            "save_btn",
            "anchor_btn",
            "add_rdm_actor_btn",
            "play_btn",
            "del_btn",
            "place_ego_btn",
            "place_vehicle_btn",
            "place_ped_btn",
        ):
            if hasattr(self, widget):
                getattr(self, widget).font = self.font
        for widget in ("scenario_selector", "level_selector"):
            if hasattr(self, widget):
                selector = getattr(self, widget)
                selector.font = self.font
                selector.set_metrics(self.option_height, self.option_gap)
        if hasattr(self, "field_boxes"):
            for box in self.field_boxes.values():
                box.font = self.font

    def handle_timeline_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and self.timeline_rect.collidepoint(event.pos):
            self.update_frame_from_mouse(event.pos[0])
        elif event.type == pygame.MOUSEMOTION and event.buttons[0]:
            if self.timeline_rect.collidepoint(event.pos):
                self.update_frame_from_mouse(event.pos[0])

    def handle_event(self, event):
        if event.type == pygame.VIDEORESIZE:
            surface = pygame.display.get_surface()
            if surface is None:
                return None
            new_w = max(self.min_window_size[0], surface.get_width())
            new_h = max(self.min_window_size[1], surface.get_height())
            self.screen = surface
            self._last_window_size = surface.get_size()
            self._apply_responsive_metrics(new_w, new_h)
            self.update_layout(self._map_surface_size)
            return None

        self.scene_name.handle_event(event)

        prev_scenario = self.scenario_selector.selected
        self.scenario_selector.handle_event(event)
        if self.scenario_selector.selected != prev_scenario:
            self.sync_scenario_form()

        self.level_selector.handle_event(event)
        for box in self.field_boxes.values():
            box.handle_event(event)
        self.handle_timeline_event(event)

        if event.type == pygame.MOUSEBUTTONUP:
            for idx, row_rect in enumerate(self.actor_row_rects):
                if row_rect.collidepoint(event.pos):
                    self.selected_actor_index = idx
                    return None

        ignored_click = False
        if event.type == pygame.MOUSEBUTTONUP:
            ignored_click = any(
                rect.collidepoint(event.pos)
                for rect in [
                    self.scene_name.rect,
                    self.scenario_selector.rect,
                    self.level_selector.rect,
                    self.save_btn.rect,
                    self.anchor_btn.rect,
                    self.del_btn.rect,
                    self.place_ego_btn.rect,
                    self.place_vehicle_btn.rect,
                    self.place_ped_btn.rect,
                    self.add_rdm_actor_btn.rect,
                    self.play_btn.rect,
                ]
            ) or any(box.rect.collidepoint(event.pos) for box in self.field_boxes.values()) or any(
                rect.collidepoint(event.pos) for rect in self.actor_row_rects
            )

        if event.type == pygame.MOUSEBUTTONUP and not ignored_click and self.map_rect.collidepoint(event.pos):
            if self.map_click_mode == "anchor":
                self.map_click_mode = None
                return {"action": "anchor", "pos": event.pos}
            if self.map_click_mode == "actor_start":
                self.pending_actor_start = event.pos
                self.map_click_mode = "actor_end"
                self.set_status("Select the actor end position.", "Second click completes the route.")
                return None
            if self.map_click_mode == "actor_end" and self.pending_actor_start is not None:
                start_pos = self.pending_actor_start
                actor_type = self.pending_actor_type
                self.pending_actor_start = None
                self.pending_actor_type = None
                self.map_click_mode = None
                return {
                    "action": "add_actor",
                    "actor_type": actor_type,
                    "start_pos": start_pos,
                    "end_pos": event.pos,
                }

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

        if self.anchor_btn.handle_event(event):
            self.map_click_mode = "anchor"
            self.pending_actor_start = None
            self.pending_actor_type = None
            self.set_status("Click the map to place the scenario anchor.", "Anchor placement is active.")
            return None

        if self.place_ego_btn.handle_event(event):
            self.map_click_mode = "actor_start"
            self.pending_actor_type = "agent"
            self.pending_actor_start = None
            self.set_status("Click the ego start position.", "Second click sets the ego end position.")
            return None

        if self.place_vehicle_btn.handle_event(event):
            self.map_click_mode = "actor_start"
            self.pending_actor_type = "vehicle"
            self.pending_actor_start = None
            self.set_status("Click the vehicle start position.", "Second click sets the vehicle end position.")
            return None

        if self.place_ped_btn.handle_event(event):
            self.map_click_mode = "actor_start"
            self.pending_actor_type = "pedestrian"
            self.pending_actor_start = None
            self.set_status("Click the pedestrian start position.", "Second click sets the pedestrian end position.")
            return None

        if self.del_btn.handle_event(event):
            if self.selected_actor_index is not None:
                return {"action": "delete_actor", "index": self.selected_actor_index}
            self.set_status("No actor selected.", "Select an actor row before deleting it.")
            return None

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

        left_w = min(
            int(self.layout_cfg.left_panel_width_max * self.ui_scale),
            max(int(self.layout_cfg.left_panel_width_min * self.ui_scale), screen_w // 3),
        )
        right_w = min(
            int(self.layout_cfg.right_panel_width_max * self.ui_scale),
            max(int(self.layout_cfg.right_panel_width_min * self.ui_scale), screen_w // 5),
        )
        center_min_w = self.layout_cfg.center_panel_min_width
        if left_w + right_w + center_min_w > screen_w:
            overflow = left_w + right_w + center_min_w - screen_w
            left_w -= min(
                overflow,
                max(0, left_w - int(self.layout_cfg.left_panel_floor * self.ui_scale)),
            )
            overflow = max(0, left_w + right_w + center_min_w - screen_w)
            right_w -= min(
                overflow,
                max(0, right_w - int(self.layout_cfg.right_panel_floor * self.ui_scale)),
            )

        self.left_panel_rect = pygame.Rect(0, 0, left_w, screen_h)
        self.right_panel_rect = pygame.Rect(screen_w - right_w, 0, right_w, screen_h)
        header_title_y = self.header_y
        header_subtitle_y = header_title_y + self.font_title.get_height() + self.spacing_xs
        left_header_h = (
            header_subtitle_y
            + self.font_small.get_height()
            + self.spacing_lg
        )
        self.left_header_title_pos = (self.left_panel_rect.x + self.header_x, header_title_y)
        self.left_header_subtitle_pos = (self.left_panel_rect.x + self.header_x, header_subtitle_y)
        self.right_header_title_pos = (self.right_panel_rect.x + self.header_x, header_title_y)
        self.right_header_subtitle_pos = (self.right_panel_rect.x + self.header_x, header_subtitle_y)
        self.left_body_rect = pygame.Rect(
            self.left_panel_rect.x,
            self.left_panel_rect.y + left_header_h,
            self.left_panel_rect.width,
            self.left_panel_rect.height - left_header_h,
        )
        self.right_body_rect = pygame.Rect(
            self.right_panel_rect.x,
            self.right_panel_rect.y + left_header_h,
            self.right_panel_rect.width,
            self.right_panel_rect.height - left_header_h,
        )

        center_x = self.left_panel_rect.right + self.spacing_lg
        center_w = self.right_panel_rect.left - center_x - self.spacing_lg
        center_y = self.spacing_lg + self.spacing_xs
        center_h = screen_h - max(
            self.layout_cfg.center_bottom_reserve_min,
            int(self.layout_cfg.center_bottom_reserve * self.ui_scale),
        )

        crop_w, crop_h = self._map_crop_rect.size
        map_w, map_h = crop_w, crop_h
        scale = min(center_w / map_w, center_h / map_h)
        draw_w = max(1, int(map_w * scale))
        draw_h = max(1, int(map_h * scale))
        map_x = center_x + max(0, (center_w - draw_w) // 2)
        map_y = center_y + max(0, (center_h - draw_h) // 2)
        self.map_rect = pygame.Rect(map_x, map_y, draw_w, draw_h)

        fov_size = min(
            self.right_panel_rect.width - 2 * self.panel_padding,
            int(self.layout_cfg.preview_fov_max * self.ui_scale),
        )
        self.timeline_rect = pygame.Rect(
            self.map_rect.x,
            self.map_rect.bottom + self.spacing_lg,
            self.map_rect.width,
            self.timeline_height,
        )

        self.layout_controls()
        self._layout_ready = True

    def layout_controls(self):
        left_x = self.left_panel_rect.x + self.panel_padding
        left_w = self.left_panel_rect.width - 2 * self.panel_padding
        card_title_h = self.font_section.get_height()
        card_top_pad = self.section_pad
        card_bottom_pad = self.section_pad
        card_header_gap = self.spacing_sm
        field_label_gap = self.spacing_xs
        field_row_gap = self.spacing_md
        label_h = self.font_small.get_height()
        setup_option_height = max(22, self.option_height - 6)
        setup_option_gap = max(2, self.option_gap - 1)
        self.scenario_selector.set_metrics(setup_option_height, setup_option_gap)
        self.level_selector.set_metrics(setup_option_height, setup_option_gap)
        scenario_h = len(self.scenario_selector.options) * setup_option_height + max(0, len(self.scenario_selector.options) - 1) * setup_option_gap
        level_h = len(self.level_selector.options) * setup_option_height + max(0, len(self.level_selector.options) - 1) * setup_option_gap
        gap = self.spacing_md
        col_count = 2 if len(self.active_fields) > 3 else 1
        field_w = left_w if col_count == 1 else (left_w - gap) // 2
        row_h = label_h + field_label_gap + self.textbox_height + field_row_gap
        actor_row_h = self.font_small.get_height() + self.spacing_sm
        actor_list_rows = 4
        actor_list_h = actor_row_h * actor_list_rows + self.spacing_sm
        actor_button_gap = self.spacing_sm
        actor_button_w = (left_w - actor_button_gap) // 2
        setup_col_gap = self.spacing_sm
        setup_col_w = (left_w - setup_col_gap) // 2

        rows = math.ceil(len(self.active_fields) / col_count) if self.active_fields else 0
        params_content_h = 0
        if rows > 0:
            params_content_h = rows * (label_h + field_label_gap + self.textbox_height)
            params_content_h += max(0, rows - 1) * field_row_gap
        section_heights = {
            "scene_name": card_top_pad + card_title_h + card_header_gap + self.textbox_height + card_bottom_pad,
            "setup": (
                card_top_pad
                + card_title_h
                + card_header_gap
                + label_h
                + field_label_gap
                + max(scenario_h, level_h)
                + card_bottom_pad
            ),
            "parameters": card_top_pad + card_title_h + card_header_gap + params_content_h + card_bottom_pad,
            "actors": (
                card_top_pad
                + card_title_h
                + card_header_gap
                + 2 * self.button_height
                + actor_button_gap
                + self.spacing_md
                + actor_list_h
                + card_bottom_pad
            ),
            "actions": self.button_height + 2 * self.spacing_xs,
        }
        section_order = ["scene_name", "setup", "parameters", "actors", "actions"]
        natural_gap = self.spacing_lg
        minimum_gap = self.spacing_sm
        tight_gap = self.spacing_xs
        content_top = self.left_body_rect.y + self.spacing_sm
        content_bottom = self.left_body_rect.bottom - self.spacing_sm
        available_h = max(0, content_bottom - content_top)
        total_section_h = sum(section_heights[name] for name in section_order)
        gap_count = max(1, len(section_order) - 1)
        max_fit_gap = max(0, (available_h - total_section_h) // gap_count)
        if total_section_h + natural_gap * gap_count <= available_h:
            distributed_gap = max(natural_gap, max_fit_gap)
        elif total_section_h + minimum_gap * gap_count <= available_h:
            distributed_gap = max(minimum_gap, max_fit_gap)
        else:
            distributed_gap = max(tight_gap, max_fit_gap)

        used_h = total_section_h + distributed_gap * gap_count
        top_inset = max(0, (available_h - used_h) // 2)
        y = content_top + top_inset
        self.left_scroll_offset = 0

        self.section_rects["scene_name"] = pygame.Rect(
            left_x - self.section_pad,
            y,
            left_w + 2 * self.section_pad,
            section_heights["scene_name"],
        )
        self.scene_name.rect = pygame.Rect(
            left_x,
            self.section_rects["scene_name"].y + card_top_pad + card_title_h + card_header_gap,
            left_w,
            self.textbox_height,
        )
        y += section_heights["scene_name"] + distributed_gap

        self.section_rects["setup"] = pygame.Rect(
            left_x - self.section_pad,
            y,
            left_w + 2 * self.section_pad,
            section_heights["setup"],
        )
        setup_content_y = self.section_rects["setup"].y + card_top_pad + card_title_h + card_header_gap
        scenario_y = setup_content_y + label_h + field_label_gap
        self.scenario_selector.rect = pygame.Rect(
            left_x,
            scenario_y,
            setup_col_w,
            scenario_h,
        )
        self.scenario_selector._build_option_rects()
        level_y = setup_content_y + label_h + field_label_gap
        self.level_selector.rect = pygame.Rect(
            left_x + setup_col_w + setup_col_gap,
            level_y,
            setup_col_w,
            level_h,
        )
        self.level_selector._build_option_rects()
        y += section_heights["setup"] + distributed_gap

        self.section_rects["parameters"] = pygame.Rect(
            left_x - self.section_pad,
            y,
            left_w + 2 * self.section_pad,
            section_heights["parameters"],
        )
        params_y = self.section_rects["parameters"].y + card_top_pad + card_title_h + card_header_gap
        for idx, field in enumerate(self.active_fields):
            col = idx % col_count
            row = idx // col_count
            box_x = left_x + col * (field_w + gap)
            box_y = params_y + row * row_h + label_h + field_label_gap
            self.field_boxes[field.key].rect = pygame.Rect(box_x, box_y, field_w, self.textbox_height)
        y += section_heights["parameters"] + distributed_gap

        self.section_rects["actors"] = pygame.Rect(
            left_x - self.section_pad,
            y,
            left_w + 2 * self.section_pad,
            section_heights["actors"],
        )
        actor_buttons_y = self.section_rects["actors"].y + card_top_pad + card_title_h + card_header_gap
        self.place_ego_btn.rect = pygame.Rect(left_x, actor_buttons_y, actor_button_w, self.button_height)
        self.place_vehicle_btn.rect = pygame.Rect(
            left_x + actor_button_w + actor_button_gap,
            actor_buttons_y,
            actor_button_w,
            self.button_height,
        )
        self.place_ped_btn.rect = pygame.Rect(
            left_x,
            actor_buttons_y + self.button_height + actor_button_gap,
            actor_button_w,
            self.button_height,
        )
        self.del_btn.rect = pygame.Rect(
            left_x + actor_button_w + actor_button_gap,
            actor_buttons_y + self.button_height + actor_button_gap,
            actor_button_w,
            self.button_height,
        )
        actor_list_y = actor_buttons_y + 2 * self.button_height + actor_button_gap + self.spacing_md
        self.actor_row_rects = []
        actor_card_x = self.section_rects["actors"].x + self.section_pad
        actor_card_w = self.section_rects["actors"].width - 2 * self.section_pad
        for idx in range(actor_list_rows):
            self.actor_row_rects.append(
                pygame.Rect(actor_card_x, actor_list_y + idx * actor_row_h, actor_card_w, actor_row_h)
            )
        y += section_heights["actors"] + distributed_gap

        self.section_rects["actions"] = pygame.Rect(
            left_x - self.section_pad,
            y,
            left_w + 2 * self.section_pad,
            section_heights["actions"],
        )
        anchor_y = (
            self.section_rects["actions"].y
            + max(self.spacing_xs, (self.section_rects["actions"].height - self.button_height) // 2 - self.spacing_xs)
        )
        self.anchor_btn.rect = pygame.Rect(left_x, anchor_y, left_w, self.button_height)

        content_end = self.section_rects["actions"].bottom + self.spacing_sm
        self.left_content_height = max(available_h, content_end - content_top)

        right_x = self.right_panel_rect.x + self.panel_padding
        right_w = self.right_panel_rect.width - 2 * self.panel_padding
        fov_size = min(
            right_w,
            int(self.layout_cfg.preview_fov_max * self.ui_scale),
        )
        right_card_title_h = self.font_section.get_height()
        right_card_top_pad = self.section_pad
        right_card_bottom_pad = self.section_pad
        right_card_header_gap = self.spacing_sm
        summary_line_h = self.font.get_height()
        status_line_h = self.font_small.get_height()
        preview_card_y = self.right_body_rect.y + self.spacing_sm
        preview_card_h = right_card_top_pad + right_card_title_h + right_card_header_gap + fov_size + right_card_bottom_pad
        self.preview_card_rect = pygame.Rect(
            right_x - self.section_pad,
            preview_card_y,
            right_w + 2 * self.section_pad,
            preview_card_h,
        )
        self.fov_rect = pygame.Rect(
            self.right_panel_rect.x + (self.right_panel_rect.width - fov_size) // 2,
            self.preview_card_rect.y + right_card_top_pad + right_card_title_h + right_card_header_gap,
            fov_size,
            fov_size,
        )
        action_y = self.preview_card_rect.bottom + self.spacing_lg
        self.add_rdm_actor_btn.rect = pygame.Rect(right_x, action_y, right_w, self.button_height)
        self.play_btn.rect = pygame.Rect(right_x, action_y + self.button_height + self.spacing_md, right_w, self.button_height)
        self.save_btn.rect = pygame.Rect(
            right_x,
            self.play_btn.rect.bottom + self.spacing_md,
            right_w,
            self.button_height,
        )

        summary_y = self.save_btn.rect.bottom + self.spacing_lg
        summary_lines_h = len(self.status_lines[:3]) * (status_line_h + self.spacing_xs)
        info_content_h = (
            right_card_title_h
            + right_card_header_gap
            + 3 * summary_line_h
            + 2 * self.spacing_sm
            + self.spacing_lg
            + right_card_title_h
            + right_card_header_gap
            + max(status_line_h, summary_lines_h)
        )
        self.info_card_rect = pygame.Rect(
            right_x - self.section_pad,
            summary_y,
            right_w + 2 * self.section_pad,
            max(
                int(self.layout_cfg.summary_card_min_height * self.ui_scale),
                right_card_top_pad + info_content_h + right_card_bottom_pad,
            ),
        )

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

    def map_to_screen_pos(self, map_pos):
        if map_pos is None or self.map_rect.width <= 0 or self.map_rect.height <= 0:
            return None
        map_x, map_y = map_pos
        rel_x = (map_x - self._map_crop_rect.x) / max(1, self._map_crop_rect.width)
        rel_y = (map_y - self._map_crop_rect.y) / max(1, self._map_crop_rect.height)
        screen_x = int(self.map_rect.x + rel_x * self.map_rect.width)
        screen_y = int(self.map_rect.y + rel_y * self.map_rect.height)
        return screen_x, screen_y

    def set_status(self, *lines):
        self.status_lines = list(lines)

    def draw_fov(self):
        self._draw_card_rect(self.preview_card_rect)
        title = self.font_section.render("FOV Preview", True, self.colors["text"])
        self.screen.blit(title, (self.preview_card_rect.x + self.section_pad, self.preview_card_rect.y + self.section_pad))

        vehicle_surface = self.env.map.canvas
        pygame.draw.rect(
            self.screen,
            self.colors["border"],
            self.fov_rect.inflate(2, 2),
            width=1,
            border_radius=self.button_radius,
        )
        if vehicle_surface is not None:
            scaled = pygame.transform.scale(vehicle_surface, (self.fov_rect.width, self.fov_rect.height))
            self.screen.blit(scaled, self.fov_rect.topleft)
        else:
            pygame.draw.rect(self.screen, (18, 20, 24), self.fov_rect, border_radius=self.button_radius)
            text = self.font.render("No FOV", True, (220, 223, 228))
            self.screen.blit(text, text.get_rect(center=self.fov_rect.center))

    def draw_timeline(self):
        pygame.draw.rect(self.screen, (208, 214, 220), self.timeline_rect, border_radius=self.timeline_handle_radius)
        if self.total_frames <= 0:
            return
        handle_x = self.timeline_rect.x + int((self.current_frame / self.total_frames) * self.timeline_rect.width)
        handle_x = max(
            self.timeline_rect.x + self.timeline_handle_radius,
            min(self.timeline_rect.right - self.timeline_handle_radius, handle_x),
        )
        pygame.draw.circle(
            self.screen,
            self.colors["accent"],
            (handle_x, self.timeline_rect.centery),
            self.timeline_handle_radius,
        )

    def draw_gui(self):
        self._refresh_window_metrics()
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
        self.screen.blit(
            self.font_title.render("Scenario Designer", True, self.colors["text"]),
            self.left_header_title_pos,
        )
        self.screen.blit(
            self.font_small.render("Configure a scenario, then place it on the map.", True, self.colors["muted"]),
            self.left_header_subtitle_pos,
        )
        previous_clip = self.screen.get_clip()
        self.screen.set_clip(self.left_body_rect)

        self._draw_soft_section(self.section_rects["scene_name"])
        self._draw_section_header(
            "Scene Name",
            self.section_rects["scene_name"].x + self.section_pad,
            self.section_rects["scene_name"].y + self.section_pad,
        )
        self.scene_name.draw(self.screen)

        self._draw_soft_section(self.section_rects["setup"])
        self._draw_section_header(
            "Scenario Setup",
            self.section_rects["setup"].x + self.section_pad,
            self.section_rects["setup"].y + self.section_pad,
        )
        scenario_label_y = self.scenario_selector.rect.y - self.spacing_xs - self.font_small.get_height()
        level_label_y = self.level_selector.rect.y - self.spacing_xs - self.font_small.get_height()
        self._draw_label("Scenario", self.scenario_selector.rect.x, scenario_label_y, small=True)
        self.scenario_selector.draw(self.screen)
        self._draw_label("Difficulty", self.level_selector.rect.x, level_label_y, small=True)
        self.level_selector.draw(self.screen)

        if self.active_fields:
            self._draw_soft_section(self.section_rects["parameters"])
            self._draw_section_header(
                "Parameters",
                self.section_rects["parameters"].x + self.section_pad,
                self.section_rects["parameters"].y + self.section_pad,
            )

        for field in self.active_fields:
            box = self.field_boxes[field.key]
            self._draw_label(
                field.label,
                box.rect.x,
                box.rect.y - self.spacing_xs - self.font_small.get_height(),
                small=True,
            )
            box.draw(self.screen)

        self._draw_soft_section(self.section_rects["actors"])
        self._draw_section_header(
            "Actors",
            self.section_rects["actors"].x + self.section_pad,
            self.section_rects["actors"].y + self.section_pad,
        )
        self.place_ego_btn.draw(self.screen)
        self.place_vehicle_btn.draw(self.screen)
        self.place_ped_btn.draw(self.screen)
        self.del_btn.draw(self.screen)
        self._draw_actor_rows()

        self._draw_soft_section(self.section_rects["actions"])
        self.anchor_btn.draw(self.screen)
        self.screen.set_clip(previous_clip)

    def _draw_actor_rows(self):
        actor_colors = {
            "agent": (29, 110, 235),
            "vehicle": (0, 118, 182),
            "pedestrian": (201, 63, 63),
        }
        for idx, row_rect in enumerate(self.actor_row_rects):
            is_selected = idx == self.selected_actor_index
            bg = (227, 238, 255) if is_selected else (246, 247, 249)
            border = self.colors["accent"] if is_selected else (204, 209, 216)
            pygame.draw.rect(self.screen, bg, row_rect, border_radius=self.button_radius)
            pygame.draw.rect(self.screen, border, row_rect, 1, border_radius=self.button_radius)
            if idx >= len(self.designed_actors):
                continue
            actor = self.designed_actors[idx]
            label = actor["type"].replace("agent", "ego").title()
            start = actor["start"]
            end = actor["end"]
            text = f"{label}: ({start[0]}, {start[1]}) -> ({end[0]}, {end[1]})"
            dot_color = actor_colors.get(actor["type"], self.colors["accent"])
            pygame.draw.circle(self.screen, dot_color, (row_rect.x + 10, row_rect.centery), 4)
            surf = self.font_small.render(text, True, self.colors["text"])
            self.screen.blit(surf, (row_rect.x + 20, row_rect.y + (row_rect.height - surf.get_height()) // 2))

    def _draw_actor_overlay(self):
        actor_colors = {
            "agent": (29, 110, 235),
            "vehicle": (0, 118, 182),
            "pedestrian": (201, 63, 63),
        }
        for idx, actor in enumerate(self.designed_actors):
            start = self.map_to_screen_pos(actor["start"])
            end = self.map_to_screen_pos(actor["end"])
            if start is None or end is None:
                continue
            color = actor_colors.get(actor["type"], self.colors["accent"])
            width = 4 if idx == self.selected_actor_index else 2
            pygame.draw.line(self.screen, color, start, end, width)
            pygame.draw.circle(self.screen, (50, 180, 80), start, 6 if idx == self.selected_actor_index else 5)
            pygame.draw.circle(self.screen, (215, 80, 80), end, 6 if idx == self.selected_actor_index else 5)
        if self.pending_actor_start is not None:
            start = self.map_to_screen_pos(self.screen_to_map_pos(self.pending_actor_start))
            if start is not None:
                pygame.draw.circle(self.screen, self.colors["accent"], start, 6)

    def _draw_center_canvas(self):
        frame_rect = self.map_rect.inflate(self.frame_pad, self.frame_pad)
        pygame.draw.rect(self.screen, self.colors["canvas_frame"], frame_rect, border_radius=self.card_radius + 4)
        pygame.draw.rect(
            self.screen,
            self.colors["canvas_bg"],
            frame_rect.inflate(-2, -2),
            border_radius=self.card_radius + 2,
        )

        cropped_map = self.env.map.map_surface.subsurface(self._map_crop_rect)
        scaled_map = pygame.transform.scale(cropped_map, self.map_rect.size)
        self.screen.blit(scaled_map, self.map_rect.topleft)
        self._draw_actor_overlay()

        caption = self.font_small.render("Map canvas", True, self.colors["muted"])
        self.screen.blit(caption, (frame_rect.x + self.spacing_sm, frame_rect.y + self.spacing_xs))

        help_text = self.font_small.render(
            "Click anywhere on the map to place the scenario anchor.",
            True,
            self.colors["muted"],
        )
        self.screen.blit(help_text, (self.map_rect.x, self.timeline_rect.y - self.font_small.get_height() - self.spacing_sm))
        self.draw_timeline()

    def _draw_right_panel(self):
        self.screen.blit(
            self.font_title.render("Preview", True, self.colors["text"]),
            self.right_header_title_pos,
        )
        self.screen.blit(
            self.font_small.render("Run, inspect, and save the current setup.", True, self.colors["muted"]),
            self.right_header_subtitle_pos,
        )

        self.add_rdm_actor_btn.draw(self.screen)
        self.play_btn.draw(self.screen)
        self.save_btn.draw(self.screen)

        self._draw_card_rect(self.info_card_rect)
        info_x = self.info_card_rect.x + self.section_pad
        info_y = self.info_card_rect.y + self.section_pad
        self.screen.blit(self.font_section.render("Current Setup", True, self.colors["text"]), (info_x, info_y))

        summary_lines = [
            f"Scenario: {self.scenario_spec.display_name}",
            f"Level: {self.level_selector.selection}",
            f"Fields: {len(self.active_fields)}",
        ]
        summary_y = info_y + self.font_section.get_height() + self.spacing_sm
        for idx, line in enumerate(summary_lines):
            self.screen.blit(
                self.font.render(line, True, self.colors["text"]),
                (
                    info_x,
                    summary_y + idx * (self.font.get_height() + self.spacing_sm),
                ),
            )

        status_title_y = (
            summary_y
            + len(summary_lines) * self.font.get_height()
            + max(0, len(summary_lines) - 1) * self.spacing_sm
            + self.spacing_lg
        )
        self.screen.blit(
            self.font_section.render("Status", True, self.colors["text"]),
            (info_x, status_title_y),
        )
        for idx, line in enumerate(self.status_lines[:3]):
            self._draw_wrapped_text(
                line,
                pygame.Rect(
                    info_x,
                    status_title_y + self.font_section.get_height() + self.spacing_sm + idx * (self.font_small.get_height() + self.spacing_xs),
                    self.info_card_rect.width - 2 * self.section_pad,
                    self.font_small.get_height() + self.spacing_sm,
                ),
                self.font_small,
                self.colors["muted"],
            )

    def update_frame_from_mouse(self, mouse_x):
        rel_x = max(0, min(mouse_x - self.timeline_rect.x, self.timeline_rect.width))
        self.current_frame = int((rel_x / max(1, self.timeline_rect.width)) * self.total_frames)

    def _draw_card(self, x, y, width, height):
        rect = pygame.Rect(x, y, width, height)
        self._draw_card_rect(rect)

    def _draw_card_rect(self, rect):
        pygame.draw.rect(self.screen, self.colors["card_bg"], rect, border_radius=self.card_radius)
        pygame.draw.rect(self.screen, self.colors["border"], rect, width=1, border_radius=self.card_radius)

    def _draw_soft_section(self, rect):
        pygame.draw.rect(self.screen, (250, 251, 252), rect, border_radius=self.section_radius)
        pygame.draw.rect(self.screen, (214, 219, 226), rect, width=1, border_radius=self.section_radius)

    def _draw_label(self, text, x, y, small=False):
        font = self.font_small if small else self.font
        self.screen.blit(font.render(text, True, self.colors["muted"]), (x, y))

    def _draw_section_header(self, text, x, y):
        self.screen.blit(self.font_section.render(text, True, self.colors["text"]), (x, y))

    def _draw_disabled_button(self, rect, text):
        pygame.draw.rect(self.screen, self.colors["success_soft"], rect, border_radius=self.button_radius)
        pygame.draw.rect(self.screen, self.colors["border"], rect, width=1, border_radius=self.button_radius)
        label = self.font.render(text, True, self.colors["success"])
        self.screen.blit(label, label.get_rect(center=rect.center))

    def _draw_scroll_indicator(self):
        track_w = max(4, int(round(6 * self.ui_scale)))
        track_rect = pygame.Rect(
            self.left_panel_rect.right - self.spacing_sm,
            self.left_body_rect.y + self.spacing_sm,
            track_w,
            self.left_body_rect.height - 2 * self.spacing_sm,
        )
        pygame.draw.rect(self.screen, (220, 225, 232), track_rect, border_radius=track_w // 2)
        max_scroll = max(1, self.left_content_height - self.left_body_rect.height)
        thumb_h = max(
            int(track_rect.height * (self.left_body_rect.height / max(1, self.left_content_height))),
            self.layout_cfg.scroll_thumb_min,
        )
        thumb_y = track_rect.y + int((track_rect.height - thumb_h) * (self.left_scroll_offset / max_scroll))
        thumb_rect = pygame.Rect(track_rect.x, thumb_y, track_rect.width, thumb_h)
        pygame.draw.rect(self.screen, (178, 186, 197), thumb_rect, border_radius=track_w // 2)

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
        mask = diff > self.layout_cfg.crop_diff_threshold

        # Ignore the low-information top strip by looking for the first sustained
        # band of wide horizontal content rather than isolated corner pixels.
        row_activity_ratio = mask.sum(axis=0) / max(1, surface.get_width())
        sustained_threshold = self.layout_cfg.crop_top_activity_threshold
        sustained_rows = np.where(row_activity_ratio > sustained_threshold)[0]
        if sustained_rows.size > 0:
            top_cut = int(sustained_rows[0])
            max_trim = int(surface.get_height() * self.layout_cfg.crop_top_trim_ratio_max)
            top_cut = min(top_cut, max_trim)
            if top_cut > 0:
                mask[:, :top_cut] = False

        coords = np.argwhere(mask)
        if coords.size == 0:
            return pygame.Rect(0, 0, surface.get_width(), surface.get_height())

        xs = coords[:, 0]
        ys = coords[:, 1]
        pad = self.layout_cfg.crop_padding
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

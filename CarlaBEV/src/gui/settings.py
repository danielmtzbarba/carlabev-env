from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class DesignerLayoutConfig:
    min_window_width: int = 1080
    min_window_height: int = 720
    window_width_ratio: float = 0.92
    window_height_ratio: float = 0.90
    window_margin_x: int = 32
    window_margin_y: int = 48

    scale_ref_width: float = 1440.0
    scale_ref_height: float = 960.0
    scale_min: float = 0.70
    scale_max: float = 1.35

    font_body_base: int = 22
    font_body_min: int = 20
    font_small_base: int = 18
    font_small_min: int = 16
    font_title_base: int = 34
    font_title_min: int = 28
    font_section_base: int = 24
    font_section_min: int = 22

    spacing_xs_base: int = 6
    spacing_sm_base: int = 12
    spacing_md_base: int = 18
    spacing_lg_base: int = 28

    panel_padding_base: int = 28
    section_padding_base: int = 14
    section_title_gap_base: int = 18
    param_label_gap_base: int = 16
    frame_padding_base: int = 26
    header_gap_base: int = 12

    option_height_base: int = 30
    option_gap_base: int = 4
    textbox_height_base: int = 40
    button_height_base: int = 38
    timeline_height_base: int = 12

    section_radius_base: int = 18
    card_radius_base: int = 18
    button_radius_base: int = 12
    timeline_handle_radius_base: int = 8

    left_panel_width_max: int = 380
    left_panel_width_min: int = 290
    left_panel_floor: int = 270
    right_panel_width_max: int = 290
    right_panel_width_min: int = 220
    right_panel_floor: int = 200
    center_panel_min_width: int = 360
    center_bottom_reserve: int = 110
    center_bottom_reserve_min: int = 96

    preview_fov_max: int = 210
    summary_card_min_height: int = 250

    scroll_step_min: int = 24
    scroll_thumb_min: int = 32

    crop_diff_threshold: int = 12
    crop_top_activity_threshold: float = 0.22
    crop_top_trim_ratio_max: float = 0.28
    crop_padding: int = 12


DESIGNER_LAYOUT_PRESETS = {
    "comfortable": DesignerLayoutConfig(
        font_title_base=40,
        font_title_min=34,
        font_section_base=26,
        font_section_min=24,
        spacing_sm_base=10,
        spacing_md_base=22,
        spacing_lg_base=34,
        panel_padding_base=20,
        section_padding_base=10,
        section_title_gap_base=18,
        param_label_gap_base=12,
        frame_padding_base=22,
        header_gap_base=14,
        option_height_base=28,
        textbox_height_base=36,
        button_height_base=36,
        left_panel_width_max=400,
        left_panel_width_min=310,
        left_panel_floor=290,
        right_panel_width_max=285,
        right_panel_width_min=215,
        preview_fov_max=200,
        summary_card_min_height=236,
        scroll_thumb_min=28,
        crop_padding=10,
    ),
    "compact": DesignerLayoutConfig(
        min_window_width=1024,
        min_window_height=680,
        scale_min=0.66,
        font_title_base=36,
        font_title_min=30,
        spacing_xs_base=5,
        spacing_sm_base=10,
        spacing_md_base=14,
        spacing_lg_base=22,
        panel_padding_base=22,
        section_padding_base=12,
        section_title_gap_base=14,
        param_label_gap_base=12,
        frame_padding_base=20,
        header_gap_base=10,
        option_height_base=28,
        textbox_height_base=38,
        button_height_base=36,
        left_panel_width_max=350,
        left_panel_width_min=270,
        right_panel_width_max=270,
        right_panel_width_min=210,
        center_bottom_reserve=96,
        preview_fov_max=188,
        summary_card_min_height=220,
        scroll_thumb_min=28,
        crop_padding=10,
    ),
    "dense": DesignerLayoutConfig(
        min_window_width=960,
        min_window_height=640,
        scale_min=0.62,
        font_title_base=32,
        font_title_min=28,
        spacing_xs_base=4,
        spacing_sm_base=8,
        spacing_md_base=12,
        spacing_lg_base=18,
        panel_padding_base=18,
        section_padding_base=10,
        section_title_gap_base=12,
        param_label_gap_base=10,
        frame_padding_base=16,
        header_gap_base=8,
        option_height_base=26,
        option_gap_base=3,
        textbox_height_base=34,
        button_height_base=34,
        timeline_height_base=10,
        section_radius_base=16,
        card_radius_base=16,
        button_radius_base=10,
        left_panel_width_max=320,
        left_panel_width_min=250,
        left_panel_floor=240,
        right_panel_width_max=250,
        right_panel_width_min=190,
        right_panel_floor=180,
        center_bottom_reserve=88,
        center_bottom_reserve_min=80,
        preview_fov_max=170,
        summary_card_min_height=200,
        scroll_step_min=20,
        scroll_thumb_min=24,
        crop_padding=8,
    ),
}


def get_designer_layout_config(preset="comfortable", overrides=None):
    base = DESIGNER_LAYOUT_PRESETS.get(preset)
    if base is None:
        preset_names = ", ".join(sorted(DESIGNER_LAYOUT_PRESETS))
        raise ValueError(f"Unknown layout preset '{preset}'. Expected one of: {preset_names}")
    if not overrides:
        return base
    return replace(base, **overrides)


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
    designer_layout_preset: str = "comfortable"
    designer_layout_overrides: dict = field(default_factory=dict)
    designer_layout: DesignerLayoutConfig = field(init=False)

    def __post_init__(self):
        if self.designer_layout_preset == "auto":
            self.designer_layout = DesignerLayoutConfig()
        else:
            self.designer_layout = get_designer_layout_config(
                self.designer_layout_preset,
                self.designer_layout_overrides,
            )

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import pickle
import re

import networkx as nx
import numpy as np

from CarlaBEV.envs.geometry import RAW_METERS_PER_PIXEL, raw_to_surface
from CarlaBEV.envs.utils import asset_path


DEFAULT_TOWN_NAME = "Town01"
DEFAULT_GRAPH_NAME = "town01-canonical-vehicle.pkl"
LEGACY_BACKBONE_NAME = "town01-vehicles-2lanes-100.pkl"
MANUAL_CONNECTOR_EXCLUSIONS = {
    ("intersection_1", "5_6", "6_13", "forward_a"),
    ("intersection_1", "6_13", "5_6", "forward_a"),
    ("intersection_1", "5_6", "6_13", "forward_b"),
    ("intersection_1", "6_13", "5_6", "forward_b"),
    ("intersection_2", "0_2", "2_3", "forward_a"),
    ("intersection_2", "0_2", "2_3", "forward_b"),
    ("intersection_2", "2_3", "0_2", "forward_a"),
    ("intersection_2", "2_3", "0_2", "forward_b"),
    ("intersection_2", "0_2", "2_7", "forward_a"),
    ("intersection_2", "0_2", "2_7", "forward_b"),
    ("intersection_2", "2_7", "0_2", "forward_a"),
    ("intersection_2", "2_7", "0_2", "forward_b"),
    ("intersection_2", "2_3", "2_7", "forward_a"),
    ("intersection_2", "2_3", "2_7", "forward_b"),
    ("intersection_2", "2_7", "2_3", "forward_a"),
    ("intersection_2", "2_7", "2_3", "forward_b"),
    ("intersection_6", "13_15", "12_13", "forward_a"),
    ("intersection_6", "12_13", "13_15", "forward_a"),
    ("intersection_6", "13_15", "12_13", "forward_b"),
    ("intersection_6", "12_13", "13_15", "forward_b"),
    ("intersection_12", "5_12", "12_13", "forward_a"),
    ("intersection_12", "5_12", "12_13", "forward_b"),
    ("intersection_12", "12_13", "5_12", "forward_b"),
    ("intersection_12", "12_13", "5_12", "forward_a"),
    ("intersection_11", "4_11", "11_12", "forward_a"),
    ("intersection_11", "4_11", "11_12", "forward_b"),
    ("intersection_11", "11_12", "4_11", "forward_a"),
    ("intersection_11", "11_12", "4_11", "forward_b"),
    ("intersection_8", "3_8", "8_10", "forward_a"),
    ("intersection_8", "3_8", "8_10", "forward_b"),
    ("intersection_8", "8_10", "3_8", "forward_a"),
    ("intersection_8", "8_10", "3_8", "forward_b"),
    ("intersection_8", "7_8", "8_10", "forward_a"),
    ("intersection_8", "7_8", "8_10", "forward_b"),
    ("intersection_8", "8_10", "7_8", "forward_a"),
    ("intersection_8", "8_10", "7_8", "forward_b"),
    ("intersection_8", "7_8", "3_8", "forward_a"),
    ("intersection_8", "7_8", "3_8", "forward_b"),
    ("intersection_8", "3_8", "7_8", "forward_a"),
    ("intersection_8", "3_8", "7_8", "forward_b"),
    ("intersection_7", "2_7", "7_8", "forward_a"),
    ("intersection_7", "2_7", "7_8", "forward_b"),
    ("intersection_7", "7_8", "2_7", "forward_a"),
    ("intersection_7", "7_8", "2_7", "forward_b"),
    ("intersection_7", "2_7", "7_9", "forward_a"),
    ("intersection_7", "2_7", "7_9", "forward_b"),
    ("intersection_7", "7_9", "2_7", "forward_a"),
    ("intersection_7", "7_9", "2_7", "forward_b"),
    ("intersection_7", "7_8", "7_9", "forward_a"),
    ("intersection_7", "7_8", "7_9", "forward_b"),
    ("intersection_7", "7_9", "7_8", "forward_a"),
    ("intersection_7", "7_9", "7_8", "forward_b"),
    ("intersection_9", "7_9", "9_10", "forward_a"),
    ("intersection_9", "7_9", "9_10", "forward_b"),
    ("intersection_9", "9_10", "7_9", "forward_a"),
    ("intersection_9", "9_10", "7_9", "forward_b"),
    ("intersection_9", "7_9", "9_14", "forward_a"),
    ("intersection_9", "7_9", "9_14", "forward_b"),
    ("intersection_9", "9_14", "7_9", "forward_a"),
    ("intersection_9", "9_14", "7_9", "forward_b"),
    ("intersection_9", "9_10", "9_14", "forward_a"),
    ("intersection_9", "9_10", "9_14", "forward_b"),
    ("intersection_9", "9_14", "9_10", "forward_a"),
    ("intersection_9", "9_14", "9_10", "forward_b"),
    ("intersection_10", "8_10", "10_11", "forward_a"),
    ("intersection_10", "8_10", "10_11", "forward_b"),
    ("intersection_10", "10_11", "8_10", "forward_a"),
    ("intersection_10", "10_11", "8_10", "forward_b"),
    ("intersection_10", "8_10", "9_10", "forward_a"),
    ("intersection_10", "8_10", "9_10", "forward_b"),
    ("intersection_10", "9_10", "8_10", "forward_a"),
    ("intersection_10", "9_10", "8_10", "forward_b"),
    ("intersection_3", "2_3", "3_8", "forward_a"),
    ("intersection_3", "2_3", "3_8", "forward_b"),
    ("intersection_3", "3_8", "2_3", "forward_a"),
    ("intersection_3", "3_8", "2_3", "forward_b"),
    ("intersection_3", "3_4", "3_8", "forward_a"),
    ("intersection_3", "3_4", "3_8", "forward_b"),
    ("intersection_3", "3_8", "3_4", "forward_a"),
    ("intersection_3", "3_8", "3_4", "forward_b"),
    ("intersection_4", "3_4", "4_5", "forward_a"),
    ("intersection_4", "3_4", "4_5", "forward_b"),
    ("intersection_4", "4_5", "3_4", "forward_a"),
    ("intersection_4", "4_5", "3_4", "forward_b"),
    ("intersection_4", "3_4", "4_11", "forward_a"),
    ("intersection_4", "3_4", "4_11", "forward_b"),
    ("intersection_4", "4_11", "3_4", "forward_a"),
    ("intersection_4", "4_11", "3_4", "forward_b"),
    ("intersection_4", "4_5", "4_11", "forward_a"),
    ("intersection_4", "4_5", "4_11", "forward_b"),
    ("intersection_4", "4_11", "4_5", "forward_a"),
    ("intersection_4", "4_11", "4_5", "forward_b"),
    ("intersection_5", "4_5", "5_6", "forward_a"),
    ("intersection_5", "4_5", "5_6", "forward_b"),
    ("intersection_5", "5_6", "4_5", "forward_a"),
    ("intersection_5", "5_6", "4_5", "forward_b"),
    ("intersection_5", "4_5", "5_12", "forward_a"),
    ("intersection_5", "4_5", "5_12", "forward_b"),
    ("intersection_5", "5_12", "4_5", "forward_a"),
    ("intersection_5", "5_12", "4_5", "forward_b"),
    ("intersection_5", "5_6", "5_12", "forward_a"),
    ("intersection_5", "5_6", "5_12", "forward_b"),
    ("intersection_5", "5_12", "5_6", "forward_a"),
    ("intersection_5", "5_12", "5_6", "forward_b"),
}
MANUAL_VALIDATED_JUNCTIONS = {
    "intersection_1",
    "intersection_10",
    "intersection_11",
    "intersection_12",
    "intersection_2",
    "intersection_3",
    "intersection_14",
    "intersection_7",
    "intersection_8",
    "intersection_9",
    "intersection_6",
}
MANUAL_CORRIDOR_FLOW_LANE_SWAPS = {
    "4_11",
    "5_12",
}
MANUAL_CORRIDOR_LABEL_SWAPS = {
    "4_11",
    "5_12",
}
MANUAL_CORRIDOR_FLOW_INVERSIONS = {
    "4_11",
    "5_12",
}
MANUAL_CORRIDOR_SIDE_SWAPS = {
    "3_8",
    "8_10",
}
MANUAL_NODE_JUNCTION_OVERRIDES = {
    "14_15_L_0": "intersection_13",
    "14_15_R_49": "intersection_13",
}
MANUAL_CONNECTOR_ADDITIONS = (
    {
        "src": "12_13_R_0",
        "dst": "5_12_R_0",
        "edge_kind": "junction_connector",
        "lane_change": False,
        "confidence": 1.0,
        "source_hint": "manual_connector_override",
        "priority": None,
        "signal_group": None,
        "legal_for": None,
    },
    {
        "src": "12_13_R_17",
        "dst": "13_15_L_0",
        "edge_kind": "junction_connector",
        "lane_change": False,
        "confidence": 1.0,
        "source_hint": "manual_connector_override",
        "priority": None,
        "signal_group": None,
        "legal_for": None,
    },
    {
        "src": "13_15_R_5",
        "dst": "12_13_L_0",
        "edge_kind": "junction_connector",
        "lane_change": False,
        "confidence": 1.0,
        "source_hint": "manual_connector_override",
        "priority": None,
        "signal_group": None,
        "legal_for": None,
    },
    {
        "src": "5_6_L_0",
        "dst": "6_13_R_36",
        "edge_kind": "junction_connector",
        "lane_change": False,
        "confidence": 1.0,
        "source_hint": "manual_connector_override",
        "priority": None,
        "signal_group": None,
        "legal_for": None,
    },
    {
        "src": "6_13_L_0",
        "dst": "5_6_R_17",
        "edge_kind": "junction_connector",
        "lane_change": False,
        "confidence": 1.0,
        "source_hint": "manual_connector_override",
        "priority": None,
        "signal_group": None,
        "legal_for": None,
    },
    {
        "src": "5_12_L_36",
        "dst": "12_13_L_17",
        "edge_kind": "junction_connector",
        "lane_change": False,
        "confidence": 1.0,
        "source_hint": "manual_connector_override",
        "priority": None,
        "signal_group": None,
        "legal_for": None,
    },
    {
        "src": "4_11_L_36",
        "dst": "11_12_L_8",
        "edge_kind": "junction_connector",
        "lane_change": False,
        "confidence": 1.0,
        "source_hint": "manual_connector_override",
        "priority": None,
        "signal_group": None,
        "legal_for": None,
    },
    {
        "src": "11_12_R_0",
        "dst": "4_11_R_0",
        "edge_kind": "junction_connector",
        "lane_change": False,
        "confidence": 1.0,
        "source_hint": "manual_connector_override",
        "priority": None,
        "signal_group": None,
        "legal_for": None,
    },
    {
        "src": "8_10_L_25",
        "dst": "9_10_R_6",
        "edge_kind": "junction_connector",
        "lane_change": False,
        "confidence": 1.0,
        "source_hint": "manual_connector_override",
        "priority": None,
        "signal_group": None,
        "legal_for": None,
    },
    {
        "src": "8_10_L_25",
        "dst": "10_11_L_9",
        "edge_kind": "junction_connector",
        "lane_change": False,
        "confidence": 1.0,
        "source_hint": "manual_connector_override",
        "priority": None,
        "signal_group": None,
        "legal_for": None,
    },
    {
        "src": "9_10_L_0",
        "dst": "8_10_R_0",
        "edge_kind": "junction_connector",
        "lane_change": False,
        "confidence": 1.0,
        "source_hint": "manual_connector_override",
        "priority": None,
        "signal_group": None,
        "legal_for": None,
    },
    {
        "src": "10_11_R_0",
        "dst": "8_10_R_0",
        "edge_kind": "junction_connector",
        "lane_change": False,
        "confidence": 1.0,
        "source_hint": "manual_connector_override",
        "priority": None,
        "signal_group": None,
        "legal_for": None,
    },
    {
        "src": "8_10_R_25",
        "dst": "3_8_R_0",
        "edge_kind": "junction_connector",
        "lane_change": False,
        "confidence": 1.0,
        "source_hint": "manual_connector_override",
        "priority": None,
        "signal_group": None,
        "legal_for": None,
    },
    {
        "src": "8_10_R_25",
        "dst": "7_8_R_6",
        "edge_kind": "junction_connector",
        "lane_change": False,
        "confidence": 1.0,
        "source_hint": "manual_connector_override",
        "priority": None,
        "signal_group": None,
        "legal_for": None,
    },
    {
        "src": "3_8_L_7",
        "dst": "8_10_L_0",
        "edge_kind": "junction_connector",
        "lane_change": False,
        "confidence": 1.0,
        "source_hint": "manual_connector_override",
        "priority": None,
        "signal_group": None,
        "legal_for": None,
    },
    {
        "src": "3_8_L_7",
        "dst": "7_8_R_6",
        "edge_kind": "junction_connector",
        "lane_change": False,
        "confidence": 1.0,
        "source_hint": "manual_connector_override",
        "priority": None,
        "signal_group": None,
        "legal_for": None,
    },
    {
        "src": "7_8_L_0",
        "dst": "8_10_L_0",
        "edge_kind": "junction_connector",
        "lane_change": False,
        "confidence": 1.0,
        "source_hint": "manual_connector_override",
        "priority": None,
        "signal_group": None,
        "legal_for": None,
    },
    {
        "src": "7_8_L_0",
        "dst": "3_8_R_0",
        "edge_kind": "junction_connector",
        "lane_change": False,
        "confidence": 1.0,
        "source_hint": "manual_connector_override",
        "priority": None,
        "signal_group": None,
        "legal_for": None,
    },
    {
        "src": "2_7_L_0",
        "dst": "7_8_L_6",
        "edge_kind": "junction_connector",
        "lane_change": False,
        "confidence": 1.0,
        "source_hint": "manual_connector_override",
        "priority": None,
        "signal_group": None,
        "legal_for": None,
    },
    {
        "src": "2_7_L_0",
        "dst": "7_9_L_25",
        "edge_kind": "junction_connector",
        "lane_change": False,
        "confidence": 1.0,
        "source_hint": "manual_connector_override",
        "priority": None,
        "signal_group": None,
        "legal_for": None,
    },
    {
        "src": "7_8_R_0",
        "dst": "2_7_R_7",
        "edge_kind": "junction_connector",
        "lane_change": False,
        "confidence": 1.0,
        "source_hint": "manual_connector_override",
        "priority": None,
        "signal_group": None,
        "legal_for": None,
    },
    {
        "src": "7_8_R_0",
        "dst": "7_9_L_25",
        "edge_kind": "junction_connector",
        "lane_change": False,
        "confidence": 1.0,
        "source_hint": "manual_connector_override",
        "priority": None,
        "signal_group": None,
        "legal_for": None,
    },
    {
        "src": "7_9_R_0",
        "dst": "2_7_R_7",
        "edge_kind": "junction_connector",
        "lane_change": False,
        "confidence": 1.0,
        "source_hint": "manual_connector_override",
        "priority": None,
        "signal_group": None,
        "legal_for": None,
    },
    {
        "src": "7_9_R_0",
        "dst": "7_8_L_6",
        "edge_kind": "junction_connector",
        "lane_change": False,
        "confidence": 1.0,
        "source_hint": "manual_connector_override",
        "priority": None,
        "signal_group": None,
        "legal_for": None,
    },
    {
        "src": "7_9_L_0",
        "dst": "9_10_L_6",
        "edge_kind": "junction_connector",
        "lane_change": False,
        "confidence": 1.0,
        "source_hint": "manual_connector_override",
        "priority": None,
        "signal_group": None,
        "legal_for": None,
    },
    {
        "src": "0_2_L_0",
        "dst": "2_3_L_6",
        "edge_kind": "junction_connector",
        "lane_change": False,
        "confidence": 1.0,
        "source_hint": "manual_connector_override",
        "priority": None,
        "signal_group": None,
        "legal_for": None,
    },
    {
        "src": "0_2_L_0",
        "dst": "2_7_L_7",
        "edge_kind": "junction_connector",
        "lane_change": False,
        "confidence": 1.0,
        "source_hint": "manual_connector_override",
        "priority": None,
        "signal_group": None,
        "legal_for": None,
    },
    {
        "src": "2_3_R_0",
        "dst": "0_2_R_11",
        "edge_kind": "junction_connector",
        "lane_change": False,
        "confidence": 1.0,
        "source_hint": "manual_connector_override",
        "priority": None,
        "signal_group": None,
        "legal_for": None,
    },
    {
        "src": "2_3_R_0",
        "dst": "2_7_L_7",
        "edge_kind": "junction_connector",
        "lane_change": False,
        "confidence": 1.0,
        "source_hint": "manual_connector_override",
        "priority": None,
        "signal_group": None,
        "legal_for": None,
    },
    {
        "src": "2_7_R_0",
        "dst": "0_2_R_11",
        "edge_kind": "junction_connector",
        "lane_change": False,
        "confidence": 1.0,
        "source_hint": "manual_connector_override",
        "priority": None,
        "signal_group": None,
        "legal_for": None,
    },
    {
        "src": "2_7_R_0",
        "dst": "2_3_L_6",
        "edge_kind": "junction_connector",
        "lane_change": False,
        "confidence": 1.0,
        "source_hint": "manual_connector_override",
        "priority": None,
        "signal_group": None,
        "legal_for": None,
    },
    {
        "src": "7_9_L_0",
        "dst": "9_14_L_6",
        "edge_kind": "junction_connector",
        "lane_change": False,
        "confidence": 1.0,
        "source_hint": "manual_connector_override",
        "priority": None,
        "signal_group": None,
        "legal_for": None,
    },
    {
        "src": "9_10_R_0",
        "dst": "7_9_R_25",
        "edge_kind": "junction_connector",
        "lane_change": False,
        "confidence": 1.0,
        "source_hint": "manual_connector_override",
        "priority": None,
        "signal_group": None,
        "legal_for": None,
    },
    {
        "src": "9_10_R_0",
        "dst": "9_14_L_6",
        "edge_kind": "junction_connector",
        "lane_change": False,
        "confidence": 1.0,
        "source_hint": "manual_connector_override",
        "priority": None,
        "signal_group": None,
        "legal_for": None,
    },
    {
        "src": "9_14_R_0",
        "dst": "7_9_R_25",
        "edge_kind": "junction_connector",
        "lane_change": False,
        "confidence": 1.0,
        "source_hint": "manual_connector_override",
        "priority": None,
        "signal_group": None,
        "legal_for": None,
    },
    {
        "src": "9_14_R_0",
        "dst": "9_10_L_6",
        "edge_kind": "junction_connector",
        "lane_change": False,
        "confidence": 1.0,
        "source_hint": "manual_connector_override",
        "priority": None,
        "signal_group": None,
        "legal_for": None,
    },
    {
        "src": "2_3_L_0",
        "dst": "3_4_L_9",
        "edge_kind": "junction_connector",
        "lane_change": False,
        "confidence": 1.0,
        "source_hint": "manual_connector_override",
        "priority": None,
        "signal_group": None,
        "legal_for": None,
    },
    {
        "src": "2_3_L_0",
        "dst": "3_8_L_0",
        "edge_kind": "junction_connector",
        "lane_change": False,
        "confidence": 1.0,
        "source_hint": "manual_connector_override",
        "priority": None,
        "signal_group": None,
        "legal_for": None,
    },
    {
        "src": "3_4_R_0",
        "dst": "2_3_R_6",
        "edge_kind": "junction_connector",
        "lane_change": False,
        "confidence": 1.0,
        "source_hint": "manual_connector_override",
        "priority": None,
        "signal_group": None,
        "legal_for": None,
    },
    {
        "src": "3_4_R_0",
        "dst": "3_8_L_0",
        "edge_kind": "junction_connector",
        "lane_change": False,
        "confidence": 1.0,
        "source_hint": "manual_connector_override",
        "priority": None,
        "signal_group": None,
        "legal_for": None,
    },
    {
        "src": "3_8_R_7",
        "dst": "2_3_R_6",
        "edge_kind": "junction_connector",
        "lane_change": False,
        "confidence": 1.0,
        "source_hint": "manual_connector_override",
        "priority": None,
        "signal_group": None,
        "legal_for": None,
    },
    {
        "src": "3_8_R_7",
        "dst": "3_4_L_9",
        "edge_kind": "junction_connector",
        "lane_change": False,
        "confidence": 1.0,
        "source_hint": "manual_connector_override",
        "priority": None,
        "signal_group": None,
        "legal_for": None,
    },
    {
        "src": "3_4_L_0",
        "dst": "4_5_L_7",
        "edge_kind": "junction_connector",
        "lane_change": False,
        "confidence": 1.0,
        "source_hint": "manual_connector_override",
        "priority": None,
        "signal_group": None,
        "legal_for": None,
    },
    {
        "src": "3_4_L_0",
        "dst": "4_11_L_0",
        "edge_kind": "junction_connector",
        "lane_change": False,
        "confidence": 1.0,
        "source_hint": "manual_connector_override",
        "priority": None,
        "signal_group": None,
        "legal_for": None,
    },
    {
        "src": "4_5_R_0",
        "dst": "3_4_R_9",
        "edge_kind": "junction_connector",
        "lane_change": False,
        "confidence": 1.0,
        "source_hint": "manual_connector_override",
        "priority": None,
        "signal_group": None,
        "legal_for": None,
    },
    {
        "src": "4_5_R_0",
        "dst": "4_11_L_0",
        "edge_kind": "junction_connector",
        "lane_change": False,
        "confidence": 1.0,
        "source_hint": "manual_connector_override",
        "priority": None,
        "signal_group": None,
        "legal_for": None,
    },
    {
        "src": "4_11_R_36",
        "dst": "3_4_R_9",
        "edge_kind": "junction_connector",
        "lane_change": False,
        "confidence": 1.0,
        "source_hint": "manual_connector_override",
        "priority": None,
        "signal_group": None,
        "legal_for": None,
    },
    {
        "src": "4_11_R_36",
        "dst": "4_5_L_7",
        "edge_kind": "junction_connector",
        "lane_change": False,
        "confidence": 1.0,
        "source_hint": "manual_connector_override",
        "priority": None,
        "signal_group": None,
        "legal_for": None,
    },
    {
        "src": "4_5_L_0",
        "dst": "5_6_L_17",
        "edge_kind": "junction_connector",
        "lane_change": False,
        "confidence": 1.0,
        "source_hint": "manual_connector_override",
        "priority": None,
        "signal_group": None,
        "legal_for": None,
    },
    {
        "src": "4_5_L_0",
        "dst": "5_12_L_0",
        "edge_kind": "junction_connector",
        "lane_change": False,
        "confidence": 1.0,
        "source_hint": "manual_connector_override",
        "priority": None,
        "signal_group": None,
        "legal_for": None,
    },
    {
        "src": "5_6_R_0",
        "dst": "4_5_R_7",
        "edge_kind": "junction_connector",
        "lane_change": False,
        "confidence": 1.0,
        "source_hint": "manual_connector_override",
        "priority": None,
        "signal_group": None,
        "legal_for": None,
    },
    {
        "src": "5_6_R_0",
        "dst": "5_12_L_0",
        "edge_kind": "junction_connector",
        "lane_change": False,
        "confidence": 1.0,
        "source_hint": "manual_connector_override",
        "priority": None,
        "signal_group": None,
        "legal_for": None,
    },
    {
        "src": "5_12_R_36",
        "dst": "4_5_R_7",
        "edge_kind": "junction_connector",
        "lane_change": False,
        "confidence": 1.0,
        "source_hint": "manual_connector_override",
        "priority": None,
        "signal_group": None,
        "legal_for": None,
    },
    {
        "src": "5_12_R_36",
        "dst": "5_6_L_17",
        "edge_kind": "junction_connector",
        "lane_change": False,
        "confidence": 1.0,
        "source_hint": "manual_connector_override",
        "priority": None,
        "signal_group": None,
        "legal_for": None,
    },
)
_CORRIDOR_RE = re.compile(r"^(?P<a>\d+)[_-](?P<b>\d+)_(?P<lane>[LR])(?:_(?P<idx>\d+))?$")


@dataclass(frozen=True)
class CanonicalGraphSummary:
    node_count: int
    edge_count: int
    weak_component_count: int
    junction_count: int
    junction_connector_count: int
    flow_groups: dict[str, int]


@dataclass(frozen=True)
class JunctionJoinRecord:
    junction_id: str
    src: str
    dst: str
    src_side: str
    dst_side: str
    turn_class: str
    lateral_error: float
    is_best_match: bool
    is_suspicious: bool
    reason: str


def _town_asset_dir(town_name: str = DEFAULT_TOWN_NAME) -> Path:
    return Path(asset_path) / town_name


def _load_graph(path: Path) -> nx.Graph:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _as_xy(position) -> np.ndarray:
    return np.array([float(position[0]), float(position[1])], dtype=float)


def _flow_group_from_lane(lane: str | None) -> str:
    if lane == "L":
        return "forward_a"
    if lane == "R":
        return "forward_b"
    return "unknown"


def _corridor_id(node_id: str, data: dict) -> str:
    segment = data.get("segment")
    if isinstance(segment, tuple) and len(segment) == 2:
        return f"{segment[0]}_{segment[1]}"
    match = _CORRIDOR_RE.match(str(node_id))
    if match:
        return f"{match.group('a')}_{match.group('b')}"
    return str(node_id).rsplit("_", 1)[0]


def _node_index(node_id: str) -> int | None:
    match = _CORRIDOR_RE.match(str(node_id))
    if match and match.group("idx") is not None:
        return int(match.group("idx"))
    return None


def _local_heading(graph: nx.DiGraph, node_id: str) -> float | None:
    center = _as_xy(graph.nodes[node_id]["pos"])
    vectors: list[np.ndarray] = []

    for succ in graph.successors(node_id):
        vectors.append(_as_xy(graph.nodes[succ]["pos"]) - center)
    for pred in graph.predecessors(node_id):
        vectors.append(center - _as_xy(graph.nodes[pred]["pos"]))

    vectors = [vec for vec in vectors if np.linalg.norm(vec) > 1e-6]
    if not vectors:
        return None

    mean_vec = np.mean(np.stack(vectors), axis=0)
    if np.linalg.norm(mean_vec) <= 1e-6:
        return None
    # The legacy Town01 directed assets encode lane progression opposite to the
    # visually expected traffic flow on the raster map, so we flip the heading
    # by pi to align the canonical graph's displayed flow with the map.
    return float(math.atan2(mean_vec[1], mean_vec[0]) + math.pi)


def _angle_delta(target: float | None, source: float | None) -> float | None:
    if target is None or source is None:
        return None
    delta = target - source
    return float((delta + math.pi) % (2.0 * math.pi) - math.pi)


def _turn_class_from_delta(delta: float | None, *, straight_thresh: float = 0.35) -> str:
    if delta is None:
        return "unknown"
    if abs(delta) <= straight_thresh:
        return "straight"
    return "left" if delta > 0.0 else "right"


def _junction_anchor_nodes(backbone: nx.DiGraph) -> dict[str, np.ndarray]:
    anchors = {}
    for node_id, data in backbone.nodes(data=True):
        if data.get("type") == "intersection":
            anchors[str(node_id)] = _as_xy(data["pos"])
    return anchors


def _nearest_junction_id(
    position: np.ndarray,
    anchors: dict[str, np.ndarray],
    *,
    radius_raw: float,
) -> str | None:
    best_id = None
    best_dist = float("inf")
    for junction_id, anchor in anchors.items():
        dist = float(np.linalg.norm(position - anchor))
        if dist <= radius_raw and dist < best_dist:
            best_id = junction_id
            best_dist = dist
    return best_id


def _node_kind(
    *,
    indegree: int,
    outdegree: int,
    junction_id: str | None,
) -> str:
    if indegree == 0 and outdegree == 0:
        return "dead_end"
    if junction_id is None:
        return "dead_end" if indegree == 0 or outdegree == 0 else "lane"
    if indegree <= 1 and outdegree > 1:
        return "junction_entry"
    if indegree > 1 and outdegree <= 1:
        return "junction_exit"
    return "junction_inner"


def _swap_lane_label(lane: str | None) -> str | None:
    if lane == "L":
        return "R"
    if lane == "R":
        return "L"
    return lane


def _refresh_node_kinds(graph: nx.DiGraph) -> None:
    for node_id, data in graph.nodes(data=True):
        indegree = graph.in_degree(node_id)
        outdegree = graph.out_degree(node_id)
        data["node_kind"] = _node_kind(
            indegree=int(indegree),
            outdegree=int(outdegree),
            junction_id=data.get("junction_id"),
        )


def _refresh_edge_turn_classes(graph: nx.DiGraph) -> None:
    for src, dst, data in graph.edges(data=True):
        turn_delta = _angle_delta(
            graph.nodes[dst].get("heading_rad"),
            graph.nodes[src].get("heading_rad"),
        )
        data["turn_class"] = _turn_class_from_delta(turn_delta)


def _apply_manual_flow_inversions(
    graph: nx.DiGraph,
    *,
    corridor_ids: set[str],
) -> None:
    if not corridor_ids:
        return

    for _, data in graph.nodes(data=True):
        if data.get("corridor_id") not in corridor_ids:
            continue
        heading = data.get("heading_rad")
        if heading is None:
            continue
        data["heading_rad"] = float((float(heading) + math.pi) % (2.0 * math.pi))

    reversed_edges = []
    to_remove = []
    for src, dst, data in graph.edges(data=True):
        src_corridor = graph.nodes[src].get("corridor_id")
        dst_corridor = graph.nodes[dst].get("corridor_id")
        if src_corridor in corridor_ids and dst_corridor in corridor_ids and src_corridor == dst_corridor:
            reversed_edges.append((dst, src, dict(data)))
            to_remove.append((src, dst))

    if to_remove:
        graph.remove_edges_from(to_remove)
        for src, dst, data in reversed_edges:
            graph.add_edge(src, dst, **data)

    _refresh_node_kinds(graph)
    _refresh_edge_turn_classes(graph)


def _apply_manual_label_swaps(
    graph: nx.DiGraph,
    *,
    corridor_ids: set[str],
) -> None:
    if not corridor_ids:
        return
    for _, data in graph.nodes(data=True):
        if data.get("corridor_id") not in corridor_ids:
            continue
        data["legacy_lane"] = str(_swap_lane_label(data.get("legacy_lane")))


def _apply_manual_node_id_relabels(
    graph: nx.DiGraph,
    *,
    corridor_ids: set[str],
) -> nx.DiGraph:
    if not corridor_ids:
        return graph

    mapping: dict[str, str] = {}
    for node_id, data in graph.nodes(data=True):
        corridor_id = data.get("corridor_id")
        if corridor_id not in corridor_ids:
            continue
        match = _CORRIDOR_RE.match(str(node_id))
        if match is None:
            continue
        desired_lane = "R" if data.get("flow_group") == "forward_b" else "L"
        current_lane = match.group("lane")
        if current_lane == desired_lane:
            continue
        mapping[str(node_id)] = f"{match.group('a')}_{match.group('b')}_{desired_lane}_{match.group('idx')}"

    if not mapping:
        return graph

    relabeled = nx.DiGraph()
    for node_id, data in graph.nodes(data=True):
        new_node_id = mapping.get(str(node_id), str(node_id))
        new_data = dict(data)
        new_data["node_id"] = str(new_node_id)
        relabeled.add_node(str(new_node_id), **new_data)

    for src, dst, data in graph.edges(data=True):
        new_src = mapping.get(str(src), str(src))
        new_dst = mapping.get(str(dst), str(dst))
        relabeled.add_edge(str(new_src), str(new_dst), **dict(data))

    return relabeled


def _realign_lane_labels_to_flow(
    graph: nx.DiGraph,
    *,
    corridor_ids: set[str],
) -> None:
    if not corridor_ids:
        return
    for _, data in graph.nodes(data=True):
        if data.get("corridor_id") not in corridor_ids:
            continue
        flow_group = data.get("flow_group")
        if flow_group == "forward_a":
            data["legacy_lane"] = "L"
        elif flow_group == "forward_b":
            data["legacy_lane"] = "R"


def _refresh_junction_membership(
    graph: nx.DiGraph,
    *,
    anchors: dict[str, np.ndarray],
    junction_radius_raw: float,
) -> None:
    for node_id, data in graph.nodes(data=True):
        pos_raw = _as_xy(data["pos_raw"])
        junction_id = MANUAL_NODE_JUNCTION_OVERRIDES.get(str(node_id))
        if junction_id is None:
            junction_id = _nearest_junction_id(pos_raw, anchors, radius_raw=junction_radius_raw)
        data["junction_id"] = junction_id
        data["node_kind"] = _node_kind(
            indegree=graph.in_degree(node_id),
            outdegree=graph.out_degree(node_id),
            junction_id=junction_id,
        )


def _apply_manual_corridor_side_swaps(
    graph: nx.DiGraph,
    *,
    corridor_ids: set[str],
) -> nx.DiGraph:
    if not corridor_ids:
        return graph

    for corridor_id in corridor_ids:
        lane_pairs: dict[int, dict[str, str]] = {}
        for node_id, data in graph.nodes(data=True):
            if data.get("corridor_id") != corridor_id:
                continue
            lane = str(data.get("legacy_lane"))
            index = data.get("legacy_index")
            if lane not in {"L", "R"} or index is None:
                continue
            lane_pairs.setdefault(int(index), {})[lane] = str(node_id)

        for pair in lane_pairs.values():
            left = pair.get("L")
            right = pair.get("R")
            if left is None or right is None:
                continue
            left_data = graph.nodes[left]
            right_data = graph.nodes[right]
            left_pos_raw = tuple(left_data["pos_raw"])
            left_pos_surface = tuple(left_data["pos_surface"])
            right_pos_raw = tuple(right_data["pos_raw"])
            right_pos_surface = tuple(right_data["pos_surface"])

            left_data["pos_raw"] = right_pos_raw
            left_data["pos_surface"] = right_pos_surface
            left_data["legacy_lane"] = "R"

            right_data["pos_raw"] = left_pos_raw
            right_data["pos_surface"] = left_pos_surface
            right_data["legacy_lane"] = "L"

    mapping: dict[str, str] = {}
    for node_id, data in graph.nodes(data=True):
        if data.get("corridor_id") not in corridor_ids:
            continue
        match = _CORRIDOR_RE.match(str(node_id))
        if match is None:
            continue
        swapped_lane = _swap_lane_label(match.group("lane"))
        mapping[str(node_id)] = f"{match.group('a')}_{match.group('b')}_{swapped_lane}_{match.group('idx')}"

    if not mapping:
        return graph

    relabeled = nx.DiGraph()
    for node_id, data in graph.nodes(data=True):
        new_node_id = mapping.get(str(node_id), str(node_id))
        new_data = dict(data)
        new_data["node_id"] = str(new_node_id)
        relabeled.add_node(str(new_node_id), **new_data)

    for src, dst, data in graph.edges(data=True):
        new_src = mapping.get(str(src), str(src))
        new_dst = mapping.get(str(dst), str(dst))
        relabeled.add_edge(str(new_src), str(new_dst), **dict(data))

    return relabeled


def _apply_manual_connector_additions(graph: nx.DiGraph) -> None:
    for spec in MANUAL_CONNECTOR_ADDITIONS:
        src = str(spec["src"])
        dst = str(spec["dst"])
        if src not in graph or dst not in graph:
            continue
        src_pos = np.array(graph.nodes[src]["pos_raw"], dtype=float)
        dst_pos = np.array(graph.nodes[dst]["pos_raw"], dtype=float)
        length_m = float(np.linalg.norm(dst_pos - src_pos) * RAW_METERS_PER_PIXEL)
        turn_delta = _angle_delta(
            graph.nodes[dst].get("heading_rad"),
            graph.nodes[src].get("heading_rad"),
        )
        payload = dict(spec)
        payload.pop("src", None)
        payload.pop("dst", None)
        payload["length_m"] = length_m
        payload["cost_m"] = max(length_m, 0.05)
        payload["turn_class"] = _turn_class_from_delta(turn_delta)
        graph.add_edge(src, dst, **payload)


def build_town01_canonical_vehicle_graph(
    *,
    town_name: str = DEFAULT_TOWN_NAME,
    junction_radius_raw: float = 240.0,
    lane_change_radius_raw: float = 96.0,
    add_lane_change_edges: bool = False,
) -> nx.DiGraph:
    asset_dir = _town_asset_dir(town_name)
    backbone = _load_graph(asset_dir / LEGACY_BACKBONE_NAME)
    if not isinstance(backbone, nx.DiGraph):
        raise TypeError(f"Expected directed legacy backbone graph, got {type(backbone).__name__}")

    canonical = nx.DiGraph()
    anchors = _junction_anchor_nodes(backbone)

    for node_id, data in backbone.nodes(data=True):
        original_lane = data.get("lane")
        if original_lane not in {"L", "R"}:
            continue

        corridor_id = _corridor_id(str(node_id), data)
        flow_lane = (
            _swap_lane_label(original_lane)
            if corridor_id in MANUAL_CORRIDOR_FLOW_LANE_SWAPS
            else original_lane
        )
        pos_raw = _as_xy(data["pos"])
        pos_surface = raw_to_surface(pos_raw)
        junction_id = _nearest_junction_id(pos_raw, anchors, radius_raw=junction_radius_raw)
        indegree = sum(1 for pred in backbone.predecessors(node_id) if backbone.nodes[pred].get("lane") in {"L", "R"})
        outdegree = sum(1 for succ in backbone.successors(node_id) if backbone.nodes[succ].get("lane") in {"L", "R"})
        heading = _local_heading(backbone, node_id)
        confidence = 0.95 if heading is not None else 0.75
        canonical.add_node(
            node_id,
            node_id=str(node_id),
            pos_raw=(float(pos_raw[0]), float(pos_raw[1])),
            pos_surface=(float(pos_surface[0]), float(pos_surface[1])),
            heading_rad=heading,
            flow_group=_flow_group_from_lane(flow_lane),
            corridor_id=corridor_id,
            legacy_lane=str(original_lane),
            legacy_index=_node_index(str(node_id)),
            junction_id=junction_id,
            node_kind=_node_kind(indegree=indegree, outdegree=outdegree, junction_id=junction_id),
            confidence=confidence,
            canonical_road_id=None,
            canonical_section_id=None,
            canonical_lane_id=None,
            canonical_s=None,
        )

    for src, dst, data in backbone.edges(data=True):
        if src not in canonical or dst not in canonical:
            continue
        src_pos = _as_xy(backbone.nodes[src]["pos"])
        dst_pos = _as_xy(backbone.nodes[dst]["pos"])
        dist_raw = float(np.linalg.norm(dst_pos - src_pos))
        length_m = dist_raw * RAW_METERS_PER_PIXEL
        src_corridor = canonical.nodes[src]["corridor_id"]
        dst_corridor = canonical.nodes[dst]["corridor_id"]
        src_junction = canonical.nodes[src]["junction_id"]
        dst_junction = canonical.nodes[dst]["junction_id"]
        edge_kind = "forward"
        source_hint = "legacy_graph"
        confidence = 0.95

        if src_corridor != dst_corridor and (src_junction is not None or dst_junction is not None):
            edge_kind = "junction_connector"
            source_hint = "legacy_graph_junction"
            confidence = 0.9

        junction_key = str(src_junction or dst_junction) if (src_junction or dst_junction) is not None else None
        exclusion_key = (
            junction_key,
            src_corridor,
            dst_corridor,
            str(canonical.nodes[src]["flow_group"]),
        )
        if edge_kind == "junction_connector" and exclusion_key in MANUAL_CONNECTOR_EXCLUSIONS:
            continue

        turn_delta = _angle_delta(
            canonical.nodes[dst]["heading_rad"],
            canonical.nodes[src]["heading_rad"],
        )

        canonical.add_edge(
            src,
            dst,
            length_m=length_m,
            cost_m=max(length_m, 0.05),
            edge_kind=edge_kind,
            turn_class=_turn_class_from_delta(turn_delta),
            lane_change=False,
            confidence=confidence,
            source_hint=source_hint,
            priority=None,
            signal_group=None,
            legal_for=None,
        )

    if add_lane_change_edges:
        _add_lane_change_edges(canonical, lane_change_radius_raw=lane_change_radius_raw)

    _apply_manual_flow_inversions(
        canonical,
        corridor_ids=MANUAL_CORRIDOR_FLOW_INVERSIONS,
    )
    _apply_manual_label_swaps(
        canonical,
        corridor_ids=MANUAL_CORRIDOR_LABEL_SWAPS,
    )
    canonical = _apply_manual_node_id_relabels(
        canonical,
        corridor_ids=MANUAL_CORRIDOR_LABEL_SWAPS,
    )
    canonical = _apply_manual_corridor_side_swaps(
        canonical,
        corridor_ids=MANUAL_CORRIDOR_SIDE_SWAPS,
    )
    _realign_lane_labels_to_flow(
        canonical,
        corridor_ids=MANUAL_CORRIDOR_SIDE_SWAPS,
    )
    canonical = _apply_manual_node_id_relabels(
        canonical,
        corridor_ids=MANUAL_CORRIDOR_SIDE_SWAPS,
    )
    _refresh_junction_membership(
        canonical,
        anchors=anchors,
        junction_radius_raw=junction_radius_raw,
    )
    _apply_manual_connector_additions(canonical)

    return canonical


def _add_lane_change_edges(canonical: nx.DiGraph, *, lane_change_radius_raw: float) -> None:
    nodes = list(canonical.nodes)
    for idx, src in enumerate(nodes):
        src_data = canonical.nodes[src]
        src_pos = np.array(src_data["pos_raw"], dtype=float)
        src_heading = src_data["heading_rad"]
        if src_heading is None:
            continue

        for dst in nodes[idx + 1 :]:
            dst_data = canonical.nodes[dst]
            if src_data["flow_group"] != dst_data["flow_group"]:
                continue
            if src_data["corridor_id"] == dst_data["corridor_id"]:
                continue
            dst_heading = dst_data["heading_rad"]
            if dst_heading is None:
                continue
            heading_delta = abs(_angle_delta(dst_heading, src_heading) or math.pi)
            if heading_delta > 0.25:
                continue
            dst_pos = np.array(dst_data["pos_raw"], dtype=float)
            lateral = float(np.linalg.norm(dst_pos - src_pos))
            if lateral > lane_change_radius_raw:
                continue
            length_m = lateral * RAW_METERS_PER_PIXEL
            turn_class = _turn_class_from_delta(_angle_delta(dst_heading, src_heading))
            for a, b in ((src, dst), (dst, src)):
                if canonical.has_edge(a, b):
                    continue
                canonical.add_edge(
                    a,
                    b,
                    length_m=length_m,
                    cost_m=max(3.0 * length_m, 0.25),
                    edge_kind="lane_change_candidate",
                    turn_class=turn_class,
                    lane_change=True,
                    confidence=0.55,
                    source_hint="lane_change_inference",
                    priority=None,
                    signal_group=None,
                    legal_for=None,
                )


def summarize_canonical_graph(graph: nx.DiGraph) -> CanonicalGraphSummary:
    flow_groups: dict[str, int] = {}
    junction_ids = {
        data["junction_id"]
        for _, data in graph.nodes(data=True)
        if data.get("junction_id") is not None
    }
    for _, data in graph.nodes(data=True):
        key = str(data.get("flow_group", "unknown"))
        flow_groups[key] = flow_groups.get(key, 0) + 1
    connector_count = sum(
        1 for _, _, data in graph.edges(data=True) if data.get("edge_kind") == "junction_connector"
    )
    return CanonicalGraphSummary(
        node_count=graph.number_of_nodes(),
        edge_count=graph.number_of_edges(),
        weak_component_count=nx.number_weakly_connected_components(graph),
        junction_count=len(junction_ids),
        junction_connector_count=connector_count,
        flow_groups=flow_groups,
    )


def validate_town01_canonical_vehicle_graph(graph: nx.DiGraph) -> list[str]:
    problems: list[str] = []
    if not isinstance(graph, nx.DiGraph):
        problems.append(f"Graph must be a DiGraph, got {type(graph).__name__}")
        return problems

    required_node_fields = {
        "node_id",
        "pos_raw",
        "pos_surface",
        "flow_group",
        "corridor_id",
        "node_kind",
        "confidence",
    }
    required_edge_fields = {
        "length_m",
        "cost_m",
        "edge_kind",
        "turn_class",
        "lane_change",
        "confidence",
    }

    for node_id, data in graph.nodes(data=True):
        missing = sorted(field for field in required_node_fields if field not in data)
        if missing:
            problems.append(f"Node {node_id} missing required fields: {missing}")
            break

    for src, dst, data in graph.edges(data=True):
        missing = sorted(field for field in required_edge_fields if field not in data)
        if missing:
            problems.append(f"Edge {src}->{dst} missing required fields: {missing}")
            break

    isolated_junction_nodes = [
        node_id
        for node_id, data in graph.nodes(data=True)
        if data.get("junction_id") is not None and graph.in_degree(node_id) == 0 and graph.out_degree(node_id) == 0
    ]
    if isolated_junction_nodes:
        problems.append(
            f"Found isolated junction nodes in canonical graph: {isolated_junction_nodes[:10]}"
        )

    if graph.number_of_nodes() == 0 or graph.number_of_edges() == 0:
        problems.append("Canonical graph is empty.")
        return problems

    weak_component_sizes = sorted(
        (len(component) for component in nx.weakly_connected_components(graph)),
        reverse=True,
    )
    major_component_floor = max(50, int(0.4 * graph.number_of_nodes()))
    small_component_ceiling = 30
    if weak_component_sizes and weak_component_sizes[0] < major_component_floor:
        problems.append(
            "Largest weakly connected component is unexpectedly small for the canonical Town01 graph."
        )
    elif len(weak_component_sizes) >= 2 and weak_component_sizes[1] >= major_component_floor:
        if len(weak_component_sizes) > 2:
            problems.append(
                f"Expected at most two major weak components in the staged canonical graph, got {weak_component_sizes}."
            )
    elif any(size > small_component_ceiling for size in weak_component_sizes[1:]):
        problems.append(
            f"Unexpected medium-sized weak components in canonical Town01 graph: {weak_component_sizes}."
        )

    flow_groups = {data.get("flow_group") for _, data in graph.nodes(data=True)}
    if not {"forward_a", "forward_b"}.issubset(flow_groups):
        problems.append(f"Expected both flow groups in canonical graph, got {sorted(flow_groups)}")

    if not any(data.get("edge_kind") == "junction_connector" for _, _, data in graph.edges(data=True)):
        problems.append("Canonical graph does not contain any junction connector edges.")

    return problems


def junction_centroids(graph: nx.DiGraph) -> dict[str, tuple[float, float]]:
    grouped: dict[str, list[tuple[float, float]]] = {}
    for _, data in graph.nodes(data=True):
        junction_id = data.get("junction_id")
        if junction_id is None:
            continue
        grouped.setdefault(str(junction_id), []).append(tuple(data["pos_raw"]))

    centroids = {}
    for junction_id, points in grouped.items():
        arr = np.asarray(points, dtype=float)
        centroids[junction_id] = (float(arr[:, 0].mean()), float(arr[:, 1].mean()))
    return centroids


def _junction_side(position: tuple[float, float], center: tuple[float, float]) -> str:
    dx = float(position[0] - center[0])
    dy = float(position[1] - center[1])
    if abs(dx) > abs(dy):
        return "east" if dx > 0.0 else "west"
    return "south" if dy > 0.0 else "north"


def _side_lateral_value(side: str, position: tuple[float, float]) -> float:
    x, y = position
    if side in {"north", "south"}:
        return float(x)
    return float(y)


def validate_junction_joins(
    graph: nx.DiGraph,
    *,
    straight_tolerance_raw: float = 140.0,
) -> list[JunctionJoinRecord]:
    centroids = junction_centroids(graph)
    records: list[JunctionJoinRecord] = []

    outgoing_by_source: dict[tuple[str, str], list[tuple[str, str, float]]] = {}
    for src, dst, data in graph.edges(data=True):
        if data.get("edge_kind") != "junction_connector":
            continue
        junction_id = graph.nodes[src].get("junction_id") or graph.nodes[dst].get("junction_id")
        if junction_id is None or junction_id not in centroids:
            continue
        center = centroids[str(junction_id)]
        src_side = _junction_side(tuple(graph.nodes[src]["pos_raw"]), center)
        dst_side = _junction_side(tuple(graph.nodes[dst]["pos_raw"]), center)
        lateral = abs(
            _side_lateral_value(src_side, tuple(graph.nodes[src]["pos_raw"]))
            - _side_lateral_value(dst_side, tuple(graph.nodes[dst]["pos_raw"]))
        )
        outgoing_by_source.setdefault((str(junction_id), str(src)), []).append((str(dst), dst_side, lateral))

    for src, dst, data in graph.edges(data=True):
        if data.get("edge_kind") != "junction_connector":
            continue
        junction_id = graph.nodes[src].get("junction_id") or graph.nodes[dst].get("junction_id")
        if junction_id is None or str(junction_id) not in centroids:
            continue

        center = centroids[str(junction_id)]
        src_pos = tuple(graph.nodes[src]["pos_raw"])
        dst_pos = tuple(graph.nodes[dst]["pos_raw"])
        src_side = _junction_side(src_pos, center)
        dst_side = _junction_side(dst_pos, center)
        turn_class = str(data.get("turn_class", "unknown"))
        lateral_error = abs(
            _side_lateral_value(src_side, src_pos) - _side_lateral_value(dst_side, dst_pos)
        )
        candidate_laterals = [
            lateral for _, candidate_side, lateral in outgoing_by_source[(str(junction_id), str(src))]
            if candidate_side == dst_side
        ]
        best_lateral = min(candidate_laterals) if candidate_laterals else lateral_error
        is_best_match = abs(lateral_error - best_lateral) <= 1e-6

        suspicious = False
        reason = "ok"
        if src_side == dst_side:
            reason = "connector_keeps_same_side"
        elif turn_class == "straight" and lateral_error > straight_tolerance_raw:
            suspicious = True
            reason = "straight_connector_large_lateral_error"
        elif not is_best_match and turn_class == "straight":
            suspicious = True
            reason = "straight_connector_not_nearest_lane_match"

        if str(junction_id) in MANUAL_VALIDATED_JUNCTIONS:
            suspicious = False
            reason = "manually_validated_junction"

        records.append(
            JunctionJoinRecord(
                junction_id=str(junction_id),
                src=str(src),
                dst=str(dst),
                src_side=src_side,
                dst_side=dst_side,
                turn_class=turn_class,
                lateral_error=float(lateral_error),
                is_best_match=bool(is_best_match),
                is_suspicious=bool(suspicious),
                reason=reason,
            )
        )

    return records


def save_town01_canonical_vehicle_graph(
    *,
    town_name: str = DEFAULT_TOWN_NAME,
    output_path: str | Path | None = None,
    add_lane_change_edges: bool = False,
) -> Path:
    graph = build_town01_canonical_vehicle_graph(
        town_name=town_name,
        add_lane_change_edges=add_lane_change_edges,
    )
    problems = validate_town01_canonical_vehicle_graph(graph)
    if problems:
        raise ValueError(f"Canonical Town01 graph validation failed: {problems}")

    if output_path is None:
        output = _town_asset_dir(town_name) / DEFAULT_GRAPH_NAME
    else:
        output = Path(output_path)
    with output.open("wb") as handle:
        pickle.dump(graph, handle)
    return output

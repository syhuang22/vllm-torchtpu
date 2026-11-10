# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""2D phase map, local FileReader calibration dropzone, and bucket export UI."""

from __future__ import annotations

from .assets import load_asset


def render_phase_stage_tab_button() -> str:
    """Return the stage tab button for the 2D Tokens x Context Phase Map."""
    return (
        '<button class="wb-tab-btn" id="btn-stage-phase" type="button" '
        "onclick=\"wbSetStageTab('phase')\">"
        "2D Tokens &times; Context Phase Map &amp; Bucket Schedule</button>"
    )


def render_phase_extension_js() -> str:
    """Return ES2022 logic for 2D phase heatmap, FileReader dropzone, and export."""
    return load_asset("workbench_phase.js")

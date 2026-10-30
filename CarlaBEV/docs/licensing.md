# Licensing and attribution

## CarlaBEV software

CarlaBEV's original software and documentation are licensed under
[Apache-2.0](../../LICENSE.md). Copyright 2024-2026 CarlaBEV contributors.
The project attribution and third-party scope are recorded in [NOTICE](../../NOTICE).

Apache-2.0 permits research and commercial use, modification, and redistribution,
including proprietary derivatives, subject to its conditions. It includes an
explicit contributor patent grant with a patent-litigation termination clause.
Redistributors must preserve required notices, provide the license, and identify
modified files. It does not grant trademark rights or require research citation.
See the [authoritative Apache license](https://www.apache.org/licenses/LICENSE-2.0).

This choice supports broad use of the simulator by researchers and commercial
teams. It does not require downstream users to publish their modifications.
Projects that require reciprocal publication should consider a copyleft license
before adopting that policy; this project currently chooses permissive reuse.

## Included third-party code

Planning and control modules contain adaptations attributed to Atsushi Sakai and
the PythonRobotics contributors. Their [MIT license](../../LICENSES/PythonRobotics-MIT.txt)
and existing source credits are retained. The root Apache license does not erase
those notices or rights. See [PythonRobotics](https://github.com/AtsushiSakai/PythonRobotics).

## Map assets

The maintainer created the Town01 raster maps and graph files from CARLA.
Their [asset notice](../assets/Town01/NOTICE) records CARLA attribution and the
local processing. CARLA publishes its software under MIT and its own assets
under CC-BY; those are separate licensing scopes. See
[CARLA's licensing statement](https://github.com/carla-simulator/carla#licenses).

The exact CARLA release and CC-BY version for these exports are not recorded.
The asset notice preserves this limitation instead of assuming a license version.
Recover the original release's asset-license text before claiming a fully
verified licensing inventory or distributing a standalone Town01 dataset.
Adding the software license alone does not complete asset provenance.

## Dependencies and distribution

Dependencies listed in `pyproject.toml` retain their own licenses. Their presence
does not make them Apache-licensed. Wheels and source distributions include the
root license, attribution notice, retained PythonRobotics license, and Town01
asset notice. Keep these files together when redistributing the project.

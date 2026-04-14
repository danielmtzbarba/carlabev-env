# CarlaBEV Scenario Specifications

This document outlines the architecture, variables, and progressive complexity levels of the predefined driving scenarios within the `CarlaBEV` framework. These specifications dictate how the Scene Designer tool should parameterize its UI forms.

---

## 1. Jaywalk Scenario (`jaywalk.py`)
**Description:** A pedestrian crosses the road ahead of the ego vehicle. The unpredictable behavior of the pedestrian creates an imminent collision risk if the ego vehicle fails to yield.

### Global Parameters
- **`ego_speed`**: (Float) Target speed of the ego vehicle (e.g., 40-70 m/s).
- **`level`**: (Int) 1 to 4 scaling difficulty.

### Actors & Properties
1. **Agent (Ego Vehicle)**
   - Start Position: Approaching an unmarked crossing.
   - Variables: `ego_speed`

2. **Pedestrian (Adversary)**
   - Start Position: Right curb (or dynamically offset).
   - Variables: `cross_delay` (time before stepping into road), `target_speed` (walking pace).
   - Behavior by Level:
     - **Level 1**: `CrossBehavior` (Walks linearly across).
     - **Level 2**: `StopMidBehavior` (Crosses but freezes into the ego lane).
     - **Level 3 & 4**: `StopReturnBehavior` (Crosses halfway, then turns around).

3. **Background Vehicle (Follower)** *(Triggered at Level 4)*
   - Start Position: Right behind Ego.
   - Variables: `rear_gap` (Distance behind ego), `rear_speed` (Matches or exceeds ego speed).

---

## 2. Lead Brake Scenario (`lead_brake.py`)
**Description:** The ego vehicle follows a leading vehicle that will suddenly execute an emergency or hard brake. Progressive levels add traffic to box the ego vehicle in.

### Global Parameters
- **`ego_speed`**: (Float) Target speed of the ego vehicle.
- **`level`**: (Int) 1 to 3 scaling difficulty.

### Actors & Properties
1. **Agent (Ego Vehicle)**
   - Variables: `ego_speed`

2. **Vehicle (Lead)**
   - Start Position: In front of ego.
   - Variables: `lead_gap` (Distance ahead of ego), `lead_speed` (Target velocity matches/exceeds ego), `brake_delay` (Time before slamming brakes), `brake_strength` (Deceleration Rate).
   - Behavior: `LeadBrakeBehavior`

3. **Vehicle (Left Lane Traffic)** *(Triggered at Level $\ge$ 2)*
   - Start Position: Parallel lane moving in the same direction or oncoming.
   - Variables: `left_speed`. (Denies lane-change escape).

4. **Vehicle (Follower Traffic)** *(Triggered at Level 3)*
   - Start Position: Behind Ego.
   - Variables: `rear_gap`, `brake_delay`, `brake_strength`.
   - Behavior: `LeadBrakeBehavior` (Tailgater).

---

## 3. Red Light Running Scenario (`red_light_running.py`)
**Description:** The ego vehicle approaches a four-way intersection with a green light, while an adversarial vehicle blows through the intersection from a perpendicular angle at high speed.

### Global Parameters
- **`level`**: (Int) Difficulty mapping (Currently defaults).

### Actors & Properties
1. **Agent (Ego Vehicle)**
   - Start Position: Approaching intersection (South $\rightarrow$ North).
   - Variables: `ego_speed` (Moderate, matching green light right-of-way).

2. **Vehicle (Adversary)**
   - Start Position: Approaching intersection (West $\rightarrow$ East).
   - Variables: `adv_speed` (Speeding aggressively).

3. **Traffic Lights (Static/Target Markers)**
   - `tl_ego`: Positioned for Ego (Green).
   - `tl_adv`: Positioned for Adversary (Red).

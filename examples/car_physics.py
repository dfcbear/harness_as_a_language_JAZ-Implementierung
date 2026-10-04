"""
car_physics.py - Comprehensive car vehicle dynamics and physics module.

Implements a Car class using a dynamic bicycle model with:
  - Simplified Pacejka "magic formula" tire models (lateral + longitudinal)
  - Combined slip tire force calculation (friction circle/ellipse)
  - Longitudinal and lateral weight transfer
  - Drivetrain simulation (engine torque curve, gear ratios, RPM)
  - Aerodynamic drag and downforce
  - Rolling resistance
  - Traction control (TC) and anti-lock braking (ABS)
  - Multiple integration methods (semi-implicit Euler, RK4)
  - Optional pygame-based top-down renderer

Conventions (from SHARED_SPEC): meters, seconds, radians. World space, x→right, y→up.
Fixed dt = 1/60 s recommended. Plain dataclasses, no hidden globals, no randomness.

Author: JAZ Agent
License: MIT
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Any, List

try:
    import pygame
except ImportError:  # pragma: no cover - pygame is optional for physics-only use
    pygame = None


# ---------------------------------------------------------------------------
# Helper math
# ---------------------------------------------------------------------------

def clamp(value: float, low: float, high: float) -> float:
    """Clamp *value* to the range [low, high]."""
    return max(low, min(high, value))


def lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation between *a* and *b* by factor *t* in [0, 1]."""
    return a + (b - a) * clamp(t, 0.0, 1.0)


def normalize_angle(angle: float) -> float:
    """Normalize an angle to the range [-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


def angle_difference(a: float, b: float) -> float:
    """Smallest signed difference from angle *a* to angle *b*."""
    return normalize_angle(b - a)


# ---------------------------------------------------------------------------
# Simplified Pacejka "magic formula" tire model
# ---------------------------------------------------------------------------

def pacejka_lateral(slip_angle: float,
                    B: float = 10.0,
                    C: float = 1.9,
                    D: float = 1.0,
                    E: float = 0.97) -> float:
    """
    Simplified Pacejka magic formula for lateral force coefficient.

    Fy = D * sin(C * atan(B * slip - E * (B * slip - atan(B * slip))))

    Returns a *normalized* lateral force coefficient (multiply by normal load
    to obtain force).
    """
    slip = clamp(slip_angle, -math.pi / 2, math.pi / 2)
    Bslip = B * slip
    return D * math.sin(C * math.atan(Bslip - E * (Bslip - math.atan(Bslip))))


def pacejka_longitudinal(slip_ratio: float,
                         B: float = 10.0,
                         C: float = 1.65,
                         D: float = 1.0,
                         E: float = 0.97) -> float:
    """
    Simplified Pacejka magic formula for longitudinal force coefficient.

    Returns a *normalized* longitudinal force coefficient.
    """
    s = clamp(slip_ratio, -1.0, 1.0)
    Bs = B * s
    return D * math.sin(C * math.atan(Bs - E * (Bs - math.atan(Bs))))


def combined_slip_pacejka(slip_angle: float,
                          slip_ratio: float,
                          mu: float = 1.0,
                          B_lat: float = 10.0,
                          C_lat: float = 1.9,
                          D_lat: float = 1.0,
                          E_lat: float = 0.97,
                          B_long: float = 10.0,
                          C_long: float = 1.65,
                          D_long: float = 1.0,
                          E_long: float = 0.97) -> Tuple[float, float]:
    """
    Combined-slip tire force using friction ellipse approximation.

    When both lateral and longitudinal slip are present, the available grip
    is shared via a friction ellipse:

        (Fx / Fx_max)^2 + (Fy / Fy_max)^2 <= 1

    We compute the pure-slip forces and then scale them so their vector
    lies on the ellipse boundary.

    Returns (Fx_normalized, Fy_normalized) — multiply by normal load for force.
    """
    fx_pure = pacejka_longitudinal(slip_ratio, B_long, C_long, D_long, E_long) * mu
    fy_pure = pacejka_lateral(slip_angle, B_lat, C_lat, D_lat, E_lat) * mu

    # Friction ellipse: scale down if combined demand exceeds available grip
    fx_max = max(abs(fx_pure), 1e-9)
    fy_max = max(abs(fy_pure), 1e-9)

    # Combined demand ratio
    ratio_sq = (fx_pure / fx_max) ** 2 + (fy_pure / fy_max) ** 2
    if ratio_sq > 1.0:
        scale = 1.0 / math.sqrt(ratio_sq)
        fx_pure *= scale
        fy_pure *= scale

    return fx_pure, fy_pure


# ---------------------------------------------------------------------------
# Drivetrain
# ---------------------------------------------------------------------------

@dataclass
class GearRatio:
    """A single gear ratio with its ratio value."""
    ratio: float       # gear ratio (e.g. 3.5 means 3.5:1)
    max_speed: float   # approximate max speed for this gear (m/s), for auto-shift


@dataclass
class DrivetrainParams:
    """Drivetrain and engine parameters."""
    # Engine torque curve: list of (rpm, torque_Nm) points
    # Torque is interpolated linearly between points
    torque_curve: List[Tuple[float, float]] = field(default_factory=lambda: [
        (0.0, 200.0),
        (1000.0, 250.0),
        (2000.0, 300.0),
        (3000.0, 350.0),
        (4000.0, 380.0),
        (5000.0, 400.0),
        (6000.0, 390.0),
        (7000.0, 360.0),
        (8000.0, 300.0),
        (9000.0, 200.0),
    ])
    idle_rpm: float = 800.0
    max_rpm: float = 9000.0
    redline: float = 8500.0

    # Gear ratios (final drive included in each ratio for simplicity)
    gear_ratios: List[float] = field(default_factory=lambda: [
        3.5 * 3.7,   # gear 1
        2.1 * 3.7,   # gear 2
        1.4 * 3.7,   # gear 3
        1.0 * 3.7,   # gear 4
        0.8 * 3.7,   # gear 5
        0.65 * 3.7,  # gear 6
    ])
    reverse_ratio: float = -3.0 * 3.7

    # Drivetrain efficiency
    drivetrain_efficiency: float = 0.9

    # Wheel radius (m)
    wheel_radius: float = 0.34

    # Automatic shift thresholds
    upshift_rpm: float = 7000.0
    downshift_rpm: float = 3000.0

    # Engine braking torque (Nm at idle throttle)
    engine_brake_torque: float = 50.0


class Drivetrain:
    """
    Simulates a simple drivetrain with engine torque curve and gear ratios.

    Computes engine force at the wheels from throttle, RPM, and current gear.
    Supports automatic and manual shifting.
    """

    def __init__(self, params: Optional[DrivetrainParams] = None) -> None:
        self.params = params or DrivetrainParams()
        self.gear: int = 1  # 0 = reverse, 1..N = forward gears
        self.rpm: float = self.params.idle_rpm
        self.auto_shift: bool = True

    @property
    def num_gears(self) -> int:
        return len(self.params.gear_ratios)

    def get_torque(self, throttle: float) -> float:
        """Interpolate engine torque from the torque curve at current RPM."""
        curve = self.params.torque_curve
        rpm = clamp(self.rpm, curve[0][0], curve[-1][0])

        # Linear interpolation between torque curve points
        for i in range(len(curve) - 1):
            r0, t0 = curve[i]
            r1, t1 = curve[i + 1]
            if r0 <= rpm <= r1:
                t = (rpm - r0) / (r1 - r0) if r1 > r0 else 0.0
                base_torque = lerp(t0, t1, t)
                # Apply throttle (with some engine braking at zero throttle)
                if throttle > 0.01:
                    return base_torque * throttle
                else:
                    return -self.params.engine_brake_torque * (rpm / self.params.max_rpm)
        return 0.0

    def get_engine_force(self, throttle: float, wheel_speed: float) -> float:
        """
        Compute tractive force at the wheels.

        Parameters
        ----------
        throttle : float
            Throttle input [0, 1].
        wheel_speed : float
            Wheel angular speed (rad/s) = v_long / wheel_radius.

        Returns
        -------
        float
            Force at the wheels (N).
        """
        p = self.params

        # Determine gear ratio
        if self.gear == 0:
            ratio = p.reverse_ratio
        elif 1 <= self.gear <= self.num_gears:
            ratio = p.gear_ratios[self.gear - 1]
        else:
            ratio = p.gear_ratios[-1]

        # Compute engine RPM from wheel speed
        # rpm = wheel_speed * gear_ratio * 60 / (2*pi)
        if abs(ratio) > 1e-9:
            self.rpm = abs(wheel_speed * ratio) * 60.0 / (2.0 * math.pi)
        else:
            self.rpm = p.idle_rpm

        self.rpm = clamp(self.rpm, p.idle_rpm, p.max_rpm)

        # Engine torque
        engine_torque = self.get_torque(throttle)

        # Wheel force = torque * gear_ratio * efficiency / wheel_radius
        force = engine_torque * ratio * p.drivetrain_efficiency / p.wheel_radius

        return force

    def update_shift(self, throttle: float, brake: float, speed: float) -> None:
        """Automatic gear shifting logic."""
        if not self.auto_shift:
            return

        p = self.params

        # Don't shift while braking hard
        if brake > 0.5:
            return

        # Upshift
        if self.rpm > p.upshift_rpm and self.gear < self.num_gears and self.gear >= 1:
            self.gear += 1
        # Downshift
        elif self.rpm < p.downshift_rpm and self.gear > 1:
            self.gear -= 1

        # Shift to reverse only at very low speed
        if speed < 1.0 and brake > 0.8 and self.gear >= 1:
            self.gear = 0
        elif speed > 2.0 and self.gear == 0:
            self.gear = 1

    def shift_up(self) -> None:
        """Manually shift up one gear."""
        if self.gear < self.num_gears:
            self.gear += 1

    def shift_down(self) -> None:
        """Manually shift down one gear."""
        if self.gear > 0:
            self.gear -= 1

    def get_info(self) -> Dict[str, Any]:
        return {
            "gear": self.gear,
            "rpm": self.rpm,
            "num_gears": self.num_gears,
        }


# ---------------------------------------------------------------------------
# Vehicle parameters
# ---------------------------------------------------------------------------

@dataclass
class VehicleParams:
    """Container for physical vehicle parameters."""

    mass: float = 1200.0              # kg
    inertia: float = 2500.0           # kg*m^2  (yaw moment of inertia)
    wheelbase: float = 2.6            # m  (distance front axle -> rear axle)
    track_width: float = 1.6          # m
    cg_height: float = 0.5            # m  (center of gravity height)
    cg_to_front: float = 1.3          # m  (distance CG -> front axle)
    cg_to_rear: float = 1.3           # m  (distance CG -> rear axle)

    # Engine / drivetrain
    engine_force_max: float = 8000.0  # N  (max tractive force, used when no drivetrain)
    brake_force_max: float = 12000.0  # N  (max braking force)
    brake_bias_front: float = 0.6     # front brake bias (0..1)

    # Steering
    max_steer_angle: float = 0.6      # rad (~34 deg)
    steer_speed: float = 5.0         # steering rate (1/s) for smooth steering

    # Tire
    mu: float = 1.0                   # friction coefficient
    tire_cornering_stiffness_f: float = 50000.0  # N/rad (front)
    tire_cornering_stiffness_r: float = 60000.0  # N/rad (rear)

    # Aero
    drag_coeff: float = 0.3           # Cd
    frontal_area: float = 2.2        # m^2
    air_density: float = 1.225        # kg/m^3
    downforce_coeff: float = 0.0      # Cl (lift coefficient, negative = downforce)
    downforce_area: float = 2.0       # m^2 (effective area for downforce)

    # Rolling resistance
    rolling_resistance_coeff: float = 0.013

    # Gravity
    g: float = 9.81                   # m/s^2

    # Dimensions for rendering (pixels per metre applied externally)
    body_length: float = 4.5         # m
    body_width: float = 1.9          # m
    wheel_length: float = 0.7        # m
    wheel_width: float = 0.25        # m

    # Traction control and ABS
    tc_enabled: bool = True
    abs_enabled: bool = True
    tc_threshold: float = 0.15        # slip ratio threshold for TC intervention
    abs_threshold: float = 0.15       # slip ratio threshold for ABS intervention

    # Use drivetrain simulation (if False, uses simple engine_force_max)
    use_drivetrain: bool = False

    # Integration method: 'euler' or 'rk4'
    integration_method: str = "euler"


# ---------------------------------------------------------------------------
# Car state (for clean state management)
# ---------------------------------------------------------------------------

@dataclass
class CarState:
    """Immutable snapshot of car state for integration."""
    x: float = 0.0
    y: float = 0.0
    angle: float = 0.0
    v_long: float = 0.0
    v_lat: float = 0.0
    yaw_rate: float = 0.0


# ---------------------------------------------------------------------------
# Car
# ---------------------------------------------------------------------------

class Car:
    """
    A car with dynamic bicycle-model dynamics and simplified Pacejka tires.

    Parameters
    ----------
    x, y : float
        Initial world position (metres).
    angle : float
        Initial heading (radians, 0 = +x axis).
    params : VehicleParams, optional
        Physical parameters.  Defaults are used if not supplied.
    drivetrain : Drivetrain, optional
        Drivetrain simulation.  Created automatically if params.use_drivetrain is True.
    """

    def __init__(self,
                 x: float = 0.0,
                 y: float = 0.0,
                 angle: float = 0.0,
                 params: Optional[VehicleParams] = None) -> None:

        self.params: VehicleParams = params or VehicleParams()

        # Kinematic state -------------------------------------------------
        self.x: float = x
        self.y: float = y
        self.angle: float = angle          # heading (rad)
        self.vx: float = 0.0              # world-frame velocity x (m/s)
        self.vy: float = 0.0              # world-frame velocity y (m/s)
        self.yaw_rate: float = 0.0        # angular velocity (rad/s)

        # Body-frame velocities (computed each update)
        self.v_long: float = 0.0          # longitudinal speed in body frame
        self.v_lat: float = 0.0           # lateral speed in body frame
        self.speed: float = 0.0           # scalar speed (m/s)

        # Control inputs (stored for inspection / rendering)
        self.throttle: float = 0.0
        self.brake: float = 0.0
        self.steer: float = 0.0          # -1 .. 1
        self.steer_angle: float = 0.0    # actual steering angle (rad)
        self._steer_smoothed: float = 0.0  # smoothed steering for rate limiting

        # Forces (stored for inspection)
        self.engine_force: float = 0.0
        self.brake_force: float = 0.0
        self.drag_force: float = 0.0
        self.downforce: float = 0.0
        self.rolling_resistance: float = 0.0

        # Tire forces (stored for inspection)
        self.Fyf: float = 0.0    # front lateral force (N)
        self.Fyr: float = 0.0    # rear lateral force (N)
        self.Fxf: float = 0.0    # front longitudinal force (N)
        self.Fxr: float = 0.0    # rear longitudinal force (N)
        self.Fz_f: float = 0.0   # front normal load (N)
        self.Fz_r: float = 0.0   # rear normal load (N)

        # Slip quantities (stored for inspection)
        self.slip_front: float = 0.0
        self.slip_rear: float = 0.0
        self.slip_ratio_front: float = 0.0
        self.slip_ratio_rear: float = 0.0

        # Weight transfer
        self.weight_transfer_long: float = 0.0
        self.weight_transfer_lat: float = 0.0

        # Drivetrain
        self.drivetrain: Optional[Drivetrain] = None
        if self.params.use_drivetrain:
            self.drivetrain = Drivetrain()

        # TC/ABS state
        self.tc_active: bool = False
        self.abs_active: bool = False

        # Off-track penalty (set externally)
        self.off_track: bool = False
        self.off_track_grip_multiplier: float = 1.0

        # Rendering surface cache
        self._surface_cache: dict = {}

    # ------------------------------------------------------------------
    # Core physics: compute derivatives
    # ------------------------------------------------------------------

    def _compute_derivatives(self, state: CarState, throttle: float,
                             brake: float, steer_angle: float) -> Tuple[float, float, float, float, float]:
        """
        Compute state derivatives for the current state.

        Returns (dx, dy, d_angle, dv_long, dv_lat, dyaw_rate).
        """
        p = self.params
        a = p.cg_to_front
        b = p.cg_to_rear

        # World-frame velocity from body-frame
        cos_a = math.cos(state.angle)
        sin_a = math.sin(state.angle)
        vx = state.v_long * cos_a - state.v_lat * sin_a
        vy = state.v_long * sin_a + state.v_lat * cos_a

        # Speed
        speed = math.hypot(vx, vy)

        # Guard against very low speeds
        v_long_safe = state.v_long if abs(state.v_long) > 0.1 else (
            0.1 if state.v_long >= 0 else -0.1
        )

        # --- Slip angles (bicycle model) --------------------------------
        slip_front = math.atan2(
            state.v_lat + state.yaw_rate * a, abs(v_long_safe)
        ) - steer_angle
        slip_rear = math.atan2(
            state.v_lat - state.yaw_rate * b, abs(v_long_safe)
        )

        # --- Longitudinal forces -----------------------------------------
        # Engine force
        if self.drivetrain is not None:
            wheel_speed = state.v_long / p.wheel_radius if hasattr(p, 'wheel_radius') else state.v_long / 0.34
            engine_force = self.drivetrain.get_engine_force(throttle, wheel_speed)
        else:
            engine_force = throttle * p.engine_force_max

        # Aerodynamic drag  F_drag = 0.5 * rho * Cd * A * v^2
        drag_force = 0.5 * p.air_density * p.drag_coeff * p.frontal_area * speed ** 2

        # Downforce  F_down = 0.5 * rho * Cl * A * v^2
        downforce = 0.5 * p.air_density * abs(p.downforce_coeff) * p.downforce_area * speed ** 2

        # Rolling resistance  F_rr = Crr * m * g  (opposes motion)
        if abs(state.v_long) > 1e-3:
            rolling_resistance = p.rolling_resistance_coeff * p.mass * p.g
        else:
            rolling_resistance = 0.0

        # --- Weight transfer --------------------------------------------
        # Longitudinal: weight shifts forward under acceleration, back under braking
        ax_body = engine_force / p.mass  # approximate for weight transfer
        weight_transfer_long = p.mass * ax_body * p.cg_height / p.wheelbase

        # Lateral: weight shifts to outside wheels
        ay_body = abs(state.v_lat + state.yaw_rate * state.v_long)  # approximate lateral accel
        weight_transfer_lat = p.mass * ay_body * p.cg_height / p.track_width

        # Normal loads on front and rear axles
        Fz_f = (p.mass * p.g * b / p.wheelbase) - weight_transfer_long
        Fz_r = (p.mass * p.g * a / p.wheelbase) + weight_transfer_long
        Fz_f = max(Fz_f, 0.0)
        Fz_r = max(Fz_r, 0.0)

        # Add downforce to normal loads (distributed by static weight distribution)
        total_downforce = downforce
        front_fraction = b / p.wheelbase
        rear_fraction = a / p.wheelbase
        Fz_f += total_downforce * front_fraction
        Fz_r += total_downforce * rear_fraction

        # Off-track grip penalty
        grip_mult = self.off_track_grip_multiplier if self.off_track else 1.0
        mu_eff = p.mu * grip_mult

        # --- Tire forces (combined slip) --------------------------------
        # Compute slip ratios (simplified: based on throttle/brake demand vs speed)
        if self.drivetrain is not None:
            # With drivetrain, compute slip ratio from wheel speed vs ground speed
            slip_ratio_front = 0.0  # RWD assumption: front wheels are free-rolling
            # Rear wheel slip from engine force
            wheel_speed_rear = state.v_long / p.wheel_radius if hasattr(p, 'wheel_radius') else state.v_long / 0.34
            # Simplified: slip ratio proportional to engine force
            slip_ratio_rear = clamp(engine_force / (Fz_r * mu_eff + 1e-9) * 0.1, -1.0, 1.0)
        else:
            # Without drivetrain, estimate slip ratio from throttle/brake
            if throttle > 0.01:
                slip_ratio_rear = clamp(throttle * 0.2, -1.0, 1.0)
                slip_ratio_front = 0.0
            elif brake > 0.01:
                slip_ratio_front = clamp(-brake * 0.2, -1.0, 1.0)
                slip_ratio_rear = clamp(-brake * 0.2, -1.0, 1.0)
            else:
                slip_ratio_front = 0.0
                slip_ratio_rear = 0.0

        # Traction control: reduce throttle if rear slip is too high
        tc_active = False
        if p.tc_enabled and throttle > 0.01 and abs(slip_ratio_rear) > p.tc_threshold:
            tc_active = True
            # Reduce engine force
            tc_factor = p.tc_threshold / max(abs(slip_ratio_rear), 1e-9)
            engine_force *= tc_factor
            slip_ratio_rear *= tc_factor

        # ABS: reduce brake if slip is too high
        abs_active = False
        brake_force_total = brake * p.brake_force_max
        if p.abs_enabled and brake > 0.01:
            if abs(slip_ratio_front) > p.abs_threshold or abs(slip_ratio_rear) > p.abs_threshold:
                abs_active = True
                abs_factor = p.abs_threshold / max(
                    max(abs(slip_ratio_front), abs(slip_ratio_rear)), 1e-9
                )
                brake_force_total *= abs_factor

        # Combined slip tire forces
        Fxf, Fyf = combined_slip_pacejka(
            slip_front, slip_ratio_front, mu=mu_eff
        )
        Fxr, Fyr = combined_slip_pacejka(
            slip_rear, slip_ratio_rear, mu=mu_eff
        )

        # Negate lateral forces: pacejka returns positive for positive slip,
        # but the tire force must oppose the slip direction.
        Fxf *= Fz_f
        Fyf *= -Fz_f
        Fxr *= Fz_r
        Fyr *= -Fz_r

        # Speed-dependent scaling: tires cannot generate lateral force at
        # standstill because there is no relative motion (no slip velocity).
        speed_factor = clamp(speed / 1.0, 0.0, 1.0)
        Fyf *= speed_factor
        Fyr *= speed_factor

        # If no significant slip, use linear cornering stiffness (more stable at low speed)
        if abs(slip_front) < 0.02 and abs(slip_ratio_front) < 0.02:
            Fyf = -p.tire_cornering_stiffness_f * slip_front * grip_mult * speed_factor
        if abs(slip_rear) < 0.02 and abs(slip_ratio_rear) < 0.02:
            Fyr = -p.tire_cornering_stiffness_r * slip_rear * grip_mult * speed_factor

        # Limit engine force by available tire grip (traction)
        max_traction = Fz_r * mu_eff
        if abs(engine_force) > max_traction:
            engine_force = math.copysign(max_traction, engine_force)

        # --- Net longitudinal force in body frame -----------------------
        Fx = engine_force
        if state.v_long > 0.1:
            Fx -= brake_force_total
            Fx -= drag_force
            Fx -= rolling_resistance
        elif state.v_long < -0.1:
            Fx += brake_force_total
            Fx += drag_force
            Fx += rolling_resistance
        else:
            # Near-zero speed
            sign = 1.0 if state.v_long >= 0 else -1.0
            Fx -= brake_force_total * sign
            Fx -= drag_force * 0.0  # negligible at very low speed

        # --- Equations of motion (bicycle model) ------------------------
        ax_body = Fx / p.mass
        ay_body = (Fyf * math.cos(steer_angle) + Fyr) / p.mass
        yaw_moment = a * Fyf * math.cos(steer_angle) - b * Fyr
        yaw_accel = yaw_moment / p.inertia

        # Store forces for inspection (only on the final integration step)
        self._store_forces(
            engine_force, brake_force_total, drag_force, downforce,
            rolling_resistance, Fyf, Fyr, Fxf, Fxr, Fz_f, Fz_r,
            slip_front, slip_rear, slip_ratio_front, slip_ratio_rear,
            weight_transfer_long, weight_transfer_lat, tc_active, abs_active
        )

        return vx, vy, state.yaw_rate, ax_body, ay_body, yaw_accel

    def _store_forces(self, engine_force, brake_force, drag_force, downforce,
                      rolling_resistance, Fyf, Fyr, Fxf, Fxr, Fz_f, Fz_r,
                      slip_front, slip_rear, slip_ratio_front, slip_ratio_rear,
                      weight_transfer_long, weight_transfer_lat, tc_active, abs_active) -> None:
        """Store computed forces for inspection."""
        self.engine_force = engine_force
        self.brake_force = brake_force
        self.drag_force = drag_force
        self.downforce = downforce
        self.rolling_resistance = rolling_resistance
        self.Fyf = Fyf
        self.Fyr = Fyr
        self.Fxf = Fxf
        self.Fxr = Fxr
        self.Fz_f = Fz_f
        self.Fz_r = Fz_r
        self.slip_front = slip_front
        self.slip_rear = slip_rear
        self.slip_ratio_front = slip_ratio_front
        self.slip_ratio_rear = slip_ratio_rear
        self.weight_transfer_long = weight_transfer_long
        self.weight_transfer_lat = weight_transfer_lat
        self.tc_active = tc_active
        self.abs_active = abs_active

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, dt: float,
               throttle: float = 0.0,
               brake: float = 0.0,
               steer: float = 0.0) -> None:
        """
        Advance the simulation by *dt* seconds.

        Parameters
        ----------
        dt : float
            Time step in seconds.
        throttle : float
            Throttle input in [0, 1].
        brake : float
            Brake input in [0, 1].
        steer : float
            Steering input in [-1, 1] (left = +1).
        """
        p = self.params

        # --- Clamp inputs ------------------------------------------------
        self.throttle = clamp(throttle, 0.0, 1.0)
        self.brake = clamp(brake, 0.0, 1.0)
        self.steer = clamp(steer, -1.0, 1.0)

        # Smooth steering (rate-limited)
        steer_target = self.steer * p.max_steer_angle
        max_steer_rate = p.steer_speed * p.max_steer_angle
        steer_diff = steer_target - self._steer_smoothed
        max_change = max_steer_rate * dt
        self._steer_smoothed += clamp(steer_diff, -max_change, max_change)
        self.steer_angle = self._steer_smoothed

        # --- Update drivetrain -------------------------------------------
        if self.drivetrain is not None:
            self.drivetrain.update_shift(self.throttle, self.brake, self.speed)

        # --- Integrate ---------------------------------------------------
        if p.integration_method == "rk4":
            self._integrate_rk4(dt, self.throttle, self.brake, self.steer_angle)
        else:
            self._integrate_euler(dt, self.throttle, self.brake, self.steer_angle)

        # Recompute derived quantities
        cos_a = math.cos(self.angle)
        sin_a = math.sin(self.angle)
        self.vx = self.v_long * cos_a - self.v_lat * sin_a
        self.vy = self.v_long * sin_a + self.v_lat * cos_a
        self.speed = math.hypot(self.vx, self.vy)
        self.angle = normalize_angle(self.angle)

    def _integrate_euler(self, dt: float, throttle: float, brake: float,
                         steer_angle: float) -> None:
        """Semi-implicit Euler integration."""
        state = CarState(
            x=self.x, y=self.y, angle=self.angle,
            v_long=self.v_long, v_lat=self.v_lat, yaw_rate=self.yaw_rate
        )

        vx, vy, yaw_rate, ax_body, ay_body, yaw_accel = self._compute_derivatives(
            state, throttle, brake, steer_angle
        )

        # Update body-frame velocities first (semi-implicit)
        self.v_long += ax_body * dt
        self.v_lat += ay_body * dt
        self.yaw_rate += yaw_accel * dt

        # Convert to world frame and update position
        cos_a = math.cos(self.angle)
        sin_a = math.sin(self.angle)
        self.vx = self.v_long * cos_a - self.v_lat * sin_a
        self.vy = self.v_long * sin_a + self.v_lat * cos_a

        self.x += self.vx * dt
        self.y += self.vy * dt
        self.angle += self.yaw_rate * dt

    def _integrate_rk4(self, dt: float, throttle: float, brake: float,
                      steer_angle: float) -> None:
        """4th-order Runge-Kutta integration for higher accuracy."""
        state0 = CarState(
            x=self.x, y=self.y, angle=self.angle,
            v_long=self.v_long, v_lat=self.v_lat, yaw_rate=self.yaw_rate
        )

        # k1
        s1 = CarState(**state0.__dict__)
        _, _, _, ax1, ay1, r1 = self._compute_derivatives(s1, throttle, brake, steer_angle)
        k1 = (self.vx, self.vy, self.yaw_rate, ax1, ay1, r1)

        # k2
        s2 = CarState(
            x=state0.x + k1[0] * dt / 2,
            y=state0.y + k1[1] * dt / 2,
            angle=state0.angle + k1[2] * dt / 2,
            v_long=state0.v_long + k1[3] * dt / 2,
            v_lat=state0.v_lat + k1[4] * dt / 2,
            yaw_rate=state0.yaw_rate + k1[5] * dt / 2,
        )
        # Compute world velocity for s2
        cos2 = math.cos(s2.angle)
        sin2 = math.sin(s2.angle)
        vx2 = s2.v_long * cos2 - s2.v_lat * sin2
        vy2 = s2.v_long * sin2 + s2.v_lat * cos2
        _, _, _, ax2, ay2, r2 = self._compute_derivatives(s2, throttle, brake, steer_angle)
        k2 = (vx2, vy2, s2.yaw_rate, ax2, ay2, r2)

        # k3
        s3 = CarState(
            x=state0.x + k2[0] * dt / 2,
            y=state0.y + k2[1] * dt / 2,
            angle=state0.angle + k2[2] * dt / 2,
            v_long=state0.v_long + k2[3] * dt / 2,
            v_lat=state0.v_lat + k2[4] * dt / 2,
            yaw_rate=state0.yaw_rate + k2[5] * dt / 2,
        )
        cos3 = math.cos(s3.angle)
        sin3 = math.sin(s3.angle)
        vx3 = s3.v_long * cos3 - s3.v_lat * sin3
        vy3 = s3.v_long * sin3 + s3.v_lat * cos3
        _, _, _, ax3, ay3, r3 = self._compute_derivatives(s3, throttle, brake, steer_angle)
        k3 = (vx3, vy3, s3.yaw_rate, ax3, ay3, r3)

        # k4
        s4 = CarState(
            x=state0.x + k3[0] * dt,
            y=state0.y + k3[1] * dt,
            angle=state0.angle + k3[2] * dt,
            v_long=state0.v_long + k3[3] * dt,
            v_lat=state0.v_lat + k3[4] * dt,
            yaw_rate=state0.yaw_rate + k3[5] * dt,
        )
        cos4 = math.cos(s4.angle)
        sin4 = math.sin(s4.angle)
        vx4 = s4.v_long * cos4 - s4.v_lat * sin4
        vy4 = s4.v_long * sin4 + s4.v_lat * cos4
        _, _, _, ax4, ay4, r4 = self._compute_derivatives(s4, throttle, brake, steer_angle)
        k4 = (vx4, vy4, s4.yaw_rate, ax4, ay4, r4)

        # Combine
        self.x += dt / 6.0 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        self.y += dt / 6.0 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        self.angle += dt / 6.0 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
        self.v_long += dt / 6.0 * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3])
        self.v_lat += dt / 6.0 * (k1[4] + 2 * k2[4] + 2 * k3[4] + k4[4])
        self.yaw_rate += dt / 6.0 * (k1[5] + 2 * k2[5] + 2 * k3[5] + k4[5])

    # ------------------------------------------------------------------
    # Getters
    # ------------------------------------------------------------------

    def get_position(self) -> Tuple[float, float]:
        """Return (x, y) world position in metres."""
        return (self.x, self.y)

    def get_speed(self) -> float:
        """Return scalar speed in m/s."""
        return self.speed

    def get_speed_kmh(self) -> float:
        """Return scalar speed in km/h."""
        return self.speed * 3.6

    def get_heading(self) -> float:
        """Return heading angle in radians (normalized to [-pi, pi])."""
        return normalize_angle(self.angle)

    def get_heading_deg(self) -> float:
        """Return heading angle in degrees."""
        return math.degrees(self.get_heading())

    def get_velocity(self) -> Tuple[float, float]:
        """Return world-frame velocity (vx, vy) in m/s."""
        return (self.vx, self.vy)

    def get_body_velocity(self) -> Tuple[float, float]:
        """Return body-frame velocity (longitudinal, lateral) in m/s."""
        return (self.v_long, self.v_lat)

    def get_angular_velocity(self) -> float:
        """Return yaw rate in rad/s."""
        return self.yaw_rate

    def get_lateral_acceleration(self) -> float:
        """Return lateral acceleration in m/s^2 (body frame)."""
        return self.v_lat  # simplified

    def get_longitudinal_acceleration(self) -> float:
        """Return longitudinal acceleration in m/s^2 (body frame)."""
        p = self.params
        Fx = self.engine_force - self.brake_force - self.drag_force - self.rolling_resistance
        return Fx / p.mass

    def get_info(self) -> dict:
        """Return a dictionary of all current state info."""
        info = {
            "x": self.x,
            "y": self.y,
            "angle": self.get_heading(),
            "angle_deg": self.get_heading_deg(),
            "vx": self.vx,
            "vy": self.vy,
            "speed": self.speed,
            "speed_kmh": self.get_speed_kmh(),
            "v_long": self.v_long,
            "v_lat": self.v_lat,
            "yaw_rate": self.yaw_rate,
            "throttle": self.throttle,
            "brake": self.brake,
            "steer": self.steer,
            "steer_angle": self.steer_angle,
            "engine_force": self.engine_force,
            "brake_force": self.brake_force,
            "drag_force": self.drag_force,
            "downforce": self.downforce,
            "rolling_resistance": self.rolling_resistance,
            "slip_front": self.slip_front,
            "slip_rear": self.slip_rear,
            "slip_ratio_front": self.slip_ratio_front,
            "slip_ratio_rear": self.slip_ratio_rear,
            "Fyf": self.Fyf,
            "Fyr": self.Fyr,
            "Fxf": self.Fxf,
            "Fxr": self.Fxr,
            "Fz_f": self.Fz_f,
            "Fz_r": self.Fz_r,
            "weight_transfer_long": self.weight_transfer_long,
            "weight_transfer_lat": self.weight_transfer_lat,
            "tc_active": self.tc_active,
            "abs_active": self.abs_active,
            "off_track": self.off_track,
        }
        if self.drivetrain is not None:
            info.update(self.drivetrain.get_info())
        return info

    # ------------------------------------------------------------------
    # Setters / control
    # ------------------------------------------------------------------

    def set_position(self, x: float, y: float, angle: Optional[float] = None) -> None:
        """Set the car position and optionally heading."""
        self.x = x
        self.y = y
        if angle is not None:
            self.angle = angle
        self.vx = 0.0
        self.vy = 0.0
        self.v_long = 0.0
        self.v_lat = 0.0
        self.yaw_rate = 0.0
        self.speed = 0.0
        self._steer_smoothed = 0.0
        self.steer_angle = 0.0

    def reset(self) -> None:
        """Reset car to origin with zero velocity."""
        self.set_position(0.0, 0.0, 0.0)

    def set_off_track(self, off_track: bool, grip_multiplier: float = 0.5) -> None:
        """Set off-track status and grip penalty."""
        self.off_track = off_track
        self.off_track_grip_multiplier = grip_multiplier

    # ------------------------------------------------------------------
    # Rendering (pygame)
    # ------------------------------------------------------------------

    def render(self, surface, scale: float = 10.0,
               color: Tuple[int, int, int] = (220, 50, 50),
               wheel_color: Tuple[int, int, int] = (30, 30, 30),
               show_direction: bool = True,
               font: Optional[object] = None) -> None:
        """
        Draw the car onto a pygame *surface*.

        Parameters
        ----------
        surface : pygame.Surface
            The target surface to draw on.
        scale : float
            Pixels per metre.
        color : tuple
            Body colour (R, G, B).
        wheel_color : tuple
            Wheel colour (R, G, B).
        show_direction : bool
            If True, draw a direction indicator arrow.
        font : pygame.font.Font, optional
            If provided, draws a small speed readout near the car.
        """
        if pygame is None:
            raise RuntimeError("pygame is required for rendering but not installed.")

        p = self.params
        cx = int(self.x * scale)
        cy = int(self.y * scale)

        # Body dimensions in pixels
        L = p.body_length * scale
        W = p.body_width * scale
        wL = p.wheel_length * scale
        wW = p.wheel_width * scale

        cos_a = math.cos(self.angle)
        sin_a = math.sin(self.angle)

        def world_to_screen(lx: float, ly: float) -> Tuple[int, int]:
            """Transform a local (body-frame) point to screen coordinates."""
            rx = lx * cos_a - ly * sin_a
            ry = lx * sin_a + ly * cos_a
            return (int(cx + rx), int(cy + ry))

        # --- Body polygon (rounded rectangle approximation) --------------
        half_L = L / 2
        half_W = W / 2
        body_pts = [
            world_to_screen(-half_L, -half_W),
            world_to_screen(half_L, -half_W),
            world_to_screen(half_L, half_W),
            world_to_screen(-half_L, half_W),
        ]
        pygame.draw.polygon(surface, color, body_pts)
        pygame.draw.polygon(surface, (0, 0, 0), body_pts, 2)

        # --- Windshield (visual cue for orientation) ---------------------
        ws_front = half_L * 0.3
        ws_back = -half_L * 0.1
        ws_W = half_W * 0.7
        ws_pts = [
            world_to_screen(ws_back, -ws_W),
            world_to_screen(ws_front, -ws_W * 0.8),
            world_to_screen(ws_front, ws_W * 0.8),
            world_to_screen(ws_back, ws_W),
        ]
        pygame.draw.polygon(surface, (100, 180, 220, 180), ws_pts)

        # --- Wheels ------------------------------------------------------
        fa = p.cg_to_front * scale
        ra = -p.cg_to_rear * scale
        tw = p.track_width * scale / 2

        wheel_positions_local = [
            (fa, -tw),   # front-left
            (fa, tw),    # front-right
            (ra, -tw),   # rear-left
            (ra, tw),    # rear-right
        ]

        for i, (lx, ly) in enumerate(wheel_positions_local):
            is_front = i < 2
            if is_front:
                steer_cos = math.cos(self.steer_angle)
                steer_sin = math.sin(self.steer_angle)
            else:
                steer_cos = 1.0
                steer_sin = 0.0

            def wheel_corner(dl: float, dw: float) -> Tuple[int, int]:
                rx = dl * steer_cos - dw * steer_sin
                ry = dl * steer_sin + dw * steer_cos
                wx = (lx + rx) * cos_a - (ly + ry) * sin_a
                wy = (lx + rx) * sin_a + (ly + ry) * cos_a
                return (int(cx + wx), int(cy + wy))

            corners = [
                wheel_corner(-wL / 2, -wW / 2),
                wheel_corner(wL / 2, -wW / 2),
                wheel_corner(wL / 2, wW / 2),
                wheel_corner(-wL / 2, wW / 2),
            ]
            pygame.draw.polygon(surface, wheel_color, corners)

        # --- Direction indicator arrow -----------------------------------
        if show_direction:
            arrow_len = L * 0.6
            tip = world_to_screen(half_L + arrow_len, 0)
            base_l = world_to_screen(half_L + arrow_len * 0.6, -W * 0.15)
            base_r = world_to_screen(half_L + arrow_len * 0.6, W * 0.15)
            base_c = world_to_screen(half_L, 0)
            pygame.draw.polygon(surface, (255, 255, 0),
                                [tip, base_l, base_c, base_r])
            pygame.draw.line(surface, (255, 255, 0),
                             world_to_screen(half_L, 0), tip, 2)

        # --- Speed readout -----------------------------------------------
        if font is not None:
            txt = font.render(f"{self.get_speed_kmh():.1f} km/h",
                              True, (255, 255, 255))
            surface.blit(txt, (cx - txt.get_width() // 2, cy - int(L) - 20))

    def render_top_down(self, surface, camera_offset: Tuple[float, float] = (0, 0),
                        scale: float = 10.0, **kwargs) -> None:
        """
        Render with a camera offset (world_x, world_y) in metres.
        """
        saved_x, saved_y = self.x, self.y
        self.x -= camera_offset[0]
        self.y -= camera_offset[1]
        try:
            self.render(surface, scale=scale, **kwargs)
        finally:
            self.x, self.y = saved_x, saved_y


# ---------------------------------------------------------------------------
# Convenience: run a short demo if executed directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    car = Car(x=0, y=0, angle=0)

    dt = 0.01
    for i in range(500):
        throttle = 0.5
        brake = 0.0
        steer = 0.0
        if 200 < i < 300:
            steer = 0.3  # gentle left turn
        car.update(dt, throttle, brake, steer)

        if i % 50 == 0:
            info = car.get_info()
            print(f"t={i*dt:6.2f}s  "
                  f"pos=({info['x']:7.2f},{info['y']:7.2f})  "
                  f"speed={info['speed_kmh']:6.2f} km/h  "
                  f"heading={info['angle_deg']:7.2f} deg  "
                  f"yaw_rate={info['yaw_rate']:+.4f} rad/s")

    print("\nFinal state:")
    for k, v in car.get_info().items():
        print(f"  {k:25s}: {v}")

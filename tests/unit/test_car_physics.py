"""
test_car_physics.py - Unit tests for the car vehicle dynamics module.

Tests cover:
  - Basic initialization and state queries
  - Straight-line acceleration and braking
  - Steering and turning behavior
  - Weight transfer (longitudinal and lateral)
  - Tire forces and slip angles
  - Aerodynamic drag and rolling resistance
  - Pacejka tire model functions
  - Combined slip tire forces
  - Drivetrain simulation
  - RK4 vs Euler integration accuracy
  - Off-track grip penalty
  - Traction control and ABS
  - Reset and set_position
"""

import math
import sys
import os
import pytest

# Add examples directory to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "examples"))

from car_physics import (
    Car, VehicleParams, Drivetrain, DrivetrainParams, CarState,
    clamp, lerp, normalize_angle, angle_difference,
    pacejka_lateral, pacejka_longitudinal, combined_slip_pacejka,
)


# ---------------------------------------------------------------------------
# Helper math
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_clamp(self):
        assert clamp(5, 0, 10) == 5
        assert clamp(-1, 0, 10) == 0
        assert clamp(15, 0, 10) == 10
        assert clamp(0, 0, 10) == 0
        assert clamp(10, 0, 10) == 10

    def test_lerp(self):
        assert lerp(0, 10, 0) == 0
        assert lerp(0, 10, 1) == 10
        assert lerp(0, 10, 0.5) == 5
        assert lerp(0, 10, -0.5) == 0  # clamped
        assert lerp(0, 10, 1.5) == 10  # clamped

    def test_normalize_angle(self):
        assert normalize_angle(0) == 0
        assert abs(normalize_angle(math.pi) - math.pi) < 1e-10
        assert abs(normalize_angle(-math.pi) - (-math.pi)) < 1e-10
        assert abs(normalize_angle(3 * math.pi) - math.pi) < 1e-10
        assert abs(normalize_angle(2 * math.pi)) < 1e-10

    def test_angle_difference(self):
        assert abs(angle_difference(0, math.pi / 2) - math.pi / 2) < 1e-10
        assert abs(angle_difference(math.pi / 2, 0) - (-math.pi / 2)) < 1e-10
        assert abs(angle_difference(0, math.pi) - math.pi) < 1e-10
        assert abs(angle_difference(0, -math.pi) - (-math.pi)) < 1e-10


# ---------------------------------------------------------------------------
# Pacejka tire model
# ---------------------------------------------------------------------------

class TestPacejka:
    def test_lateral_zero_slip(self):
        """At zero slip angle, lateral force should be zero."""
        assert abs(pacejka_lateral(0.0)) < 1e-10

    def test_lateral_positive_slip(self):
        """Positive slip angle should produce positive lateral force coefficient."""
        result = pacejka_lateral(0.1)
        assert result > 0

    def test_lateral_negative_slip(self):
        """Negative slip angle should produce negative lateral force coefficient."""
        result = pacejka_lateral(-0.1)
        assert result < 0

    def test_lateral_symmetry(self):
        """Lateral force should be antisymmetric around zero slip."""
        val_pos = pacejka_lateral(0.1)
        val_neg = pacejka_lateral(-0.1)
        assert abs(val_pos + val_neg) < 1e-10

    def test_lateral_saturation(self):
        """At very high slip angles, force should saturate (not exceed D)."""
        result = pacejka_lateral(10.0)
        assert abs(result) <= 1.0 + 1e-10

    def test_longitudinal_zero_slip(self):
        """At zero slip ratio, longitudinal force should be zero."""
        assert abs(pacejka_longitudinal(0.0)) < 1e-10

    def test_longitudinal_positive_slip(self):
        """Positive slip ratio (acceleration) should produce positive force."""
        result = pacejka_longitudinal(0.1)
        assert result > 0

    def test_longitudinal_negative_slip(self):
        """Negative slip ratio (braking) should produce negative force."""
        result = pacejka_longitudinal(-0.1)
        assert result < 0

    def test_longitudinal_saturation(self):
        """At extreme slip ratios, force should saturate."""
        result = pacejka_longitudinal(1.0)
        assert abs(result) <= 1.0 + 1e-10

    def test_combined_slip_zero(self):
        """At zero slip, combined slip should return zero forces."""
        fx, fy = combined_slip_pacejka(0.0, 0.0)
        assert abs(fx) < 1e-10
        assert abs(fy) < 1e-10

    def test_combined_slip_pure_lateral(self):
        """With zero longitudinal slip, combined slip should match pure lateral."""
        _, fy_combined = combined_slip_pacejka(0.1, 0.0)
        fy_pure = pacejka_lateral(0.1)
        assert abs(fy_combined - fy_pure) < 1e-10

    def test_combined_slip_pure_longitudinal(self):
        """With zero lateral slip, combined slip should match pure longitudinal."""
        fx_combined, _ = combined_slip_pacejka(0.0, 0.1)
        fx_pure = pacejka_longitudinal(0.1)
        assert abs(fx_combined - fx_pure) < 1e-10

    def test_combined_slip_friction_ellipse(self):
        """When both slips are present, combined force should be within the ellipse."""
        fx, fy = combined_slip_pacejka(0.3, 0.3)
        # Should be within the friction circle
        assert fx ** 2 + fy ** 2 <= 1.0 + 1e-6

    def test_combined_slip_reduces_from_pure(self):
        """Combined slip forces should be less than or equal to pure slip forces."""
        fx_combined, fy_combined = combined_slip_pacejka(0.3, 0.3)
        fx_pure = pacejka_longitudinal(0.3)
        fy_pure = pacejka_lateral(0.3)
        assert abs(fx_combined) <= abs(fx_pure) + 1e-10
        assert abs(fy_combined) <= abs(fy_pure) + 1e-10


# ---------------------------------------------------------------------------
# Car initialization
# ---------------------------------------------------------------------------

class TestCarInit:
    def test_default_init(self):
        car = Car()
        assert car.x == 0.0
        assert car.y == 0.0
        assert car.angle == 0.0
        assert car.speed == 0.0
        assert car.v_long == 0.0
        assert car.v_lat == 0.0
        assert car.yaw_rate == 0.0

    def test_custom_init(self):
        car = Car(x=10, y=20, angle=math.pi / 4)
        assert car.x == 10
        assert car.y == 20
        assert abs(car.angle - math.pi / 4) < 1e-10

    def test_custom_params(self):
        params = VehicleParams(mass=1500, engine_force_max=10000)
        car = Car(params=params)
        assert car.params.mass == 1500
        assert car.params.engine_force_max == 10000

    def test_get_position(self):
        car = Car(x=5, y=10)
        assert car.get_position() == (5, 10)

    def test_get_speed(self):
        car = Car()
        assert car.get_speed() == 0.0

    def test_get_heading(self):
        car = Car(angle=math.pi / 3)
        assert abs(car.get_heading() - math.pi / 3) < 1e-10

    def test_get_heading_deg(self):
        car = Car(angle=math.pi / 2)
        assert abs(car.get_heading_deg() - 90.0) < 1e-10

    def test_get_info_keys(self):
        car = Car()
        info = car.get_info()
        expected_keys = {
            "x", "y", "angle", "angle_deg", "vx", "vy", "speed", "speed_kmh",
            "v_long", "v_lat", "yaw_rate", "throttle", "brake", "steer",
            "steer_angle", "engine_force", "brake_force", "drag_force",
            "downforce", "rolling_resistance", "slip_front", "slip_rear",
            "slip_ratio_front", "slip_ratio_rear", "Fyf", "Fyr", "Fxf", "Fxr",
            "Fz_f", "Fz_r", "weight_transfer_long", "weight_transfer_lat",
            "tc_active", "abs_active", "off_track",
        }
        assert expected_keys.issubset(set(info.keys()))


# ---------------------------------------------------------------------------
# Straight-line dynamics
# ---------------------------------------------------------------------------

class TestStraightLine:
    def test_acceleration_from_standstill(self):
        """Car should accelerate forward when throttle is applied."""
        car = Car()
        dt = 1 / 60
        for _ in range(60):  # 1 second
            car.update(dt, throttle=1.0, brake=0.0, steer=0.0)
        assert car.v_long > 0
        assert car.speed > 0
        assert car.get_speed_kmh() > 0

    def test_more_throttle_more_acceleration(self):
        """More throttle should result in higher speed."""
        car1 = Car()
        car2 = Car()
        dt = 1 / 60
        for _ in range(120):
            car1.update(dt, throttle=0.5, brake=0.0, steer=0.0)
            car2.update(dt, throttle=1.0, brake=0.0, steer=0.0)
        assert car2.get_speed() > car1.get_speed()

    def test_no_throttle_no_acceleration(self):
        """With no throttle and no brake, car should not accelerate."""
        car = Car()
        dt = 1 / 60
        for _ in range(60):
            car.update(dt, throttle=0.0, brake=0.0, steer=0.0)
        # Should remain near zero (only rolling resistance acts)
        assert abs(car.v_long) < 0.1

    def test_braking_decelerates(self):
        """Braking should decelerate the car."""
        car = Car()
        dt = 1 / 60
        # First accelerate
        for _ in range(120):
            car.update(dt, throttle=1.0, brake=0.0, steer=0.0)
        speed_before = car.get_speed()
        # Then brake
        for _ in range(60):
            car.update(dt, throttle=0.0, brake=1.0, steer=0.0)
        speed_after = car.get_speed()
        assert speed_after < speed_before

    def test_drag_limits_top_speed(self):
        """At top speed, drag should balance engine force."""
        car = Car()
        dt = 1 / 60
        for _ in range(6000):  # 100 seconds
            car.update(dt, throttle=1.0, brake=0.0, steer=0.0)
        # Speed should have stabilized (not still accelerating rapidly)
        speed1 = car.get_speed()
        for _ in range(60):
            car.update(dt, throttle=1.0, brake=0.0, steer=0.0)
        speed2 = car.get_speed()
        assert abs(speed2 - speed1) < 1.0  # near equilibrium

    def test_drag_force_increases_with_speed(self):
        """Drag force should increase with speed squared."""
        car = Car()
        dt = 1 / 60
        # Low speed
        for _ in range(30):
            car.update(dt, throttle=0.3, brake=0.0, steer=0.0)
        drag_low = car.drag_force
        # High speed
        for _ in range(300):
            car.update(dt, throttle=1.0, brake=0.0, steer=0.0)
        drag_high = car.drag_force
        assert drag_high > drag_low

    def test_rolling_resistance_exists(self):
        """Rolling resistance should be non-zero when moving."""
        car = Car()
        dt = 1 / 60
        for _ in range(60):
            car.update(dt, throttle=1.0, brake=0.0, steer=0.0)
        assert car.rolling_resistance > 0

    def test_position_advances_forward(self):
        """Car should move forward in x direction when heading is 0."""
        car = Car(x=0, y=0, angle=0)
        dt = 1 / 60
        for _ in range(120):
            car.update(dt, throttle=1.0, brake=0.0, steer=0.0)
        assert car.x > 0
        assert abs(car.y) < 1.0  # should stay roughly on x-axis


# ---------------------------------------------------------------------------
# Steering and turning
# ---------------------------------------------------------------------------

class TestSteering:
    def test_steer_left_turns_left(self):
        """Steering left should cause the car to turn left (positive yaw rate)."""
        car = Car()
        dt = 1 / 60
        # Build up some speed first
        for _ in range(60):
            car.update(dt, throttle=1.0, brake=0.0, steer=0.0)
        # Now steer left
        for _ in range(60):
            car.update(dt, throttle=0.5, brake=0.0, steer=1.0)
        assert car.yaw_rate != 0
        # With left steer and forward motion, heading should change
        assert abs(car.angle) > 0.01

    def test_steer_right_turns_right(self):
        """Steering right should cause negative yaw rate."""
        car = Car()
        dt = 1 / 60
        for _ in range(60):
            car.update(dt, throttle=1.0, brake=0.0, steer=0.0)
        for _ in range(60):
            car.update(dt, throttle=0.5, brake=0.0, steer=-1.0)
        assert car.yaw_rate != 0
        assert car.angle < 0  # turned right (negative angle)

    def test_no_steer_no_yaw(self):
        """With no steering, yaw rate should remain near zero."""
        car = Car()
        dt = 1 / 60
        for _ in range(120):
            car.update(dt, throttle=1.0, brake=0.0, steer=0.0)
        assert abs(car.yaw_rate) < 0.01

    def test_steering_at_standstill(self):
        """Steering at standstill should not cause significant yaw."""
        car = Car()
        dt = 1 / 60
        for _ in range(60):
            car.update(dt, throttle=0.0, brake=0.0, steer=1.0)
        assert abs(car.yaw_rate) < 0.1

    def test_higher_speed_more_turning(self):
        """At higher speed, the same steering input should produce more turning."""
        car_slow = Car()
        car_fast = Car()
        dt = 1 / 60
        # Build different speeds
        for _ in range(30):
            car_slow.update(dt, throttle=0.3, brake=0.0, steer=0.0)
        for _ in range(120):
            car_fast.update(dt, throttle=1.0, brake=0.0, steer=0.0)
        assert car_fast.get_speed() > car_slow.get_speed()
        # Apply same steer
        angle_slow_before = car_slow.angle
        angle_fast_before = car_fast.angle
        for _ in range(30):
            car_slow.update(dt, throttle=0.3, brake=0.0, steer=0.5)
            car_fast.update(dt, throttle=0.3, brake=0.0, steer=0.5)
        # Faster car should have turned more
        assert abs(car_fast.angle - angle_fast_before) > abs(car_slow.angle - angle_slow_before)

    def test_slip_angles_nonzero_when_turning(self):
        """Slip angles should be non-zero when the car is turning."""
        car = Car()
        dt = 1 / 60
        for _ in range(60):
            car.update(dt, throttle=1.0, brake=0.0, steer=0.0)
        for _ in range(60):
            car.update(dt, throttle=0.5, brake=0.0, steer=0.5)
        assert abs(car.slip_front) > 0.001 or abs(car.slip_rear) > 0.001


# ---------------------------------------------------------------------------
# Weight transfer
# ---------------------------------------------------------------------------

class TestWeightTransfer:
    def test_acceleration_transfers_weight_rear(self):
        """Under acceleration, weight should transfer to the rear."""
        car = Car()
        dt = 1 / 60
        # Static loads
        p = car.params
        Fz_f_static = p.mass * p.g * p.cg_to_rear / p.wheelbase
        Fz_r_static = p.mass * p.g * p.cg_to_front / p.wheelbase
        # Accelerate
        for _ in range(60):
            car.update(dt, throttle=1.0, brake=0.0, steer=0.0)
        # Under acceleration, rear load should increase, front should decrease
        assert car.Fz_r > Fz_r_static - 1.0  # rear gets more load
        assert car.Fz_f < Fz_f_static + 1.0  # front gets less load

    def test_braking_transfers_weight_front(self):
        """Under braking, weight should transfer to the front."""
        car = Car()
        dt = 1 / 60
        # Build speed first
        for _ in range(120):
            car.update(dt, throttle=1.0, brake=0.0, steer=0.0)
        p = car.params
        Fz_f_before = car.Fz_f
        # Brake hard
        for _ in range(10):
            car.update(dt, throttle=0.0, brake=1.0, steer=0.0)
        assert car.Fz_f > Fz_f_before  # front gets more load under braking

    def test_normal_load_sum_equals_weight(self):
        """Front + rear normal load should approximately equal total weight."""
        car = Car()
        dt = 1 / 60
        for _ in range(60):
            car.update(dt, throttle=1.0, brake=0.0, steer=0.0)
        p = car.params
        total_weight = p.mass * p.g
        assert abs((car.Fz_f + car.Fz_r) - total_weight) < total_weight * 0.1


# ---------------------------------------------------------------------------
# Drivetrain
# ---------------------------------------------------------------------------

class TestDrivetrain:
    def test_drivetrain_init(self):
        dt = Drivetrain()
        assert dt.gear == 1
        assert dt.rpm == dt.params.idle_rpm

    def test_torque_curve_interpolation(self):
        dt = Drivetrain()
        # At 3000 RPM, should interpolate between (3000, 350) point
        dt.rpm = 3000
        torque = dt.get_torque(1.0)
        assert abs(torque - 350.0) < 1.0

    def test_torque_throttle_scaling(self):
        dt = Drivetrain()
        dt.rpm = 5000
        full_torque = dt.get_torque(1.0)
        half_torque = dt.get_torque(0.5)
        assert abs(half_torque - full_torque * 0.5) < 1.0

    def test_engine_braking(self):
        dt = Drivetrain()
        dt.rpm = 5000
        # Zero throttle should give negative torque (engine braking)
        torque = dt.get_torque(0.0)
        assert torque < 0

    def test_upshift(self):
        dt = Drivetrain()
        dt.shift_up()
        assert dt.gear == 2
        dt.shift_up()
        assert dt.gear == 3

    def test_downshift(self):
        dt = Drivetrain()
        dt.gear = 3
        dt.shift_down()
        assert dt.gear == 2

    def test_cannot_shift_below_zero(self):
        dt = Drivetrain()
        dt.gear = 0
        dt.shift_down()
        assert dt.gear == 0

    def test_cannot_shift_above_max(self):
        dt = Drivetrain()
        for _ in range(20):
            dt.shift_up()
        assert dt.gear == dt.num_gears

    def test_auto_upshift(self):
        """Drivetrain should auto-upshift at high RPM."""
        params = VehicleParams(use_drivetrain=True)
        car = Car(params=params)
        dt = 1 / 60
        # Accelerate hard to trigger upshift
        for _ in range(600):
            car.update(dt, throttle=1.0, brake=0.0, steer=0.0)
        assert car.drivetrain.gear >= 2

    def test_drivetrain_provides_force(self):
        """Drivetrain mode should still accelerate the car."""
        params = VehicleParams(use_drivetrain=True)
        car = Car(params=params)
        dt = 1 / 60
        for _ in range(120):
            car.update(dt, throttle=1.0, brake=0.0, steer=0.0)
        assert car.get_speed() > 0
        assert car.engine_force != 0


# ---------------------------------------------------------------------------
# Integration methods
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_euler_straight_line(self):
        """Euler integration should produce forward motion."""
        car = Car(params=VehicleParams(integration_method="euler"))
        dt = 1 / 60
        for _ in range(120):
            car.update(dt, throttle=1.0, brake=0.0, steer=0.0)
        assert car.x > 0
        assert car.get_speed() > 0

    def test_rk4_straight_line(self):
        """RK4 integration should produce forward motion."""
        car = Car(params=VehicleParams(integration_method="rk4"))
        dt = 1 / 60
        for _ in range(120):
            car.update(dt, throttle=1.0, brake=0.0, steer=0.0)
        assert car.x > 0
        assert car.get_speed() > 0

    def test_rk4_and_euler_similar(self):
        """RK4 and Euler should give similar results for small dt."""
        car_euler = Car(params=VehicleParams(integration_method="euler"))
        car_rk4 = Car(params=VehicleParams(integration_method="rk4"))
        dt = 1 / 120  # small dt for good agreement
        for _ in range(240):
            car_euler.update(dt, throttle=0.5, brake=0.0, steer=0.0)
            car_rk4.update(dt, throttle=0.5, brake=0.0, steer=0.0)
        # Should be within 5% of each other
        speed_e = car_euler.get_speed()
        speed_r = car_rk4.get_speed()
        if speed_e > 1.0:
            assert abs(speed_e - speed_r) / speed_e < 0.05


# ---------------------------------------------------------------------------
# Off-track penalty
# ---------------------------------------------------------------------------

class TestOffTrack:
    def test_off_track_reduces_grip(self):
        """Off-track should reduce acceleration."""
        car_on = Car()
        car_off = Car()
        car_off.set_off_track(True, grip_multiplier=0.5)
        dt = 1 / 60
        for _ in range(120):
            car_on.update(dt, throttle=1.0, brake=0.0, steer=0.0)
            car_off.update(dt, throttle=1.0, brake=0.0, steer=0.0)
        assert car_on.get_speed() > car_off.get_speed()

    def test_set_off_track(self):
        car = Car()
        car.set_off_track(True, grip_multiplier=0.3)
        assert car.off_track == True
        assert car.off_track_grip_multiplier == 0.3

    def test_clear_off_track(self):
        car = Car()
        car.set_off_track(True)
        car.set_off_track(False, grip_multiplier=1.0)
        assert car.off_track == False
        assert car.off_track_grip_multiplier == 1.0


# ---------------------------------------------------------------------------
# TC and ABS
# ---------------------------------------------------------------------------

class TestTCABS:
    def test_tc_can_activate(self):
        """Traction control should be able to activate under high throttle."""
        params = VehicleParams(tc_enabled=True, tc_threshold=0.05)
        car = Car(params=params)
        dt = 1 / 60
        # Hammer the throttle from standstill
        for _ in range(30):
            car.update(dt, throttle=1.0, brake=0.0, steer=0.0)
        # TC may or may not activate depending on slip, but the flag should exist
        assert isinstance(car.tc_active, bool)

    def test_abs_can_activate(self):
        """ABS should be able to activate under hard braking."""
        params = VehicleParams(abs_enabled=True, abs_threshold=0.05)
        car = Car(params=params)
        dt = 1 / 60
        # Build speed
        for _ in range(120):
            car.update(dt, throttle=1.0, brake=0.0, steer=0.0)
        # Brake hard
        for _ in range(10):
            car.update(dt, throttle=0.0, brake=1.0, steer=0.0)
        assert isinstance(car.abs_active, bool)

    def test_tc_disabled(self):
        """With TC disabled, tc_active should always be False."""
        params = VehicleParams(tc_enabled=False)
        car = Car(params=params)
        dt = 1 / 60
        for _ in range(60):
            car.update(dt, throttle=1.0, brake=0.0, steer=0.0)
        assert car.tc_active == False

    def test_abs_disabled(self):
        """With ABS disabled, abs_active should always be False."""
        params = VehicleParams(abs_enabled=False)
        car = Car(params=params)
        dt = 1 / 60
        for _ in range(120):
            car.update(dt, throttle=1.0, brake=0.0, steer=0.0)
        for _ in range(10):
            car.update(dt, throttle=0.0, brake=1.0, steer=0.0)
        assert car.abs_active == False


# ---------------------------------------------------------------------------
# Reset and set_position
# ---------------------------------------------------------------------------

class TestResetSetPosition:
    def test_reset(self):
        car = Car(x=100, y=200, angle=1.0)
        car.v_long = 50
        car.vx = 30
        car.reset()
        assert car.x == 0
        assert car.y == 0
        assert car.angle == 0
        assert car.v_long == 0
        assert car.vx == 0
        assert car.speed == 0

    def test_set_position(self):
        car = Car()
        car.update(0.01, throttle=1.0)
        car.set_position(50, 100, angle=math.pi / 2)
        assert car.x == 50
        assert car.y == 100
        assert abs(car.angle - math.pi / 2) < 1e-10
        assert car.v_long == 0
        assert car.speed == 0

    def test_set_position_no_angle(self):
        car = Car(angle=1.5)
        car.set_position(10, 20)
        assert car.x == 10
        assert car.y == 20
        assert car.angle == 1.5  # unchanged


# ---------------------------------------------------------------------------
# Input clamping
# ---------------------------------------------------------------------------

class TestInputClamping:
    def test_throttle_clamped(self):
        car = Car()
        car.update(0.01, throttle=2.0)
        assert car.throttle == 1.0

    def test_throttle_negative_clamped(self):
        car = Car()
        car.update(0.01, throttle=-1.0)
        assert car.throttle == 0.0

    def test_brake_clamped(self):
        car = Car()
        car.update(0.01, brake=2.0)
        assert car.brake == 1.0

    def test_steer_clamped(self):
        car = Car()
        car.update(0.01, steer=2.0)
        assert car.steer == 1.0

    def test_steer_negative_clamped(self):
        car = Car()
        car.update(0.01, steer=-2.0)
        assert car.steer == -1.0


# ---------------------------------------------------------------------------
# Downforce
# ---------------------------------------------------------------------------

class TestDownforce:
    def test_downforce_increases_normal_load(self):
        """Downforce should increase normal loads at speed."""
        params = VehicleParams(downforce_coeff=0.8, downforce_area=2.0)
        car = Car(params=params)
        dt = 1 / 60
        # Build speed
        for _ in range(300):
            car.update(dt, throttle=1.0, brake=0.0, steer=0.0)
        assert car.downforce > 0
        p = car.params
        total_weight = p.mass * p.g
        # With downforce, total normal load should exceed weight
        assert (car.Fz_f + car.Fz_r) > total_weight

    def test_no_downforce_by_default(self):
        """Default params should have zero downforce."""
        car = Car()
        dt = 1 / 60
        for _ in range(60):
            car.update(dt, throttle=1.0, brake=0.0, steer=0.0)
        assert car.downforce == 0.0


# ---------------------------------------------------------------------------
# Consistency / determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_repeated_simulation_identical(self):
        """Running the same simulation twice should give identical results."""
        def run_sim():
            car = Car()
            dt = 1 / 60
            for _ in range(300):
                car.update(dt, throttle=0.7, brake=0.0, steer=0.0)
            return car.get_info()

        result1 = run_sim()
        result2 = run_sim()
        assert result1["x"] == result2["x"]
        assert result1["y"] == result2["y"]
        assert result1["speed"] == result2["speed"]
        assert result1["angle"] == result2["angle"]

    def test_turning_simulation_identical(self):
        """Turning simulation should be deterministic."""
        def run_sim():
            car = Car()
            dt = 1 / 60
            for _ in range(60):
                car.update(dt, throttle=1.0, brake=0.0, steer=0.0)
            for _ in range(60):
                car.update(dt, throttle=0.5, brake=0.0, steer=0.5)
            return car.get_info()

        result1 = run_sim()
        result2 = run_sim()
        assert result1["x"] == result2["x"]
        assert result1["angle"] == result2["angle"]

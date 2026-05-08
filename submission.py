"""
Competition instructions:
Please do not change anything else but fill out the to-do sections.
"""

from collections import deque
from functools import reduce
import json
import os
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import math
import numpy as np
import roar_py_interface
from LateralController import LatController
from ThrottleController import ThrottleController
from WaypointLine import WaypointLine
from SectionStats import SectionStats
from section_bundle import SectionBundle, load_section_bundle, load_section_indices
import atexit

# from scipy.interpolate import interp1d

useDebug = False
useDebugPrinting = False
debugData = {}
dbg_carLocations = []
dbg_wpsToFollow = []
dbg_str = []
dbg_str2 = []
dbg_steer = []

START_SECTIONS = {0, 1, 2, 3, 4, 5}
EARLY_SMOOTH_SECTIONS = {6, 7, 8}
FAST_ENTRY_SECTIONS = {9, 10, 11}
HAIRPIN_SECTIONS = {12, 13, 14, 15}
MID_BRAKE_SECTIONS = {16, 17, 18}
MID_ACCEL_SECTIONS = {19, 20, 21, 22}
HIGH_SPEED_TURN_SECTIONS = {23, 24}
RECOVERY_SECTIONS = {25}
FAST_APPROACH_SECTIONS = {26, 27, 28, 29}
FINAL_SECTIONS = {30, 31, 32, 33, 34, 35}

# If the runner does not pass bundle_path (e.g. competition_runner.py), load tuning from
# the data folder next to competition_code when this file exists.
_DEFAULT_TUNING_NAME = "section_tuning_36.json"


def dist_to_waypoint(location, waypoint: roar_py_interface.RoarPyWaypoint):
    return np.linalg.norm(location[:2] - waypoint.location[:2])


def filter_waypoints(
    location: np.ndarray,
    current_idx: int,
    waypoints: List[roar_py_interface.RoarPyWaypoint],
) -> int:
    for i in range(current_idx, len(waypoints) + current_idx):
        if dist_to_waypoint(location, waypoints[i % len(waypoints)]) < 3:
            return i % len(waypoints)
    min_dist = 1000
    min_ind = current_idx
    for i in range(0, 20):
        ind = (current_idx + i) % len(waypoints)
        d = dist_to_waypoint(location, waypoints[ind])
        if d < min_dist:
            min_dist = d
            min_ind = ind
    return min_ind


def findClosestIndex(location, waypoints: List[roar_py_interface.RoarPyWaypoint]):
    lowestDist = 100
    closestInd = 0
    for i in range(0, len(waypoints)):
        dist = dist_to_waypoint(location, waypoints[i % len(waypoints)])
        if dist < lowestDist:
            lowestDist = dist
            closestInd = i
    return closestInd % len(waypoints)


@atexit.register
def saveDebugData():
    print("Saving...")
    fname = "\\debugData\\line.txt"
    with open(
        f"{os.path.dirname(__file__)}{fname}", "w+"
    ) as outfile:
        outfile.write("\n--- Debug steer\n")
        for line in dbg_steer:
            outfile.write(f"{line}\n")
        outfile.write("\n--- Locatons\n")
        for line in dbg_carLocations:
            outfile.write(f"{line}\n")
        outfile.write("\n--- wpsToFollow\n")
        for line in dbg_wpsToFollow:
            outfile.write(f"{line}\n")
        outfile.write("\n--- Debug str\n")
        for line in dbg_str2:
            outfile.write(f"{line}\n")
        outfile.write("\n--- More Debug str\n")
        for line in dbg_str:
            outfile.write(f"{line}\n")
    print(f"Saved. {fname}")

    if useDebug:
        print("Saving debug data")
        jsonData = json.dumps(debugData, indent=4)
        with open(
            f"{os.path.dirname(__file__)}\\debugData\\debugData.json", "w+"
        ) as outfile:
            outfile.write(jsonData)
        print("Debug Data Saved")


class RoarCompetitionSolution:
    def __init__(
        self,
        maneuverable_waypoints: List[roar_py_interface.RoarPyWaypoint],
        vehicle: roar_py_interface.RoarPyActor,
        camera_sensor: roar_py_interface.RoarPyCameraSensor = None,
        location_sensor: roar_py_interface.RoarPyLocationInWorldSensor = None,
        velocity_sensor: roar_py_interface.RoarPyVelocimeterSensor = None,
        rpy_sensor: roar_py_interface.RoarPyRollPitchYawSensor = None,
        occupancy_map_sensor: roar_py_interface.RoarPyOccupancyMapSensor = None,
        collision_sensor: roar_py_interface.RoarPyCollisionSensor = None,
        bundle_path: Optional[str] = None,
    ) -> None:
        self.maneuverable_waypoints = maneuverable_waypoints
        self.vehicle = vehicle
        self.camera_sensor = camera_sensor
        self.location_sensor = location_sensor
        self.velocity_sensor = velocity_sensor
        self.rpy_sensor = rpy_sensor
        self.occupancy_map_sensor = occupancy_map_sensor
        self.collision_sensor = collision_sensor
        self.lat_controller = LatController()
        self.throttle_controller = ThrottleController()
        self.section_stats = None
        self.section_indeces = []
        self.num_ticks = 0
        self.current_section = 0
        self.lapNum = 1
        self.previous_waypoint_to_follow = None
        self.max_radius = 10000
        self.previous_location = None
        self.total_dist = 0
        self.waypoint_line = WaypointLine()
        self.previous_brake = False
        self.s3_mult = 1
        self.bundle_path = bundle_path
        self.params: Optional[SectionBundle] = None

    async def initialize(self) -> None:
        # NOTE waypoints are changed through this line
        self.maneuverable_waypoints = (
            roar_py_interface.RoarPyWaypoint.load_waypoint_list(
                np.load(f"{os.path.dirname(__file__)}\\waypoints\\waypointsPrimary.npz")
            )[35:]
        )
        self.section_stats = SectionStats(
            self.maneuverable_waypoints, self.location_sensor, self.velocity_sensor)

        sectionLocations = []
        # for i in sectionLocations:
        #     self.section_indeces.append(
        #         findClosestIndex(i, self.maneuverable_waypoints)
        #     )
        self.section_indeces = [
            2611, 25, 79, 139, 192, 288, 322, 437, 482,
            557, 614, 659, 739, 795, 817, 966, 1158, 1267,
            1296, 1317, 1370, 1381, 1463, 1516, 1829, 1881,
            1944, 1972, 2052, 2112, 2359, 2466, 2522, 2547,
            2558, 2579,
        ]
        if self.bundle_path is None:
            candidate = (
                Path(__file__).resolve().parent.parent
                / "data"
                / _DEFAULT_TUNING_NAME
            )
            if candidate.is_file():
                self.bundle_path = str(candidate.resolve())

        if self.bundle_path is not None:
            self.params = load_section_bundle(self.bundle_path)
            if self.params.section_indices_path is not None and self.params.section_indices_path.exists():
                loaded = load_section_indices(self.params.section_indices_path)
                if loaded:
                    self.section_indeces = [
                        int(np.clip(idx, 0, len(self.maneuverable_waypoints) - 1))
                        for idx in loaded
                    ]

        print(f"True total length: {len(self.maneuverable_waypoints) * 3}")
        print(f"1 lap length: {len(self.maneuverable_waypoints)}")
        print(f"Section indexes: {self.section_indeces}")
        print(f"[section36] sections={len(self.section_indeces)}")
        print("\nLap 1\n")

        # Receive location, rotation and velocity data
        vehicle_location = self.location_sensor.get_last_gym_observation()
        vehicle_rotation = self.rpy_sensor.get_last_gym_observation()
        vehicle_velocity = self.velocity_sensor.get_last_gym_observation()

        self.current_waypoint_idx = 0
        self.current_waypoint_idx = filter_waypoints(
            vehicle_location, self.current_waypoint_idx, self.maneuverable_waypoints
        )
        self.current_section = self._update_section(self.section_indeces, self.current_section)
        self.previous_location = vehicle_location


    async def step(self) -> None:
        self.num_ticks += 1
        self.section_stats.step()

        vehicle_location = self.location_sensor.get_last_gym_observation()
        vehicle_rotation = self.rpy_sensor.get_last_gym_observation()
        vehicle_velocity = self.velocity_sensor.get_last_gym_observation()
        current_speed_kmh = float(np.linalg.norm(vehicle_velocity) * 3.6)

        self.current_waypoint_idx = filter_waypoints(
            vehicle_location, self.current_waypoint_idx, self.maneuverable_waypoints
        )

        old_section = self.current_section
        self.current_section = self._update_section(self.section_indeces, self.current_section)
        if self.current_section != old_section:
            print(
                f"[section36] section {self.current_section} "
                f"at wp {self.current_waypoint_idx} tick {self.num_ticks}"
            )
            if self.current_section == 0 and self.lapNum != 3:
                self.lapNum += 1

        next_waypoint_index = self.get_lookahead_index(current_speed_kmh)
        waypoint_to_follow = self.next_waypoint_smooth(current_speed_kmh, vehicle_location)
        waypoint_to_follow_location = waypoint_to_follow.location
        snap_to_line_location = self.waypoint_line.get_next_waypoint_location(waypoint_to_follow.location)
        if self.current_section not in START_SECTIONS and self.current_section not in FINAL_SECTIONS:
            waypoint_to_follow_location = snap_to_line_location

        steer_control, _steer_debug = self.lat_controller.run(
            vehicle_location,
            vehicle_rotation,
            waypoint_to_follow_location,
        )

        waypoints_for_throttle = (self.maneuverable_waypoints * 2)[
            next_waypoint_index : next_waypoint_index + 300
        ]
        num_points_before_lookahead = 9
        wp_len = len(self.maneuverable_waypoints)
        wp_ind_for_throttle = ((next_waypoint_index + wp_len) - num_points_before_lookahead) % wp_len
        additional_waypoints = (self.maneuverable_waypoints * 2)[
            wp_ind_for_throttle : wp_ind_for_throttle + 300
        ]

        prev_mu_override = getattr(self.throttle_controller, "mu_override", None)
        prev_brake_distance_scale = getattr(self.throttle_controller, "brake_distance_scale", 1.0)
        self.throttle_controller.mu_override = self._section_friction_mu()
        self.throttle_controller.brake_distance_scale = self._section_brake_distance_scale()
        try:
            throttle, brake, gear, speed_data, _throttle_debug = self.throttle_controller.run(
                waypoints_for_throttle,
                vehicle_location,
                current_speed_kmh,
                self.current_section,
                additional_waypoints,
            )
        finally:
            self.throttle_controller.mu_override = prev_mu_override
            self.throttle_controller.brake_distance_scale = prev_brake_distance_scale

        steer_multiplier = round((current_speed_kmh + 0.001) / 120, 3)
        if self.current_waypoint_idx in [800, 801]:
            self.s3_mult = 0.85
            if current_speed_kmh >= 162:
                self.s3_mult = 0.95
                if not self.previous_brake:
                    throttle = 0
                    brake = 1
                    self.previous_brake = True
            if current_speed_kmh < 160:
                self.s3_mult = 0.75
        if self.current_waypoint_idx in [802, 803, 804]:
            self.previous_brake = False

        if self.current_section in FAST_ENTRY_SECTIONS:
            steer_multiplier *= 1.2
        if self.current_section in HAIRPIN_SECTIONS:
            if self.current_waypoint_idx < 813:
                steer_multiplier *= self.s3_mult
            elif self.current_waypoint_idx < 845:
                steer_multiplier *= 1.45
            else:
                steer_multiplier *= 1
                self.s3_mult = 1
        if self.current_section in MID_BRAKE_SECTIONS:
            steer_multiplier = min(1.45, steer_multiplier * 1.65)
        if self.current_section in MID_ACCEL_SECTIONS:
            steer_multiplier *= 1.1
        if self.current_section in HIGH_SPEED_TURN_SECTIONS:
            steer_multiplier = np.clip(steer_multiplier * 3.2, 3.1, 7)
        if self.current_section in RECOVERY_SECTIONS:
            steer_multiplier *= 1.75
        if self.current_section in FINAL_SECTIONS:
            if self.current_waypoint_idx > 2580:
                steer_multiplier = max(steer_multiplier, 1.7)
            else:
                steer_multiplier = max(steer_multiplier, 1.5)

        steer_multiplier *= self._section_steer_gain_scale()
        steer_value = float(np.clip(steer_control * steer_multiplier, -1.0, 1.0))
        if 820 < self.current_waypoint_idx < 837:
            steer_value = float(np.clip(steer_control * steer_multiplier, -0.007, 1.0))

        section_target_kmh = self._section_target_speed(self.current_section)
        if section_target_kmh is not None:
            scaled_target_kmh = section_target_kmh * self._section_speed_scale(self.current_section)
            target_kmh = self._apply_approach_cap_kmh(scaled_target_kmh)
            speed_margin = current_speed_kmh - target_kmh
            if speed_margin > 12.0:
                throttle = min(throttle, 0.0)
                brake = max(brake, 0.55)
            elif speed_margin > 6.0:
                throttle = min(throttle, 0.15)
                brake = max(brake, 0.25)
            elif speed_margin > 2.0:
                throttle = min(throttle, 0.35)
                brake = max(brake, 0.05)
            elif speed_margin < -4.0:
                throttle = max(throttle, 1.0)
                brake = min(brake, 0.0)
            elif speed_margin < -1.0:
                throttle = max(throttle, 0.85)
                brake = min(brake, 0.0)
        else:
            target_kmh = -1.0

        control = {
            "throttle": float(np.clip(throttle, 0.0, 1.0)),
            "steer": steer_value,
            "brake": float(np.clip(brake, 0.0, 1.0)),
            "hand_brake": 0.0,
            "reverse": 0,
            "target_gear": gear,
        }
        if self.current_waypoint_idx in [2381, 2382] and current_speed_kmh > 140:
            control["steer"] = 0.25

        if self.num_ticks % 10 == 0:
            print(
                f"[section36] t={self.num_ticks} sec={self.current_section} speed={current_speed_kmh:.2f} "
                f"tgt={target_kmh:.2f} thr={control['throttle']:.2f} "
                f"brk={control['brake']:.2f} steer={control['steer']:.2f} "
                f"wp={self.current_waypoint_idx}"
            )

        await self.vehicle.apply_action(control)
        return control

    def _update_section(self, section_indices, current_section):
        if not section_indices:
            return current_section
        # Pick the most recent section marker behind the car on the circular track.
        n_wp = len(self.maneuverable_waypoints)
        cur_idx = int(self.current_waypoint_idx)
        best_section = int(current_section)
        best_distance = n_wp + 1
        for i, section_ind in enumerate(section_indices):
            distance_from_marker = (cur_idx - int(section_ind)) % n_wp
            if distance_from_marker < best_distance:
                best_distance = distance_from_marker
                best_section = i
        return best_section

    def _section_speed_scale(self, section_idx: int) -> float:
        if self.params is None:
            return 1.0
        return float(np.clip(self.params.speed_scales.get(int(section_idx), 1.0), 0.4, 3.0))

    def _section_target_speed(self, section_idx: int) -> Optional[float]:
        if self.params is None:
            return None
        value = self.params.target_speeds.get(int(section_idx))
        return None if value is None else float(value)

    def _section_approach_speed(self, section_idx: int) -> float:
        if self.params is None:
            return 0.0
        return float(self.params.approach_speeds.get(int(section_idx), 0.0))

    def _section_approach_lead(self, section_idx: int) -> float:
        if self.params is None:
            return 0.0
        return float(self.params.approach_leads.get(int(section_idx), 0.0))

    def _section_lookahead_scale(self) -> float:
        if self.params is None:
            return 1.0
        return float(
            np.clip(
                self.params.lookahead_scales.get(int(self.current_section), 1.0),
                0.6,
                1.8,
            )
        )

    def _section_steer_gain_scale(self) -> float:
        if self.params is None:
            return 1.0
        return float(
            np.clip(
                self.params.steer_gain_scales.get(int(self.current_section), 1.0),
                0.7,
                1.5,
            )
        )

    def _section_friction_mu(self) -> Optional[float]:
        if self.params is None:
            return None
        value = self.params.friction_mus.get(int(self.current_section))
        return None if value is None else float(np.clip(value, 0.5, 5.0))

    def _section_brake_distance_scale(self) -> float:
        if self.params is None:
            return 1.0
        return float(
            np.clip(
                self.params.brake_distance_scales.get(int(self.current_section), 1.0),
                0.7,
                1.5,
            )
        )

    def _apply_approach_cap_kmh(self, base_target_kmh: float) -> float:
        if self.params is None or not self.section_indeces:
            return float(base_target_kmh)
        next_section = (int(self.current_section) + 1) % len(self.section_indeces)
        approach_speed = self._section_approach_speed(next_section)
        approach_lead = self._section_approach_lead(next_section)
        if approach_speed <= 0.0 or approach_lead <= 0.0:
            return float(base_target_kmh)
        next_idx = int(self.section_indeces[next_section])
        wp_remaining = float((next_idx - self.current_waypoint_idx) % len(self.maneuverable_waypoints))
        if wp_remaining > approach_lead:
            return float(base_target_kmh)
        blend = 1.0 - float(np.clip(wp_remaining / max(approach_lead, 1.0), 0.0, 1.0))
        blended = float(base_target_kmh + blend * (approach_speed - base_target_kmh))
        if approach_speed < base_target_kmh:
            return float(min(base_target_kmh, blended))
        return float(blended)

    def get_lookahead_value(self, speed):
        """
        Returns the number of waypoints to look ahead based on the speed the car is currently going
        """
        speed_to_lookahead_dict = {
            90: 9,
            110: 11,
            130: 14,
            160: 18,
            180: 22,
            200: 26,
            250: 30,
            300: 35,
        }

        # Interpolation method
        # NOTE does not work as well as the dictionary lookahead method, likely to cause crashes.

        # speedBoundList = [0, 90, 110, 130, 160, 180, 200, 250, 300]
        # lookaheadList = [5, 11, 13, 15, 18, 22, 25, 28, 32]

        # interpolationFunction = interp1d(speedBoundList, lookaheadList)
        # return int(interpolationFunction(speed))

        for speed_upper_bound, num_points in speed_to_lookahead_dict.items():
            if speed < speed_upper_bound:
                return max(1, int(round(num_points * self._section_lookahead_scale())))
        return max(1, int(round(8 * self._section_lookahead_scale())))

    def get_lookahead_index(self, speed):
        """
        Adds the lookahead waypoint to the current waypoint and normalizes it so that the value does not go out of bounds
        """
        num_waypoints = self.get_lookahead_value(speed)
        # print("speed " + str(speed)
        #       + " cur_ind " + str(self.current_waypoint_idx)
        #       + " num_points " + str(num_waypoints)
        #       + " index " + str((self.current_waypoint_idx + num_waypoints) % len(self.maneuverable_waypoints)) )
        return (self.current_waypoint_idx + num_waypoints) % len(
            self.maneuverable_waypoints
        )

    # def get_lateral_pid_config(self):
    #     """
    #     Returns the PID values for the lateral (steering) PID
    #     """
    #     with open(
    #         f"{os.path.dirname(__file__)}\\configs\\LatPIDConfig.json", "r"
    #     ) as file:
    #         config = json.load(file)
    #     return config

    # The idea and code for averaging points is from smooth_waypoint_following_local_planner.py (Summer 2023)
    def next_waypoint_smooth(self, current_speed: float, vehicle_location: float):
        """
        If the speed is higher than 70, 'smooth out' the path that the car will take
        """
        if self.current_section in [12, 13, 14, 15]:
            kdd = 0.25
            distance = kdd * current_speed
            distance = np.clip(distance, 44, 70)
            location, _ = self.waypoint_line.get_lookahead_location(vehicle_location, distance)
            point = roar_py_interface.RoarPyWaypoint(location, roll_pitch_yaw=np.ndarray([0, 0, 0]), lane_width=0.0)
            return point
        if self.current_section in [19, 20, 21, 22, 25]:
            kdd = 0.25
            distance = kdd * current_speed
            distance = np.clip(distance, 30, 70)
            location, _ = self.waypoint_line.get_lookahead_location(vehicle_location, distance)
            point = roar_py_interface.RoarPyWaypoint(location, roll_pitch_yaw=np.ndarray([0, 0, 0]), lane_width=0.0)
            return point
        if self.current_section in [23, 24]:
            kdd = 0.28
            distance = kdd * current_speed
            distance = np.clip(distance, 30, 70)
            location, _ = self.waypoint_line.get_lookahead_location(vehicle_location, distance)
            point = roar_py_interface.RoarPyWaypoint(location, roll_pitch_yaw=np.ndarray([0, 0, 0]), lane_width=0.0)
            return point
        if current_speed > 70 and current_speed < 300:
            target_waypoint = self.average_point(current_speed)
        else:
            new_waypoint_index = self.get_lookahead_index(current_speed)
            target_waypoint = self.maneuverable_waypoints[new_waypoint_index]

        return target_waypoint

    def new_RoarPyWaypoint(self, location):
        return roar_py_interface.RoarPyWaypoint(location, roll_pitch_yaw=np.ndarray([0, 0, 0]), lane_width=12.0)


    def average_point(self, current_speed):
        """
        Returns a new averaged waypoint based on the location of a number of other waypoints
        """
        next_waypoint_index = self.get_lookahead_index(current_speed)
        lookahead_value = self.get_lookahead_value(current_speed)
        num_points = lookahead_value * 2

        # Section specific tuning
        if self.current_section in [0, 1, 2, 3, 4, 5]:
            num_points = round(lookahead_value * 1.5)
        if self.current_section in [12, 13, 14, 15]:
            next_waypoint_index = self.current_waypoint_idx + 22
            num_points = 35
        if self.current_section in [16, 17, 18]:
            num_points = lookahead_value + 5
            next_waypoint_index = self.current_waypoint_idx + 24
        if self.current_section in [19, 20, 21, 22]:
            # num_points = round(lookahead_value * 1.1)
            num_points = lookahead_value
        if self.current_section in [23, 24]:
            num_points = lookahead_value
            # num_points = 5
            next_waypoint_index = self.current_waypoint_idx + 28
        if self.current_section in [25]:
            # Jolt between sections 6 and 7 likely due to the differences in lookahead values and steering multipliers. 
            num_points = round(lookahead_value * 1.25)
        if self.current_section in [30, 31, 32, 33, 34, 35]:
            # (self.current_waypoint_idx + 8) % len(self.maneuverable_waypoints)
            num_points = 0

        start_index_for_avg = (next_waypoint_index - (num_points // 2)) % len(
            self.maneuverable_waypoints
        )

        next_waypoint = self.maneuverable_waypoints[next_waypoint_index]
        next_location = next_waypoint.location

        sample_points = [
            (start_index_for_avg + i) % len(self.maneuverable_waypoints)
            for i in range(0, num_points)
        ]
        if num_points > 3:
            location_sum = reduce(
                lambda x, y: x + y,
                (self.maneuverable_waypoints[i].location for i in sample_points),
            )
            num_points = len(sample_points)
            new_location = location_sum / num_points
            shift_distance = np.linalg.norm(next_location - new_location)
            max_shift_distance = 2.0
            if self.current_section in [6, 7, 8]:
                max_shift_distance = 0.2
            if shift_distance > max_shift_distance:
                uv = (new_location - next_location) / shift_distance
                new_location = next_location + uv * max_shift_distance

            target_waypoint = roar_py_interface.RoarPyWaypoint(
                location=new_location,
                roll_pitch_yaw=np.ndarray([0, 0, 0]),
                lane_width=0.0,
            )
            # if next_waypoint_index > 1900 and next_waypoint_index < 2300:
            #   print("AVG: next_ind:" + str(next_waypoint_index) + " next_loc: " + str(next_location)
            #       + " new_loc: " + str(new_location) + " shift:" + str(shift_distance)
            #       + " num_points: " + str(num_points) + " start_ind:" + str(start_index_for_avg)
            #       + " curr_speed: " + str(current_speed))

        else:
            target_waypoint = self.maneuverable_waypoints[next_waypoint_index]

        return target_waypoint

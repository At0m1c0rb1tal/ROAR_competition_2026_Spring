import argparse
import asyncio
from pathlib import Path
from typing import Any, Dict, Optional, Type

import carla
import numpy as np
import roar_py_carla
import roar_py_interface

from infrastructure import ManualControlViewer, RoarCompetitionAgentWrapper
from submission import RoarCompetitionSolution


class OneLapRule:
    def __init__(
        self,
        waypoints,
        vehicle: roar_py_carla.RoarPyCarlaActor,
        world: roar_py_carla.RoarPyCarlaWorld,
    ) -> None:
        self.waypoints = waypoints
        self.vehicle = vehicle
        self.world = world
        self._last_vehicle_location = vehicle.get_3d_location()

    def initialize_race(self):
        self._last_vehicle_location = self.vehicle.get_3d_location()
        vehicle_location = self._last_vehicle_location
        closest_waypoint_dist = np.inf
        closest_waypoint_idx = 0
        for i, waypoint in enumerate(self.waypoints):
            waypoint_dist = np.linalg.norm(vehicle_location - waypoint.location)
            if waypoint_dist < closest_waypoint_dist:
                closest_waypoint_dist = waypoint_dist
                closest_waypoint_idx = i
        self.waypoints = self.waypoints[closest_waypoint_idx + 1 :] + self.waypoints[: closest_waypoint_idx + 1]
        self.furthest_waypoints_index = 0
        print(f"one lap length: {len(self.waypoints)}")

    def lap_finished(self, check_step=5):
        return self.furthest_waypoints_index + check_step >= len(self.waypoints)

    async def tick(self, check_step=15):
        current_location = self.vehicle.get_3d_location()
        delta_vector = current_location - self._last_vehicle_location
        delta_vector_norm = np.linalg.norm(delta_vector)
        delta_vector_unit = (
            delta_vector / delta_vector_norm if delta_vector_norm >= 1e-5 else np.zeros(3)
        )

        previous_furthest_index = self.furthest_waypoints_index
        min_dis = np.inf
        min_index = 0
        endind_index = (
            previous_furthest_index + check_step
            if previous_furthest_index + check_step <= len(self.waypoints)
            else len(self.waypoints)
        )
        for i, waypoint in enumerate(self.waypoints[previous_furthest_index:endind_index]):
            waypoint_delta = waypoint.location - current_location
            projection = np.dot(waypoint_delta, delta_vector_unit)
            projection = np.clip(projection, 0, delta_vector_norm)
            closest_point_on_segment = current_location + projection * delta_vector_unit
            distance = np.linalg.norm(waypoint.location - closest_point_on_segment)
            if distance < min_dis:
                min_dis = distance
                min_index = i

        self.furthest_waypoints_index += min_index
        self._last_vehicle_location = current_location


async def evaluate_solution_one_lap(
    world: roar_py_carla.RoarPyCarlaWorld,
    solution_constructor: Type[RoarCompetitionSolution],
    max_seconds=400.0,
    enable_visualization=False,
    num_laps=1,
) -> Optional[Dict[str, Any]]:
    viewer = ManualControlViewer() if enable_visualization else None
    num_laps = max(1, int(num_laps))

    waypoints = world.maneuverable_waypoints
    try:
        carla_world = getattr(world, "_world", None) or getattr(world, "world", None)
        if carla_world is not None:
            for actor in carla_world.get_actors().filter("vehicle.*"):
                actor.destroy()
            for _ in range(5):
                await world.step()
    except Exception:
        pass

    vehicle = world.spawn_vehicle(
        "vehicle.tesla.model3",
        waypoints[0].location + np.array([0, 0, 1]),
        waypoints[0].roll_pitch_yaw,
        True,
    )
    assert vehicle is not None
    camera = vehicle.attach_camera_sensor(
        roar_py_interface.RoarPyCameraSensorDataRGB,
        np.array([-2.0 * vehicle.bounding_box.extent[0], 0.0, 3.0 * vehicle.bounding_box.extent[2]]),
        np.array([0, 10 / 180.0 * np.pi, 0]),
        image_width=1024,
        image_height=768,
    )
    location_sensor = vehicle.attach_location_in_world_sensor()
    velocity_sensor = vehicle.attach_velocimeter_sensor()
    rpy_sensor = vehicle.attach_roll_pitch_yaw_sensor()
    occupancy_map_sensor = vehicle.attach_occupancy_map_sensor(50, 50, 2.0, 2.0)
    collision_sensor = vehicle.attach_collision_sensor(np.zeros(3), np.zeros(3))
    assert camera is not None
    assert location_sensor is not None
    assert velocity_sensor is not None
    assert rpy_sensor is not None
    assert occupancy_map_sensor is not None
    assert collision_sensor is not None

    solution = solution_constructor(
        waypoints,
        RoarCompetitionAgentWrapper(vehicle),
        camera,
        location_sensor,
        velocity_sensor,
        rpy_sensor,
        occupancy_map_sensor,
        collision_sensor,
    )
    rule = OneLapRule(waypoints, vehicle, world)

    for _ in range(20):
        await world.step()

    rule.initialize_race()
    start_time = world.last_tick_elapsed_seconds
    await vehicle.receive_observation()
    await solution.initialize()
    completed_laps = 0

    while True:
        current_time = world.last_tick_elapsed_seconds
        if current_time - start_time > max_seconds:
            vehicle.close()
            if viewer is not None:
                viewer.close()
            return None

        await vehicle.receive_observation()
        await rule.tick()

        collision_impulse_norm = np.linalg.norm(collision_sensor.get_last_observation().impulse_normal)
        if collision_impulse_norm > 100.0:
            print(f"major collision of intensity {collision_impulse_norm}")
            vehicle.close()
            if viewer is not None:
                viewer.close()
            return None

        if rule.lap_finished():
            completed_laps += 1
            lap_time = current_time - start_time
            print(f"Completed lap {completed_laps}/{num_laps} at {lap_time} seconds")
            if completed_laps >= num_laps:
                break
            print(f"\nLap {completed_laps + 1}\n")
            rule = OneLapRule(waypoints, vehicle, world)
            rule.initialize_race()
            await world.step()
            continue

        if viewer is not None and viewer.render(camera.get_last_observation()) is None:
            vehicle.close()
            viewer.close()
            return None

        await solution.step()
        await world.step()

    end_time = world.last_tick_elapsed_seconds
    vehicle.close()
    if viewer is not None:
        viewer.close()
    return {"elapsed_time": end_time - start_time, "laps_completed": completed_laps}


async def _main(args):
    carla_client = carla.Client(args.host, args.port)
    carla_client.set_timeout(5.0)
    roar_py_instance = roar_py_carla.RoarPyCarlaInstance(carla_client)
    world = roar_py_instance.world
    world.set_control_steps(0.05, 0.005)
    world.set_asynchronous(False)

    bundle_path = str(Path(args.bundle).resolve())
    result = await evaluate_solution_one_lap(
        world,
        lambda *ctor_args, **ctor_kwargs: RoarCompetitionSolution(
            *ctor_args,
            bundle_path=bundle_path,
            **ctor_kwargs,
        ),
        max_seconds=args.max_seconds,
        enable_visualization=not args.no_rendering,
        num_laps=args.num_laps,
    )
    if result is None:
        print("36-section direct failed to finish")
    else:
        print(
            f"36-section direct finished in {result['elapsed_time']} seconds "
            f"for {result.get('laps_completed', args.num_laps)} laps"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bundle",
        default=str(
            Path(__file__).resolve().parents[1] / "data" / "section_tuning_36.json"
        ),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--max-seconds", type=float, default=400.0)
    parser.add_argument("--num-laps", type=int, default=1)
    parser.add_argument("--no-rendering", action="store_true")
    args = parser.parse_args()
    asyncio.run(_main(args))


if __name__ == "__main__":
    main()

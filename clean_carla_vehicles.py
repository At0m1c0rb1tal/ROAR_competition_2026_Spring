import argparse

import carla


def main() -> None:
    parser = argparse.ArgumentParser(description="Destroy leftover CARLA vehicle actors.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=5.0)
    args = parser.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    world = client.get_world()
    vehicles = list(world.get_actors().filter("vehicle.*"))
    print(f"Found {len(vehicles)} vehicle actors")

    for actor in vehicles:
        print(f"Destroying {actor.id}: {actor.type_id}")
        actor.destroy()

    print("Vehicle cleanup complete")


if __name__ == "__main__":
    main()

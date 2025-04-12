import asyncio
import json
import os

import numpy as np
import omni.ext
import omni.kit.app
import omni.usd
import toml
import websockets
from omni.isaac.core import World
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.prims import XFormPrim
from omni.isaac.universal_robots.kinematics_solver import KinematicsSolver
from scipy.spatial.transform import Rotation as R


class MotionKinematicsExtension(omni.ext.IExt):
    def __init__(self):
        super().__init__()

        self.config = {
            "effector": None,
            "reference": None,
            "articulation": None,
            "server": "ws://localhost:8081",
            "subject": "subject.pose",
            "token": None,
        }

        try:
            ext_manager = omni.kit.app.get_app().get_extension_manager()
            ext_id = ext_manager.get_extension_id_by_module(__name__)
            ext_path = ext_manager.get_extension_path(ext_id)
            config = os.path.join(ext_path, "config", "config.toml")
            print("[MotionKinematicsExtension] Extension config: {}".format(config))
            config = toml.load(config)
            print("[MotionKinematicsExtension] Extension config: {}".format(config))
            self.config["effector"] = (
                config.get("effector", self.config["effector"])
                or self.config["effector"]
            )
            self.config["reference"] = (
                config.get("reference", self.config["reference"])
                or self.config["reference"]
            )
            self.config["articulation"] = (
                config.get("articulation", self.config["articulation"])
                or self.config["articulation"]
            )
            self.config["server"] = (
                config.get("server", self.config["server"]) or self.config["server"]
            )
            self.config["subject"] = (
                config.get("subject", self.config["subject"]) or self.config["subject"]
            )
            self.config["token"] = (
                config.get("token", self.config["token"]) or self.config["token"]
            )
        except Exception as e:
            print("[MotionKinematicsExtension] Extension config: {}".format(e))

        assert self.config[
            "effector"
        ], "[MotionKinematicsExtension] Extension config: no effector"
        assert self.config[
            "reference"
        ], "[MotionKinematicsExtension] Extension config: no reference"
        assert self.config[
            "articulation"
        ], "[MotionKinematicsExtension] Extension config: no articulation"
        print("[MotionKinematicsExtension] Extension config: {}".format(self.config))

    def on_startup(self, ext_id):
        async def g(self):
            async def value_stream(self):
                try:
                    while getattr(self, "running", True):
                        try:
                            async with websockets.connect(
                                self.config["server"],
                                extra_headers={
                                    "Authorization": "Bearer {}".format(
                                        self.config["token"]
                                    )
                                },
                            ) as ws:
                                await ws.send(
                                    "SUB {} 1\r\n".format(self.config["subject"])
                                )
                                while getattr(self, "running", True):
                                    try:
                                        response = await asyncio.wait_for(
                                            ws.recv(), timeout=1.0
                                        )
                                        print(
                                            "[MotionKinematicsExtension] Extension server value: {}".format(
                                                response
                                            )
                                        )
                                        for head, body in zip(
                                            *[iter(response.split(b"\r\n"))] * 2
                                        ):
                                            if head.startswith(b"MSG "):
                                                op, sub, sid, count = head.split(
                                                    b" ", 3
                                                )
                                                assert op == b"MSG"
                                                assert sub
                                                assert sid
                                                assert int(count) == len(body)

                                                pose = json.loads(body)
                                                position = np.array(
                                                    (
                                                        pose["position"]["x"],
                                                        pose["position"]["y"],
                                                        pose["position"]["z"],
                                                    )
                                                )
                                                orientation = np.array(
                                                    (
                                                        pose["orientation"]["x"],
                                                        pose["orientation"]["y"],
                                                        pose["orientation"]["z"],
                                                        pose["orientation"]["w"],
                                                    )
                                                )
                                                print(
                                                    "[MotionKinematicsExtension] Extension server value: {}/{}".format(
                                                        position, orientation
                                                    )
                                                )
                                                yield position, orientation

                                    except asyncio.TimeoutError:
                                        pass
                        except asyncio.CancelledError:
                            raise
                        except Exception as e:
                            print(
                                "[MotionKinematicsExtension] Extension server value: {}".format(
                                    e
                                )
                            )
                            await asyncio.sleep(1)
                except asyncio.CancelledError:
                    print("[MotionKinematicsExtension] Extension server value cancel")
                finally:
                    print("[MotionKinematicsExtension] Extension server value exit")

            async def delta_stream(stream):
                prev = None
                async for item in stream:
                    if prev is not None:
                        yield (prev, item)
                    prev = item

            async for value in value_stream(self):
                print(
                    "[MotionKinematicsExtension] Extension server value: {}".format(
                        value
                    )
                )
                self.kinematics_delta = value

        self.running = True
        loop = asyncio.get_event_loop()
        self.server_task = loop.create_task(g(self))
        print("[MotionKinematicsExtension] Extension startup")

        self.subscription = (
            omni.kit.app.get_app()
            .get_update_event_stream()
            .create_subscription_to_pop(self.world_callback, name="world_callback")
        )

    def world_callback(self, e):
        try:
            print("[MotionKinematicsExtension] Extension world wait")
            stage = omni.usd.get_context().get_stage()
            if not stage:
                return
            print("[MotionKinematicsExtension] Extension stage ready {}".format(stage))

            world = World.instance()
            if not world:
                return
            if not world.stage:
                return
            print(
                "[MotionKinematicsExtension] Extension world: {} {}".format(
                    world, world.stage
                )
            )

            if not stage.GetPrimAtPath("/physicsScene").IsValid():
                return
            print("[MotionKinematicsExtension] Extension physics scene ready")

            prim = stage.GetPrimAtPath(self.config["articulation"])
            if not prim.IsValid():
                return
            print("[MotionKinematicsExtension] Extension prim ready {}".format(prim))

            self.articulation = Articulation(self.config["articulation"])
            self.articulation.initialize()
            self.controller = self.articulation.get_articulation_controller()
            self.solver = KinematicsSolver(self.articulation, self.config["effector"])
            print(
                "[MotionKinematicsExtension] Extension articulation ready {} ({}) {} {}".format(
                    self.articulation,
                    self.articulation.dof_names,
                    self.controller,
                    self.solver,
                )
            )

            if world.physics_callback_exists("on_physics_step"):
                world.remove_physics_callback("on_physics_step")
            world.add_physics_callback("on_physics_step", self.on_physics_step)

            if self.subscription:
                self.subscription.unsubscribe()
            self.subscription = None
        except Exception as e:
            print("[MotionKinematicsExtension] Extension world exception {}".format(e))

    def on_physics_step(self, step_size):
        delta, self.kinematics_delta = getattr(self, "kinematics_delta", None), None
        if delta is not None:
            print(
                "[MotionKinematicsExtension] Extension physics: {} {}".format(
                    delta, step_size
                )
            )
            delta_p, delta_o = delta

            position, orientation = XFormPrim(self.config["reference"]).get_world_pose()
            print(
                "[MotionKinematicsExtension] Extension reference position/orientation: {} {}".format(
                    position, orientation
                )
            )
            base_p = position
            base_o = np.array(
                (orientation[1], orientation[2], orientation[3], orientation[0])
            )

            # Convert base orientation to Rotation object
            base_r = R.from_quat(base_o)
            # Convert delta orientation to Rotation object
            delta_r = R.from_quat(delta_o)
            # Compose rotations (apply delta after base)
            pose_r = base_r * delta_r

            # Rotate delta_p into base frame, then add base_p
            pose_p = base_p + base_r.apply(delta_p)
            # Get composed orientation as [x, y, z, w]
            pose_o = pose_r.as_quat()

            print(
                "[MotionKinematicsExtension] Extension pose: {} {}".format(
                    pose_p, pose_o
                )
            )

            target_position = pose_p
            target_orientation = np.array((pose_o[3], pose_o[0], pose_o[1], pose_o[2]))

            kinematics, success = self.solver.compute_inverse_kinematics(
                target_position=target_position, target_orientation=target_orientation
            )
            print(
                "[MotionKinematicsExtension] Extension kinematics: {} {}".format(
                    kinematics, success
                )
            )

            if success:
                self.controller.apply_action(kinematics)

        return

    def on_shutdown(self):
        if self.subscription:
            self.subscription.unsubscribe()
        self.subscription = None

        async def g(self):
            if getattr(self, "server_task") and self.server_task:
                self.server_task.cancel()
                try:
                    await self.server_task
                except asyncio.CancelledError:
                    print("[MotionKinematicsExtension] Extension server cancel")
                except Exception as e:
                    print(
                        "[MotionKinematicsExtension] Extension server exception {}".format(
                            e
                        )
                    )

        self.running = False
        loop = asyncio.get_event_loop()
        loop.run_until_complete(g(self))
        print("[MotionKinematicsExtension] Extension shutdown")

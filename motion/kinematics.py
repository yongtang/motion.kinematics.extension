import omni.ext
import omni.usd
import omni.kit.app
from omni.isaac.sensor import Camera
from omni.isaac.core.articulations import Articulation
from omni.isaac.dynamic_control import _dynamic_control
from scipy.spatial.transform import Rotation as R
import asyncio, websockets, toml, json, os, socket, io
import numpy as np
import PIL.Image


class MotionKinematicsExtension(omni.ext.IExt):
    def __init__(self):
        super().__init__()

        self.config = {
            "subject": "subject.pose",
            "effector": None,
            "articulation": None,
            "server": "ws://localhost:8081",
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
            self.config["articulation"] = (
                config.get("articulation", self.config["articulation"])
                or self.config["articulation"]
            )
            self.config["server"] = (
                config.get("server", self.config["server"]) or self.config["server"]
            )
        except Exception as e:
            print("[MotionKinematicsExtension] Extension config: {}".format(e))
        print("[MotionKinematicsExtension] Extension config: {}".format(self.config))

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def on_startup(self, ext_id):
        async def f(self):
            context = omni.usd.get_context()
            while context.get_stage() is None:
                print("[MotionKinematicsExtension] Extension world wait")
                await asyncio.sleep(0.5)
            print("[MotionKinematicsExtension] Extension world ready")

            stage = context.get_stage()
            print("[MotionKinematicsExtension] Extension stage {}".format(stage))

            if self.config["articulation"]:
                self.articulation = Articulation(self.config["articulation"])
                print(
                    "[MotionKinematicsExtension] Extension articulation {} ({})".format(
                        self.articulation, self.articulation.dof_names
                    )
                )

        async def g(self):
            try:
                while self.running:
                    try:
                        async with websockets.connect(self.config["server"]) as ws:
                            await ws.send("SUB {} 1\r\n".format(self.config["subject"]))
                            while self.running:
                                try:
                                    response = await asyncio.wait_for(
                                        ws.recv(), timeout=1.0
                                    )
                                    print(
                                        "[MotionKinematicsExtension] Extension server: {}".format(
                                            response
                                        )
                                    )
                                    head, body = response.split(b"\r\n", 1)
                                    if head.startswith(b"MSG "):
                                        assert body.endswith(b"\r\n")
                                        body = body[:-2]

                                        op, sub, sid, count = head.split(b" ", 3)
                                        assert op == b"MSG"
                                        assert sub
                                        assert sid
                                        assert int(count) == len(body)

                                        self.kinematics_pose = json.loads(body)
                                        print(
                                            "[MotionKinematicsExtension] Extension server pose: {}".format(
                                                self.kinematics_pose
                                            )
                                        )

                                except asyncio.TimeoutError:
                                    pass
                    except asyncio.CancelledError:
                        raise
                    except Exception as e:
                        print(
                            "[MotionKinematicsExtension] Extension server: {}".format(e)
                        )
                        await asyncio.sleep(1)
            except asyncio.CancelledError:
                print("[MotionKinematicsExtension] Extension server cancel")
            finally:
                print("[MotionKinematicsExtension] Extension server exit")

        self.kinematics_pose = None
        self.kinematics_step = None

        self.running = True
        loop = asyncio.get_event_loop()
        loop.run_until_complete(f(self))
        self.server_task = loop.create_task(g(self))
        print("[MotionKinematicsExtension] Extension startup")

        # self.subscription = (
        #    omni.kit.app.get_app()
        #    .get_update_event_stream()
        #    .create_subscription_to_pop(self.step, name="StepFunction")
        # )

    def delta(self):
        print(
            "[MotionKinematicsExtension] Extension delta: {} {}",
            self.kinematics_pose,
            self.kinematics_step,
        )

        delta_p, delta_r = np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0, 1.0])
        if self.kinematics_pose is not None:
            value = self.kinematics_pose
            if (
                self.kinematics_step is not None
                and self.kinematics_step["channel"] == value["channel"]
            ):
                delta_p = np.array(
                    (
                        value["position"]["x"],
                        value["position"]["y"],
                        value["position"]["z"],
                    )
                ) - np.array(
                    (
                        self.kinematics_step["position"]["x"],
                        self.kinematics_step["position"]["y"],
                        self.kinematics_step["position"]["z"],
                    )
                )

                delta_r = (
                    R.from_quat(
                        np.array(
                            (
                                value["orientation"]["x"],
                                value["orientation"]["y"],
                                value["orientation"]["z"],
                                value["orientation"]["w"],
                            )
                        )
                    )
                    * R.from_quat(
                        np.array(
                            (
                                self.kinematics_step["orientation"]["x"],
                                self.kinematics_step["orientation"]["y"],
                                self.kinematics_step["orientation"]["z"],
                                self.kinematics_step["orientation"]["w"],
                            )
                        )
                    ).inv()
                ).as_quat()
                print(
                    "[MotionKinematicsExtension] Extension delta: {} {}".format(
                        delta_p, delta_r
                    )
                )
            self.kinematics_step = value
        return delta_p, delta_r

    def on_shutdown(self):
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

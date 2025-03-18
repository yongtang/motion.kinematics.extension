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

                                        data = json.loads(body)
                                        print(
                                            "[MotionKinematicsExtension] Extension server: {}".format(
                                                data
                                            )
                                        )
                                        self.position = np.array(
                                            (
                                                data["position"]["x"],
                                                data["position"]["y"],
                                                data["position"]["z"],
                                                data["orientation"]["x"],
                                                data["orientation"]["y"],
                                                data["orientation"]["z"],
                                                data["orientation"]["w"],
                                            )
                                        )
                                        print(
                                            "[MotionKinematicsExtension] Extension position: {}".format(
                                                self.position
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

        self.position = None
        self.step_position = None

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

    def step(self, delta: float):
        print("[MotionKinematicsExtension] Extension step {}".format(delta))

        if self.position is not None:
            value = self.position
            if self.step_position is not None:
                delta_p = value[:3] - value[:3]
                delta_r = (
                    R.from_quat(value[3:]) * R.from_quat(value[3:]).inv()
                ).as_quat()
                print(
                    "[MotionKinematicsExtension] Extension step {} {} {}".format(
                        delta_p, delta_r, delta
                    )
                )
            self.step_position = value

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

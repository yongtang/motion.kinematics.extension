import asyncio
import omni.ext
from . import websocket


class MotionExtension(omni.ext.IExt):
    def __init__(self):
        super().__init__()

        server = "ws://localhost:8081"
        try:
            ext_manager = omni.ext.get_extension_manager()
            ext_id = ext_manager.get_extension_id_by_module(__name__)
            ext_path = ext_manager.get_extension_path(ext_id)
            config = toml.load(os.path.join(ext_path, "config.toml"))
            print("[MotionExtension] Extension config: {}".format(config))
            server = config.get("nats_ws_url", "ws://localhost:8080") or server
        except Exception as e:
            print("[MotionExtension] Extension config: {}".format(e))

        print("[MotionExtension] Extension server: {}".format(server))

        self.ws_server = websocket.WebSocketServer()

    def on_startup(self, ext_id):
        loop = asyncio.get_event_loop()
        loop.create_task(self.ws_server.start())
        print("[MotionExtension] Extension startup")

    def on_shutdown(self):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.ws_server.stop())
        print("[MotionExtension] Extension shutdown")

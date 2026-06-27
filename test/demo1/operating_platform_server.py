import logging
from pathlib import Path
import threading
import time

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from gevent import monkey

# 视频源配置
VIDEO_SOURCES = {
    "camera_1": 6,
    "camera_2": 0,
}


app = Flask(__name__)
CORS(app)
monkey.patch_all()


# 存储视频流实例
video_streams = {}
stream_status = {}

# 全局锁用于线程安全
frame_lock = threading.Lock()


class VideoStream:
    def __init__(self, stream_id, source):
        self.stream_id = stream_id
        self.source = source
        self.running = False
        self.frame_buffers = [None, None]  # 双缓冲
        self.buffer_index = 0
        self.lock = threading.Lock()

    def start(self):
        """启动视频流（仅标记为运行）"""
        if self.running:
            print("已经启动视频流")
            return True
        self.running = True
        return True

    def stop(self):
        """停止视频流"""
        self.running = False

    def update_frame(self, frame_data):
        """接收外部帧数据并更新当前帧"""
        if not self.running:
            return
        # 解码图像
        img = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return

        # 压缩图像（可选）
        img = cv2.resize(img, (640, 480))
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
        _, jpeg = cv2.imencode(".jpg", img, encode_param)
        compressed_frame = jpeg.tobytes()

        with self.lock:
            self.buffer_index = 1 - self.buffer_index
            self.frame_buffers[self.buffer_index] = compressed_frame

    def get_frame(self):
        if not self.running:
            return self.generate_blank_frame()
        with self.lock:
            return self.frame_buffers[self.buffer_index]

    @staticmethod
    def generate_blank_frame():
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        _, jpeg = cv2.imencode(".jpg", blank)
        return jpeg.tobytes()


@app.route("/api/info")
def system_info():
    """获取系统信息"""
    active_count = sum(1 for s in stream_status.values() if s["active"])

    return jsonify(
        {
            "status": "running",
            "pipeline_mode": _pipeline_mode,
            "streams_active": active_count,
            "total_streams": len(stream_status),
            "timestamp": time.time(),
            "streams": stream_status,
        }
    )


@app.route("/api/stream_info", methods=["POST"])
def update_streams():
    """更新可用视频流列表"""

    # 解析并验证传入数据
    data = request.get_json()
    if not data or "streams" not in data:
        return jsonify({"error": "Invalid data format"}), 400

    incoming_streams = data["streams"]

    # 获取当前视频源配置（如摄像头索引）
    current_sources = VIDEO_SOURCES

    # 构建新流与旧流的对比逻辑
    new_stream_ids = {stream["id"] for stream in incoming_streams}
    current_stream_ids = set(video_streams.keys())

    to_remove = current_stream_ids - new_stream_ids
    to_add = new_stream_ids - current_stream_ids

    # 停止不再需要的流
    for stream_id in to_remove:
        video_streams[stream_id].stop()
        del video_streams[stream_id]
        del stream_status[stream_id]

    # 添加新流（根据VIDEO_SOURCES配置）
    success = []
    failed = []

    for stream in incoming_streams:
        stream_id = stream["id"]
        name = stream["name"]

        if stream_id in video_streams:
            continue  # 已存在，跳过

        # 获取对应的摄像头源
        source_key = f"camera_{stream_id}"
        if source_key not in current_sources:
            failed.append(stream_id)
            continue

        source = current_sources[source_key]
        vs = VideoStream(stream_id, source)

        if vs.start():
            video_streams[stream_id] = vs
            stream_status[stream_id] = {
                "id": stream_id,
                "name": name,
                "active": True,
                "source": source,
            }
            success.append(stream_id)
        else:
            failed.append(stream_id)

    return jsonify(
        {
            "success": success,
            "failed": failed,
            "total": len(success),
            "timestamp": time.time(),
        }
    )


@app.route("/api/stream_info", methods=["GET"])
def get_streams():
    """获取可用视频流列表"""
    streams = [{"id": sid, "name": info["name"]} for sid, info in stream_status.items()]
    return jsonify({"streams": streams})


@app.route("/api/start_stream", methods=["POST"])
def start_stream():
    """启动指定视频流"""
    data = request.get_json()
    stream_id = data.get("image")

    if stream_id not in video_streams:
        return jsonify({"error": "无效的视频流ID"}), 400

    success = video_streams[stream_id].start()
    if success:
        stream_status[stream_id]["active"] = True
        return jsonify({"status": "started"})
    else:
        return jsonify({"error": "启动视频流失败"}), 500


@app.route("/api/get_stream/<stream_id>")
def stream_video(stream_id):
    if stream_id not in video_streams:
        return jsonify({"error": "视频流不存在"}), 404

    def generate():
        try:
            while True:
                frame = video_streams[stream_id].get_frame()
                yield (
                    b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                )
                time.sleep(0.03)  # 控制帧率
        except GeneratorExit:
            print(f"[INFO] 客户端断开视频流: {stream_id}")

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/update_stream/<stream_id>", methods=["POST"])
def update_frame(stream_id):
    if stream_id not in video_streams:
        logging.error(f"Invalid stream ID: {stream_id}")
        return jsonify({"error": "无效的视频流ID"}), 400

    frame_data = request.get_data()
    if not frame_data:
        logging.error("No frame data received")
        return jsonify({"error": "未接收到帧数据"}), 400

    try:
        # 可选：验证是否为 JPEG 数据
        img = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Invalid JPEG data")
    except Exception as e:
        logging.error(f"Invalid frame data: {e}")
        return jsonify({"error": "无效的帧数据"}), 400

    video_streams[stream_id].update_frame(frame_data)
    return jsonify({"status": "帧已更新"})


@app.route("/api/stop_stream/<stream_id>", methods=["POST"])
def stop_stream(stream_id):
    """停止指定视频流"""
    if stream_id not in video_streams:
        return jsonify({"error": "视频流不存在"}), 404

    video_streams[stream_id].stop()
    stream_status[stream_id]["active"] = False

    return jsonify({"status": "stopped"})



# ==== DeepCybo 数据管线 API（双模式：internal / cloud）====
import os as _os
import sys as _sys
import yaml as _yaml

_server_dir = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                             "..", "..", "..", "RoboDriver-Server", "x86")
if _server_dir not in _sys.path:
    _sys.path.insert(0, _server_dir)
import internal_sync as _internal_sync

_backend = None
_pipeline_mode = "internal"

def _load_pipeline_mode():
    """从 setup.yaml 读取 pipeline_mode"""
    global _pipeline_mode
    setup_paths = [
        _os.path.join(_server_dir, "setup.yaml"),
        _os.path.join(_os.path.dirname(_server_dir), "arm", "setup.yaml"),
    ]
    for sp in setup_paths:
        if _os.path.exists(sp):
            with open(sp) as f:
                cfg = _yaml.safe_load(f) or {}
            _pipeline_mode = cfg.get("pipeline_mode", "internal")
            return
    # fallback: 环境变量
    _pipeline_mode = _os.environ.get("PIPELINE_MODE", "internal")

def _get_backend():
    global _backend
    if _backend is None:
        _load_pipeline_mode()
        if _pipeline_mode == "cloud":
            config_path = _os.environ.get(
                "INTERNAL_CONFIG_PATH",
                _os.path.join(_server_dir, "setup.yaml")
            )
        else:
            config_path = _os.environ.get(
                "INTERNAL_CONFIG_PATH",
                _os.path.join(_server_dir, "internal_config.yaml")
            )
        _backend = _internal_sync.create_backend(
            pipeline_mode=_pipeline_mode,
            config_path=config_path,
        )
    return _backend


@app.route("/api/dataset/sync", methods=["POST"])
def dataset_sync():
    """触发数据集内网同步"""
    data = request.get_json(silent=True) or {}

    source_path = data.get("source_path", "")
    dataset_name = data.get("dataset_name", "")
    if not source_path:
        return jsonify({"error": "缺少 source_path 参数"}), 400
    if not dataset_name:
        dataset_name = Path(source_path).name

    sync_images = data.get("sync_images", True)
    sync_videos = data.get("sync_videos", False)

    try:
        syncer = _get_backend()
        result = syncer.sync_dataset(
            source_path=source_path,
            dataset_name=dataset_name,
            sync_images=sync_images,
            sync_videos=sync_videos,
        )
        return jsonify(result.to_dict())
    except Exception as e:
        return jsonify({"error": str(e), "status": "failed"}), 500


@app.route("/api/dataset/status", methods=["GET"])
def dataset_status():
    """查询同步/转换任务状态"""
    task_id = request.args.get("task_id", "")
    if not task_id:
        return jsonify({"error": "缺少 task_id 参数"}), 400

    status = _internal_sync.InternalSync.get_task_status(task_id)
    if status is None:
        return jsonify({"error": f"未找到任务: {task_id}"}), 404

    return jsonify(status)


@app.route("/api/dataset/convert", methods=["POST"])
def dataset_convert():
    """触发 raw → training-stage 格式转换"""
    data = request.get_json(silent=True) or {}

    source_path = data.get("source_path", "")
    output_path = data.get("output_path") or None
    if not source_path:
        return jsonify({"error": "缺少 source_path 参数"}), 400

    embed_images = data.get("embed_images", False)
    overwrite_existing = data.get("overwrite_existing", False)

    try:
        result = _internal_sync.convert_dataset(
            source_path=source_path,
            output_path=output_path,
            embed_images=embed_images,
            overwrite_existing=overwrite_existing,
        )
        return jsonify(result.to_dict())
    except Exception as e:
        return jsonify({"error": str(e), "status": "failed"}), 500


@app.route("/api/dataset/config_status", methods=["GET"])
def dataset_config_status():
    """返回当前管线配置状态：哪些字段已配，哪些缺失，以及 CLI 引导提示。"""
    try:
        syncer = _get_backend()
        status = syncer.get_config_status()
        return jsonify(status)
    except Exception as e:
        return jsonify({
            "mode": _pipeline_mode,
            "ok": False,
            "configured": [],
            "missing": ["backend_init_failed"],
            "hints": [f"Backend init error: {e}. Check setup.yaml / internal_config.yaml / baai_env.sh."],
        }), 500

@app.route("/api/dataset/connection_test", methods=["GET"])
def dataset_connection_test():
    """测试内网目标服务器连接"""
    try:
        syncer = _get_backend()
        result = syncer.test_connection()
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e), "reachable": False}), 500


@app.route("/api/dataset/tasks", methods=["GET"])
def dataset_tasks():
    """列出所有数据管线任务"""
    tasks = _internal_sync.InternalSync.list_tasks()
    return jsonify({"tasks": tasks})

def init_streams():
    """初始化视频流"""
    for stream_id, source in VIDEO_SOURCES.items():
        video_streams[stream_id] = VideoStream(stream_id, source)
        stream_status[stream_id] = {
            "name": f"{stream_id}",
            "active": False,
            "source": str(source),
        }


if __name__ == "__main__":
    init_streams()
    app.run(host="0.0.0.0", port=8080, debug=False, threaded=True)

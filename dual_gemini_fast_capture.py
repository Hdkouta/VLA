import os
import time
import queue
import threading
import cv2
import numpy as np
from pyorbbecsdk import (
    Context,
    Pipeline,
    Config,
    OBSensorType,
    OBFormat,
    OBError
)

# === 設定項目 ===
INTERVAL_MS  = 100        # キャプチャ周期（ミリ秒単位）
TARGET_WIDTH = 320        # 出力画像幅
TARGET_HEIGHT = 240       # 出力画像高さ
FPS          = 30         # カメラの内部動作FPS
SAVE_ROOT    = "./captures"

# 非同期保存用キュー
save_queue = queue.Queue(maxsize=200)
stop_event = threading.Event()

def image_writer_worker():
    """ディスクI/Oによる周期遅れを防ぐ非同期保存スレッド"""
    while not stop_event.is_set() or not save_queue.empty():
        try:
            file_path, image = save_queue.get(timeout=0.05)
            cv2.imwrite(file_path, image, [cv2.IMWRITE_JPEG_QUALITY, 90])
            save_queue.task_done()
        except queue.Empty:
            continue

def setup_camera(device):
    """カメラのカラーパイプラインを初期化"""
    dev_info = device.get_device_info()
    sn = dev_info.get_serial_number()
    name = dev_info.get_name()

    pipe = Pipeline(device)
    config = Config()

    color_profiles = pipe.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
    
    # デフォルトのストリームプロファイルを使用する
    # 無理にRGBや特定の解像度を指定せず、カメラが安定して出せるフォーマットを使用
    # 2台同時の USB 帯域対策として低解像度を優先し、無ければ既定にフォールバック
    try:
        color_profile = color_profiles.get_video_stream_profile(640, 480, OBFormat.MJPG, FPS)
    except OBError:
        color_profile = color_profiles.get_default_video_stream_profile()
    
    # 取得したプロファイルの情報を表示（デバッグ用）
    actual_width = color_profile.get_width()
    actual_height = color_profile.get_height()
    actual_format = color_profile.get_format()
    print(f"[{sn}] 採用プロファイル: {actual_width}x{actual_height}, フォーマット: {actual_format}")

    config.enable_stream(color_profile)
    pipe.start(config)

    return {
        "pipeline": pipe,
        "serial": sn,
        "name": name,
        "dir": os.path.join(SAVE_ROOT, f"{name}_{sn}")
    }

def process_color_frame(color_frame):
    """カメラから取得したフレームのフォーマットに応じて BGR 画像に変換する"""
    width = color_frame.get_width()
    height = color_frame.get_height()
    fmt = color_frame.get_format()
    data = np.asanyarray(color_frame.get_data())

    img_bgr = None

    if fmt == OBFormat.RGB:
        # RGB の場合 (H, W, 3) に reshape して BGR に変換
        img = np.resize(data, (height, width, 3))
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    elif fmt == OBFormat.YUYV or fmt == OBFormat.YUY2:
        # YUYV の場合 (H, W, 2) に reshape して BGR に変換
        img = np.resize(data, (height, width, 2))
        img_bgr = cv2.cvtColor(img, cv2.COLOR_YUV2BGR_YUYV)

    elif fmt == OBFormat.MJPG:
        # MJPG の場合、OpenCV の imdecode で展開
        img_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)

    elif fmt == OBFormat.UYVY:
        # UYVY の場合 (H, W, 2) に reshape して BGR に変換
        img = np.resize(data, (height, width, 2))
        img_bgr = cv2.cvtColor(img, cv2.COLOR_YUV2BGR_UYVY)
        
    elif fmt == OBFormat.I420:
        # I420 の場合 (H * 1.5, W) に reshape して BGR に変換
        img = np.resize(data, (int(height * 1.5), width))
        img_bgr = cv2.cvtColor(img, cv2.COLOR_YUV2BGR_I420)
        
    elif fmt == OBFormat.NV21:
        # NV21 の場合 (H * 1.5, W) に reshape して BGR に変換
        img = np.resize(data, (int(height * 1.5), width))
        img_bgr = cv2.cvtColor(img, cv2.COLOR_YUV2BGR_NV21)

    else:
        print(f"未対応のフォーマットです: {fmt}")
        return None

    return img_bgr

def main():
    ctx = Context()
    dev_list = ctx.query_devices()
    dev_count = dev_list.get_count()

    if dev_count < 2:
        print(f"エラー: 2台のカメラが必要です（検出数: {dev_count}台）")
        return

    cameras = []
    print("--- 2台のカメラをセットアップ中 ---")
    for i in range(2):
        dev = dev_list.get_device_by_index(i)
        try:
            cam = setup_camera(dev)
        except OBError as e:
            print(f"カメラ {i} の起動に失敗: {e}")
            for c in cameras:
                c["pipeline"].stop()
            return
        os.makedirs(cam["dir"], exist_ok=True)
        cameras.append(cam)

    writer_thread = threading.Thread(target=image_writer_worker, daemon=True)
    writer_thread.start()

    time.sleep(1.5)
    interval_sec = INTERVAL_MS / 1000.0
    print(f"\nキャプチャ開始 [周期: {INTERVAL_MS} ms] (終了: Ctrl+C)")

    next_capture_time = time.perf_counter()

    try:
        while True:
            timestamp_ms = int(time.time() * 1000)

            for cam in cameras:
                try:
                    frames = cam["pipeline"].wait_for_frames(50)
                except OBError as e:
                    print(f"[{cam['serial']}] OBError: {e}")
                    continue
                if frames is None:
                    continue

                color_frame = frames.get_color_frame()
                if color_frame is None:
                    continue

                # フレームのフォーマットに応じた適切な変換処理を実行
                img_bgr = process_color_frame(color_frame)
                
                if img_bgr is None:
                    continue

                # 指定解像度 (320x240) にリサイズ
                img_bgr = cv2.resize(
                    img_bgr, 
                    (TARGET_WIDTH, TARGET_HEIGHT), 
                    interpolation=cv2.INTER_AREA
                )

                save_path = os.path.join(cam["dir"], f"{timestamp_ms}.jpg")
                try:
                    save_queue.put_nowait((save_path, img_bgr))
                except queue.Full:
                    pass # キュー溢れ時の処理

            next_capture_time += interval_sec
            sleep_duration = next_capture_time - time.perf_counter()
            if sleep_duration > 0:
                time.sleep(sleep_duration)
            else:
                next_capture_time = time.perf_counter()

    except KeyboardInterrupt:
        print("\n停止シグナルを受信しました。")
    finally:
        stop_event.set()
        writer_thread.join()
        for cam in cameras:
            cam["pipeline"].stop()
        print("完了しました。")

if __name__ == "__main__":
    main()
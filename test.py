import pyrealsense2 as rs

# 파이프라인 시작 (IR 좌/우 스트림 활성화)
pipeline = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.infrared, 1, 848, 480, rs.format.y8, 90)  # Left IR
cfg.enable_stream(rs.stream.infrared, 2, 848, 480, rs.format.y8, 90)  # Right IR
profile = pipeline.start(cfg)

# 스트림 프로파일 얻기
streams = profile.get_streams()
ir1 = next(s.as_video_stream_profile() for s in streams
           if s.stream_type() == rs.stream.infrared and s.stream_index() == 1)
ir2 = next(s.as_video_stream_profile() for s in streams
           if s.stream_type() == rs.stream.infrared and s.stream_index() == 2)

# extrinsics (ir1 -> ir2)
extr = ir1.get_extrinsics_to(ir2)

# 베이스라인 (m) : 일반적으로 x축 이동 값이 음수이며 절댓값이 베이스라인
baseline_m = abs(extr.translation[0])

print(f"Baseline: {baseline_m}")  # 보통 ~50.0 mm 근처
pipeline.stop()

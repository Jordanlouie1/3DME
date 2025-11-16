# Pose Orientation Utility (Standalone)

Work on your own `3DImageStitcher` project without touching `MoGe/` by dropping
`pose_orientation.py` next to your custom scripts.  The helper only depends on
MediaPipe's pose world landmarks, so it remains portable and easy to hack on.

The landmarks are expressed in meters and use the following coordinate frame:

- +X → subject's right shoulder
- +Y → up
- +Z → away from the camera

## Usage

```python
import cv2
import mediapipe as mp
from pose_orientation import estimate_torso_yaw_from_world_landmarks

pose = mp.solutions.pose.Pose(static_image_mode=True, model_complexity=2)
rgb = cv2.cvtColor(cv2.imread("person.jpg"), cv2.COLOR_BGR2RGB)
result = pose.process(rgb)

if result.pose_world_landmarks:
    orientation = estimate_torso_yaw_from_world_landmarks(
        result.pose_world_landmarks.landmark
    )
    print(
        f"Facing {orientation.facing}, yaw={orientation.yaw_degrees:.1f}°, "
        f"confidence={orientation.confidence:.2f}"
    )
else:
    print("Pose landmarks missing; re-frame the person and try again.")
```

Positive yaw means the **left** shoulder is closer to the camera, negative yaw
means the **right** shoulder is closer.  Adjust `yaw_threshold_degrees` if you
need stricter or looser side detection.

## Tips

- The helper checks the MediaPipe `visibility` score; lower the
  `min_visibility` argument or skip those frames when landmarks are uncertain.
- Only the shoulders are used, which makes the calculation robust to motion and
  compatible with both photos and live video.
- Because the module lives outside `MoGe/`, you can version it alongside the
  rest of your `3DImageStitcher` code and still import any upstream MoGe APIs
  via `pip install -e ./MoGe` or a standard pip install.

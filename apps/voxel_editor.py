import cv2
import mediapipe as mp
import math
import numpy as np

PINCH_THRESHOLD = 0.1
BLOCK_SIZE = 50
GRID_COLOR = (255, 0, 0)
SMOOTHING_SPEED = 0.2

FOCAL_LENGTH = 600
CAMERA_Z_OFFSET = 400

voxel_map = set()
smooth_x = 0
smooth_y = 0

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1)

cap = cv2.VideoCapture(0)


def project_point(x, y, z, cx, cy):
    if z + CAMERA_Z_OFFSET <= 0.1:
        return None

    scale = FOCAL_LENGTH / (z + CAMERA_Z_OFFSET)
    u = int(x * scale + cx)
    v = int(y * scale + cy)

    return (u, v)


def draw_perspective_cube(img, x, y, z, size, color, thickness, cx, cy):
    s = size / 2
    corners = [
        (x-s, y-s, z-s), (x+s, y-s, z-s), (x+s, y+s, z-s), (x-s, y+s, z-s),
        (x-s, y-s, z+s), (x+s, y-s, z+s), (x+s, y+s, z+s), (x-s, y+s, z+s)
    ]

    proj_points = []
    for px, py, pz in corners:
        pt = project_point(px, py, pz, cx, cy)
        if pt is None:
            return
        proj_points.append(pt)

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]

    for s, e in edges:
        cv2.line(img, proj_points[s], proj_points[e], color, thickness)


print("Press 'q' to quit")

while True:
    success, frame = cap.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    h, w, _ = frame.shape

    cx, cy = w // 2, h // 2

    for voxel in voxel_map:
        vx, vy = voxel
        world_x = vx * BLOCK_SIZE
        world_y = vy * BLOCK_SIZE
        world_z = 0
        draw_perspective_cube(frame, world_x, world_y, world_z, BLOCK_SIZE, GRID_COLOR, 2, cx, cy)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            thumb = hand_landmarks.landmark[4]
            index = hand_landmarks.landmark[8]
            distance = math.hypot(index.x - thumb.x, index.y - thumb.y)

            hand_x = (thumb.x + index.x) / 2 * w - cx
            hand_y = (thumb.y + index.y) / 2 * h - cy

            target_grid_x = round(hand_x / BLOCK_SIZE) * BLOCK_SIZE
            target_grid_y = round(hand_y / BLOCK_SIZE) * BLOCK_SIZE

            smooth_x += (target_grid_x - smooth_x) * SMOOTHING_SPEED
            smooth_y += (target_grid_y - smooth_y) * SMOOTHING_SPEED

            save_grid_x = round(hand_x / BLOCK_SIZE)
            save_grid_y = round(hand_y / BLOCK_SIZE)

            if distance < PINCH_THRESHOLD:
                draw_perspective_cube(frame, smooth_x, smooth_y, 0, BLOCK_SIZE, (0, 255, 0), 4, cx, cy)
                voxel_map.add((save_grid_x, save_grid_y))
            else:
                draw_perspective_cube(frame, smooth_x, smooth_y, 0, BLOCK_SIZE, (0, 255, 255), 2, cx, cy)

    cv2.imshow('BoxelXR 3D Perspective', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
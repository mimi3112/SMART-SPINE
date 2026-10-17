import cv2
import mediapipe as mp
import time
import joblib
import pandas as pd
import numpy as np
import os
from datetime import datetime


# ============================================================
# 1. PATHS
# ============================================================

MODEL_PATH = "pose_landmarker_full.task"
POSTURE_MODEL_PATH = "models/posture_model.pkl"
LOG_FILE = "results/live_analysis.csv"

os.makedirs("results", exist_ok=True)


# ============================================================
# 2. BODY WEIGHT
# ============================================================

try:

    body_weight = float(
        input("Enter body weight (kg): ")
    )

except ValueError:

    body_weight = 60.0

    print(
        "Invalid input. Using 60 kg."
    )


# ============================================================
# 3. LOAD TRAINED POSTURE MODEL
# ============================================================

try:

    model_package = joblib.load(
        POSTURE_MODEL_PATH
    )

    posture_model = model_package["model"]

    feature_columns = (
        model_package["feature_columns"]
    )

    print(
        "Posture model loaded successfully."
    )

    print(
        "Number of features expected:",
        len(feature_columns)
    )

except Exception as e:

    print(
        "ERROR: Could not load posture model."
    )

    print(e)

    exit()


# ============================================================
# 4. POSTURE NAMES
# ============================================================

posture_names = {

    "TLB":
        "TRUNK LEANING BACKWARD",

    "TLF":
        "TRUNK LEANING FORWARD",

    "TLL":
        "TRUNK LEANING LEFT",

    "TLR":
        "TRUNK LEANING RIGHT",

    "TUP":
        "UPRIGHT"
}


# ============================================================
# 5. CERVICAL LOAD ESTIMATION
# ============================================================

def estimate_cervical_load(
    cva,
    body_weight
):

    # --------------------------------------------------------
    # Project-level flexion estimate
    # --------------------------------------------------------

    flexion = np.clip(
        90 - cva,
        0,
        60
    )


    # --------------------------------------------------------
    # Reference angles
    # --------------------------------------------------------

    reference_angles = np.array([

        0,
        15,
        30,
        45,
        60

    ])


    # --------------------------------------------------------
    # Reference cervical loads
    # Unit: pounds-force
    # --------------------------------------------------------

    reference_loads_lb = np.array([

        11,
        27,
        40,
        49,
        60

    ])


    # --------------------------------------------------------
    # Linear interpolation
    # --------------------------------------------------------

    load_lb = np.interp(

        flexion,

        reference_angles,

        reference_loads_lb
    )


    # --------------------------------------------------------
    # Estimate head mass
    # --------------------------------------------------------

    head_mass = (

        body_weight
        *
        0.08
    )


    # --------------------------------------------------------
    # Scale relative to 6 kg reference
    # --------------------------------------------------------

    load_lb = (

        load_lb
        *
        (
            head_mass
            /
            6.0
        )
    )


    # --------------------------------------------------------
    # Pounds-force -> Newton
    # --------------------------------------------------------

    load_newton = (

        load_lb
        *
        4.44822
    )


    # --------------------------------------------------------
    # Newton -> kg-equivalent
    # --------------------------------------------------------

    load_kg = (

        load_newton
        /
        9.80665
    )


    return load_kg, flexion


# ============================================================
# 6. MEDIAPIPE LANDMARK MAPPING
# ============================================================

landmark_mapping = {

    "nose": 0,

    "left_eye_inner": 1,
    "left_eye": 2,
    "left_eye_outer": 3,

    "right_eye_inner": 4,
    "right_eye": 5,
    "right_eye_outer": 6,

    "left_ear": 7,
    "right_ear": 8,

    "mouth_left": 9,
    "mouth_right": 10,

    "left_shoulder": 11,
    "right_shoulder": 12,

    "left_elbow": 13,
    "right_elbow": 14,

    "left_wrist": 15,
    "right_wrist": 16,

    "left_pinky": 17,
    "right_pinky": 18,

    "left_index": 19,
    "right_index": 20,

    "left_thumb": 21,
    "right_thumb": 22,

    "left_hip": 23,
    "right_hip": 24,

    "left_knee": 25,
    "right_knee": 26,

    "left_ankle": 27,
    "right_ankle": 28,

    "left_heel": 29,
    "right_heel": 30,

    "left_foot_index": 31,
    "right_foot_index": 32
}


# ============================================================
# 7. CREATE LIVE LOG FILE
# ============================================================

if not os.path.exists(LOG_FILE):

    columns = [

        "timestamp",

        "body_weight_kg",

        "posture",

        "confidence",

        "cva_proxy_deg",

        "estimated_flexion_deg",

        "estimated_cervical_load_kg"
    ]

    pd.DataFrame(
        columns=columns
    ).to_csv(

        LOG_FILE,

        index=False
    )


# ============================================================
# 8. MEDIAPIPE SETUP
# ============================================================

BaseOptions = (
    mp.tasks.BaseOptions
)

PoseLandmarker = (
    mp.tasks.vision.PoseLandmarker
)

PoseLandmarkerOptions = (
    mp.tasks.vision.PoseLandmarkerOptions
)

RunningMode = (
    mp.tasks.vision.RunningMode
)


options = PoseLandmarkerOptions(

    base_options=BaseOptions(

        model_asset_path=MODEL_PATH
    ),

    running_mode=RunningMode.VIDEO,

    num_poses=1
)


# ============================================================
# 9. CAMERA
# ============================================================

cap = cv2.VideoCapture(0)


if not cap.isOpened():

    print(
        "ERROR: Camera could not be opened."
    )

    exit()


print()

print("=" * 70)

print(
    "SMART-SPINE REAL-TIME ANALYSIS"
)

print("=" * 70)

print(
    "Body weight:",
    body_weight,
    "kg"
)

print(
    "Expected model features:",
    len(feature_columns)
)

print(
    "Press Q to quit."
)

print("=" * 70)


start_time = time.time()

last_save_time = 0


# ============================================================
# 10. MEDIAPIPE
# ============================================================

with PoseLandmarker.create_from_options(
    options
) as landmarker:

    while True:

        # ====================================================
        # READ CAMERA
        # ====================================================

        ret, frame = cap.read()


        if not ret:

            print(
                "ERROR: Could not read frame."
            )

            break


        # ====================================================
        # RESIZE
        # ====================================================

        frame = cv2.resize(

            frame,

            (640, 480)
        )


        # ====================================================
        # BGR -> RGB
        # ====================================================

        rgb = cv2.cvtColor(

            frame,

            cv2.COLOR_BGR2RGB
        )


        # ====================================================
        # CREATE MEDIAPIPE IMAGE
        # ====================================================

        mp_image = mp.Image(

            image_format=(
                mp.ImageFormat.SRGB
            ),

            data=rgb
        )


        # ====================================================
        # TIMESTAMP
        # ====================================================

        timestamp_ms = int(

            (

                time.time()
                -
                start_time

            )
            *
            1000
        )


        # ====================================================
        # DETECT POSE
        # ====================================================

        result = (

            landmarker.detect_for_video(

                mp_image,

                timestamp_ms
            )
        )


        # ====================================================
        # PERSON DETECTED
        # ====================================================

        if result.pose_landmarks:

            landmarks = (

                result.pose_landmarks[0]
            )

            h, w, _ = frame.shape


            # =================================================
            # 11. IMPORTANT LANDMARKS
            # =================================================

            left_ear = (
                landmarks[7]
            )

            right_ear = (
                landmarks[8]
            )

            left_shoulder = (
                landmarks[11]
            )

            right_shoulder = (
                landmarks[12]
            )


            # =================================================
            # 12. CVA-BASED FEATURE
            # =================================================

            # LEFT SIDE
            # -------------------------------------------------

            dx_left = (

                left_ear.x
                -
                left_shoulder.x
            )


            dy_left = (

                left_ear.y
                -
                left_shoulder.y
            )


            left_cva = np.degrees(

                np.arctan2(

                    abs(dy_left),

                    abs(dx_left)
                )
            )


            # RIGHT SIDE
            # -------------------------------------------------

            dx_right = (

                right_ear.x
                -
                right_shoulder.x
            )


            dy_right = (

                right_ear.y
                -
                right_shoulder.y
            )


            right_cva = np.degrees(

                np.arctan2(

                    abs(dy_right),

                    abs(dx_right)
                )
            )


            # AVERAGE
            # -------------------------------------------------

            cva = (

                left_cva
                +
                right_cva

            ) / 2


            # =================================================
            # 13. CERVICAL LOAD
            # =================================================

            cervical_load, flexion = (

                estimate_cervical_load(

                    cva,

                    body_weight
                )
            )


            # =================================================
            # 14. CREATE ML FEATURE VECTOR
            # =================================================

            feature_values = []


            for column in feature_columns:

                parts = column.rsplit(

                    "_",

                    1
                )


                # ---------------------------------------------
                # Invalid feature name
                # ---------------------------------------------

                if len(parts) != 2:

                    feature_values.append(0)

                    continue


                landmark_name = (
                    parts[0]
                )

                coordinate = (
                    parts[1]
                )


                # ---------------------------------------------
                # Find MediaPipe landmark index
                # ---------------------------------------------

                landmark_index = (

                    landmark_mapping.get(

                        landmark_name
                    )
                )


                # ---------------------------------------------
                # Unknown landmark
                # ---------------------------------------------

                if landmark_index is None:

                    feature_values.append(0)

                    continue


                point = landmarks[
                    landmark_index
                ]


                # ---------------------------------------------
                # X
                # ---------------------------------------------

                if coordinate == "x":

                    feature_values.append(

                        point.x
                    )


                # ---------------------------------------------
                # Y
                # ---------------------------------------------

                elif coordinate == "y":

                    feature_values.append(

                        point.y
                    )


                # ---------------------------------------------
                # Z
                # ---------------------------------------------

                elif coordinate == "z":

                    feature_values.append(

                        point.z
                    )


                # ---------------------------------------------
                # Unknown coordinate
                # ---------------------------------------------

                else:

                    feature_values.append(0)


            # =================================================
            # 15. DEBUG FEATURE COUNT
            # =================================================

            if len(feature_values) != len(
                feature_columns
            ):

                print()

                print(
                    "WARNING: FEATURE COUNT MISMATCH"
                )

                print(
                    "Expected:",
                    len(feature_columns)
                )

                print(
                    "Received:",
                    len(feature_values)
                )


            # =================================================
            # 16. CREATE DATAFRAME
            # =================================================

            X_live = pd.DataFrame(

                [feature_values],

                columns=feature_columns
            )


            # =================================================
            # 17. POSTURE PREDICTION
            # =================================================

            prediction = (

                posture_model.predict(

                    X_live

                )[0]
            )


            # =================================================
            # 18. CONFIDENCE
            # =================================================

            try:

                probabilities = (

                    posture_model.predict_proba(

                        X_live

                    )[0]
                )


                confidence = (

                    np.max(

                        probabilities

                    )
                    *
                    100
                )


            except Exception:

                probabilities = None

                confidence = 0.0


            # =================================================
            # 19. PRINT DEBUG INFORMATION
            # =================================================

            print()

            print("-" * 70)

            print(
                "LIVE MODEL DEBUG"
            )

            print("-" * 70)

            print(
                "Expected features:",
                len(feature_columns)
            )

            print(
                "Received features:",
                len(feature_values)
            )

            print(
                "Prediction:",
                prediction
            )

            print(
                "CVA:",
                f"{cva:.2f} degrees"
            )

            print(
                "Flexion:",
                f"{flexion:.2f} degrees"
            )

            print(
                "Cervical load:",
                f"{cervical_load:.2f} kg-equivalent"
            )


            # =================================================
            # 20. PRINT ALL CLASS PROBABILITIES
            # =================================================

            if probabilities is not None:

                print()

                print(
                    "CLASS PROBABILITIES:"
                )

                for (

                    class_name,

                    probability

                ) in zip(

                    posture_model.classes_,

                    probabilities

                ):

                    print(

                        f"{class_name}: "
                        f"{probability * 100:.2f}%"
                    )


            print("-" * 70)


            # =================================================
            # 21. POSTURE TEXT
            # =================================================

            posture_text = (

                posture_names.get(

                    prediction,

                    prediction
                )
            )


            # =================================================
            # 22. GENERAL POSTURE STATUS
            # =================================================

            if prediction == "TUP":

                posture_status = (
                    "GOOD"
                )

            else:

                posture_status = (
                    "NEEDS ADJUSTMENT"
                )


            # =================================================
            # 23. DISPLAY LANDMARKS
            # =================================================

            left_ear_xy = (

                int(
                    left_ear.x * w
                ),

                int(
                    left_ear.y * h
                )
            )


            right_ear_xy = (

                int(
                    right_ear.x * w
                ),

                int(
                    right_ear.y * h
                )
            )


            left_shoulder_xy = (

                int(
                    left_shoulder.x * w
                ),

                int(
                    left_shoulder.y * h
                )
            )


            right_shoulder_xy = (

                int(
                    right_shoulder.x * w
                ),

                int(
                    right_shoulder.y * h
                )
            )


            # =================================================
            # 24. DRAW 4 RELEVANT POINTS
            # =================================================

            for point in [

                left_ear_xy,

                right_ear_xy,

                left_shoulder_xy,

                right_shoulder_xy

            ]:

                cv2.circle(

                    frame,

                    point,

                    7,

                    (0, 255, 0),

                    -1
                )


            # =================================================
            # 25. EAR -> SHOULDER LINES
            # =================================================

            cv2.line(

                frame,

                left_ear_xy,

                left_shoulder_xy,

                (0, 255, 0),

                2
            )


            cv2.line(

                frame,

                right_ear_xy,

                right_shoulder_xy,

                (0, 255, 0),

                2
            )


            # =================================================
            # 26. HEADER
            # =================================================

            cv2.putText(

                frame,

                "SMART-SPINE",

                (15, 30),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.8,

                (0, 255, 0),

                2
            )


            # =================================================
            # 27. POSTURE
            # =================================================

            cv2.putText(

                frame,

                "POSTURE: "
                +
                posture_text,

                (15, 65),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.58,

                (255, 255, 255),

                2
            )


            # =================================================
            # 28. STATUS
            # =================================================

            status_color = (

                (0, 255, 0)

                if posture_status == "GOOD"

                else

                (0, 165, 255)
            )


            cv2.putText(

                frame,

                "STATUS: "
                +
                posture_status,

                (15, 95),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.65,

                status_color,

                2
            )


            # =================================================
            # 29. CONFIDENCE
            # =================================================

            cv2.putText(

                frame,

                f"MODEL CONFIDENCE: "
                f"{confidence:.1f}%",

                (15, 125),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.55,

                (255, 255, 255),

                2
            )


            # =================================================
            # 30. CVA
            # =================================================

            cv2.putText(

                frame,

                f"CVA: "
                f"{cva:.1f} deg",

                (15, 160),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.65,

                (255, 255, 255),

                2
            )


            # =================================================
            # 31. CERVICAL LOAD
            # =================================================

            cv2.putText(

                frame,

                f"CERVICAL LOAD: "
                f"{cervical_load:.1f} kg-eq",

                (15, 195),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.60,

                (255, 255, 255),

                2
            )


            # =================================================
            # 32. BODY WEIGHT
            # =================================================

            cv2.putText(

                frame,

                f"BODY WEIGHT: "
                f"{body_weight:.1f} kg",

                (15, 230),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.60,

                (255, 255, 255),

                2
            )


            # =================================================
            # 33. SAVE LIVE OBSERVATION
            # =================================================

            current_time = time.time()


            # Save once every second
            if (

                current_time
                -
                last_save_time

                >= 1

            ):

                new_row = pd.DataFrame([{

                    "timestamp":

                        datetime.now().strftime(

                            "%Y-%m-%d %H:%M:%S"
                        ),


                    "body_weight_kg":

                        body_weight,


                    "posture":

                        prediction,


                    "confidence":

                        round(

                            confidence,

                            2
                        ),


                    "cva_proxy_deg":

                        round(

                            cva,

                            2
                        ),


                    "estimated_flexion_deg":

                        round(

                            flexion,

                            2
                        ),


                    "estimated_cervical_load_kg":

                        round(

                            cervical_load,

                            2
                        )

                }])


                new_row.to_csv(

                    LOG_FILE,

                    mode="a",

                    header=False,

                    index=False
                )


                last_save_time = (
                    current_time
                )


        # ====================================================
        # NO PERSON
        # ====================================================

        else:

            cv2.putText(

                frame,

                "NO PERSON DETECTED",

                (15, 40),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.75,

                (0, 0, 255),

                2
            )


        # ====================================================
        # 34. DISPLAY
        # ====================================================

        cv2.imshow(

            "Smart-Spine - Real-Time Analysis",

            frame
        )


        # ====================================================
        # 35. QUIT
        # ====================================================

        if (

            cv2.waitKey(1) & 0xFF

            ==
            ord("q")

        ):

            break


# ============================================================
# 36. CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()


print()

print("=" * 70)

print(
    "SMART-SPINE ANALYSIS STOPPED"
)

print("=" * 70)

print(
    "Live analysis saved to:"
)

print(LOG_FILE)

print("=" * 70)
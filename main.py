import cv2
import numpy as np
import gradio as gr
from insightface.app import FaceAnalysis


# ============================================================
# AGE & GENDER DETECTION
# InsightFace + ONNX Runtime + Gradio
# ============================================================

print("Loading Age & Gender Detection model...")

# InsightFace model
# CPUExecutionProvider makes this work on normal CPU hardware
# and on Hugging Face CPU Spaces.
app = FaceAnalysis(
    name="buffalo_l",
    allowed_modules=["detection", "genderage"],
    providers=["CPUExecutionProvider"]
)

app.prepare(
    ctx_id=-1,
    det_size=(640, 640)
)

print("Model loaded successfully.")


def detect_age_gender(image):
    """
    Receives an RGB image from Gradio,
    performs face detection and age/gender prediction,
    and returns an annotated RGB image plus text results.
    """

    if image is None:
        return None, "Please upload an image or take a photo using the webcam."

    try:
        # Gradio provides RGB image.
        # OpenCV/InsightFace works with BGR.
        frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Run InsightFace
        faces = app.get(frame)

        result_lines = []

        if len(faces) == 0:
            cv2.putText(
                frame,
                "No face detected",
                (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 80, 255),
                2,
                cv2.LINE_AA
            )

            result_lines.append("No face detected.")

        else:
            for index, face in enumerate(faces, start=1):

                # Bounding box
                x1, y1, x2, y2 = face.bbox.astype(int)

                # Keep coordinates inside image
                height, width = frame.shape[:2]

                x1 = max(0, min(x1, width - 1))
                y1 = max(0, min(y1, height - 1))
                x2 = max(0, min(x2, width - 1))
                y2 = max(0, min(y2, height - 1))

                # Age
                age = int(round(float(face.age)))

                # InsightFace gender:
                # 1 = Male
                # 0 = Female
                gender = "Male" if int(face.gender) == 1 else "Female"

                # Colors are BGR for OpenCV
                if gender == "Male":
                    box_color = (255, 180, 50)
                else:
                    box_color = (220, 80, 200)

                label = f"{gender} | Age: {age}"

                # Draw bounding box
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    box_color,
                    2
                )

                # Calculate label size
                (
                    (text_width, text_height),
                    baseline
                ) = cv2.getTextSize(
                    label,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.70,
                    2
                )

                # Label position
                label_top = max(
                    0,
                    y1 - text_height - baseline - 10
                )

                label_bottom = y1

                # Draw label background
                cv2.rectangle(
                    frame,
                    (x1, label_top),
                    (x1 + text_width + 12, label_bottom),
                    box_color,
                    -1
                )

                # Draw label text
                cv2.putText(
                    frame,
                    label,
                    (x1 + 6, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.70,
                    (0, 0, 0),
                    2,
                    cv2.LINE_AA
                )

                # Add result information
                result_lines.append(
                    f"Face {index}: Gender = {gender}, Age = {age}"
                )

        # Convert BGR back to RGB for Gradio
        output_image = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        result_text = "\n".join(result_lines)

        return output_image, result_text

    except Exception as error:
        return image, f"Error during detection: {error}"


# ============================================================
# GRADIO WEB INTERFACE
# ============================================================

with gr.Blocks(
    title="Age & Gender Detection System",
    css="""
    footer {
        display: none !important;
    }

    [data-testid="footer"] {
        display: none !important;
    }

    [data-testid="settings-button"] {
        display: none !important;
    }

    button[aria-label*="Settings"] {
        display: none !important;
    }

    a[href*="/gradio_api/runs"] {
        display: none !important;
    }

    a[href*="gradio.app"] {
        display: none !important;
    }
    """
) as demo:

    gr.Markdown(
        """
        # 👤 Age & Gender Detection System
        Upload an image or use your webcam to detect faces
        and estimate **age and gender**.

        
        """
    )

    with gr.Row():

        with gr.Column():

            input_image = gr.Image(
                label="Input Image / Webcam",
                sources=["upload", "webcam"],
                type="numpy"
            )

            detect_button = gr.Button(
                "🔍 Detect Age & Gender",
                variant="primary"
            )

        with gr.Column():

            output_image = gr.Image(
                label="Detection Result",
                type="numpy"
            )

            output_text = gr.Textbox(
                label="Prediction",
                lines=5
            )

    detect_button.click(
        fn=detect_age_gender,
        inputs=input_image,
        outputs=[output_image, output_text]
    )

    gr.Markdown(
        """
        ---
        ### ⚠️ Note

        Age and gender predictions may not always be accurate.

        """
    )


# ============================================================
# START APPLICATION
# ============================================================


if __name__ == "__main__":
    demo.launch(
        css="""
        footer {
            display: none !important;
        }

        .gradio-container > .contain {
            display: none !important;
        }

        [data-testid="footer"] {
            display: none !important;
        }

        [data-testid="settings-button"] {
            display: none !important;
        }

        button[aria-label*="Settings"] {
            display: none !important;
        }

        a[href*="/gradio_api/runs"] {
            display: none !important;
        }

        a[href*="gradio.app"] {
            display: none !important;
        }
        """
    )

import av
import time
import queue
import logging

from typing import List, NamedTuple

import cv2
import streamlit as st

from fast_alpr import ALPR
from fast_alpr.default_detector import PlateDetectorModel
from fast_alpr.default_ocr import OcrModel
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

logger = logging.getLogger(__name__)

result_queue: "queue.Queue[List[Detection]]" = queue.Queue()

class VideoProcessor(VideoProcessorBase):
    def __init__(self, detector_model, ocr_model):
        self.last_processed = 0  # Initialize last processed time
        self.alpr = ALPR(detector_model=detector_model, ocr_model=ocr_model)
        self.last_annotated = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        current_time = time.time()
        # Only process if 10 seconds have passed
        if (self.last_annotated is None) or (current_time - self.last_processed > 1):
            results = self.alpr.predict(img)
            annotated_img = self.alpr.draw_predictions(img)
            self.last_annotated = annotated_img
            self.last_processed = current_time

            result_queue.put(result.ocr.text for result in results)

        # Return the annotated frame if available, otherwise the original
        output_img = self.last_annotated if self.last_annotated is not None else img
        return av.VideoFrame.from_ndarray(output_img, format="bgr24")

# ['yolo-v9-s-608-license-plate-end2end', 'yolo-v9-t-640-license-plate-end2end', 'yolo-v9-t-512-license-plate-end2end', 'yolo-v9-t-416-license-plate-end2end', 'yolo-v9-t-384-license-plate-end2end', 'yolo-v9-t-256-license-plate-end2end']
detector_model = 'yolo-v9-t-256-license-plate-end2end'
# ['cct-s-v1-global-model', 'cct-xs-v1-global-model', 'argentinian-plates-cnn-model', 'argentinian-plates-cnn-synth-model', 'european-plates-mobile-vit-v2-model', 'global-plates-mobile-vit-v2-model']
ocr_model = 'cct-xs-v1-global-model'

# Pass the selected models into the processor
webrtc_streamer(
    key="example",
    video_processor_factory=lambda: VideoProcessor(detector_model, ocr_model),
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)


logger.log(result_queue.get())

if st.checkbox("Show the detected labels", value=True):
    labels_placeholder = st.empty()
    labels_placeholder.table(result_queue.get())
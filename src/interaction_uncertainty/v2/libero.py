"""Strict LIBERO observation adapter with a wrist-camera-only default.

The adapter intentionally ignores every simulator object-state field.  RGB is
identified by a content hash, preventing frame-name lookup from masquerading as
perception.  A deployable detector/tracker may attach temporary visual anchors.
"""

from __future__ import annotations

import base64
import hashlib
import io
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from tokenize import TokenError

import numpy as np

from ..observation import PolicyObservation, VisualAnchor

AnchorDetector = Callable[
    [Mapping[str, np.ndarray], str, str], Sequence[VisualAnchor]
]
ImagePublisher = Callable[[str, np.ndarray, str], str]

_EMBEDDED_IMAGE_PREFIX = "data:application/x-npy;sha256="
_MAX_EMBEDDED_IMAGE_BYTES = 16 * 1024 * 1024


def public_image_digest(image: np.ndarray) -> str:
    """Hash one public image without using a task, scenario, or frame name."""

    contiguous = np.ascontiguousarray(image)
    return hashlib.sha256(
        contiguous.dtype.str.encode("ascii")
        + str(contiguous.shape).encode("ascii")
        + contiguous.tobytes()
    ).hexdigest()


def embed_public_image(_key: str, image: np.ndarray, digest: str) -> str:
    """Encode a lossless, remotely decodable, content-addressed NumPy image.

    An application may replace this publisher with a content-addressed HTTPS or
    object-store URI.  The returned reference must communicate ``digest`` as the
    expected content identity; the receiving service must fetch the bytes and
    independently recompute that digest before trusting a custom reference.
    """

    contiguous = np.ascontiguousarray(image)
    buffer = io.BytesIO()
    np.save(buffer, contiguous, allow_pickle=False)
    encoded = buffer.getvalue()
    if len(encoded) > _MAX_EMBEDDED_IMAGE_BYTES:
        raise ValueError("embedded public image exceeds the 16 MiB per-image limit")
    payload = base64.b64encode(encoded).decode("ascii")
    return f"{_EMBEDDED_IMAGE_PREFIX}{digest};base64,{payload}"


def decode_embedded_public_image(reference: str) -> np.ndarray:
    """Decode and authenticate an image produced by :func:`embed_public_image`."""

    if not reference.startswith(_EMBEDDED_IMAGE_PREFIX) or ";base64," not in reference:
        raise ValueError("image reference is not an embedded content-addressed NumPy image")
    metadata, payload = reference.split(",", 1)
    digest = metadata.removeprefix(_EMBEDDED_IMAGE_PREFIX).removesuffix(";base64")
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError("embedded image reference contains an invalid SHA-256 digest")
    try:
        raw = base64.b64decode(payload, validate=True)
    except ValueError as exc:
        raise ValueError("embedded image reference contains invalid base64") from exc
    if len(raw) > _MAX_EMBEDDED_IMAGE_BYTES:
        raise ValueError("embedded public image exceeds the 16 MiB per-image limit")
    try:
        image = np.load(io.BytesIO(raw), allow_pickle=False)
    except (EOFError, OSError, TokenError, UnicodeError, ValueError) as exc:
        raise ValueError("embedded public image is not a valid non-pickled NumPy array") from exc
    if not isinstance(image, np.ndarray) or public_image_digest(image) != digest:
        raise ValueError("embedded public image failed its content-digest check")
    return image


@dataclass(frozen=True)
class LiberoPublicObservationAdapter:
    image_keys: tuple[str, ...] = ("robot0_eye_in_hand_image",)
    proprioception_keys: tuple[str, ...] = (
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
    )
    anchor_detector: AnchorDetector | None = None
    image_publisher: ImagePublisher = field(default=embed_public_image)

    def adapt(
        self,
        *,
        raw_observation: Mapping[str, object],
        frame_id: str,
        prompt: str,
        action_history: Sequence[str] = (),
    ) -> PolicyObservation:
        images: dict[str, np.ndarray] = {}
        refs: list[str] = []
        for key in self.image_keys:
            if key not in raw_observation:
                raise KeyError(f"missing public LIBERO image key: {key}")
            image = np.asarray(raw_observation[key])
            if image.ndim != 3 or image.shape[-1] not in {3, 4}:
                raise ValueError(f"{key} must be an HxWx3/4 image")
            if not np.issubdtype(image.dtype, np.number):
                raise TypeError(f"{key} must have a numeric dtype")
            if not np.all(np.isfinite(image)):
                raise ValueError(f"{key} contains non-finite pixels")
            contiguous = np.ascontiguousarray(image)
            digest = public_image_digest(contiguous)
            reference = self.image_publisher(key, contiguous, digest)
            if not reference.strip() or digest not in reference:
                raise ValueError(
                    "image publisher must return a non-empty content-addressed reference"
                )
            images[key] = contiguous
            refs.append(reference)

        proprioception: list[float] = []
        for key in self.proprioception_keys:
            if key not in raw_observation:
                continue
            raw_values = np.asarray(raw_observation[key])
            if np.issubdtype(raw_values.dtype, np.bool_):
                raise TypeError(f"{key} must be numeric, not boolean")
            values = np.asarray(raw_values, dtype=np.float64).reshape(-1)
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{key} contains non-finite proprioception")
            proprioception.extend(float(value) for value in values)
        anchors = (
            ()
            if self.anchor_detector is None
            else tuple(self.anchor_detector(images, prompt, frame_id))
        )
        return PolicyObservation(
            frame_id=frame_id,
            image_refs=tuple(refs),
            proprioception=tuple(proprioception),
            anchors=anchors,
            action_history=tuple(action_history),
        )

from __future__ import annotations

import pytest
from pydantic import ValidationError

from eemls.interfaces.control_plane_api import ClassifyRequest, ReturnedLabel
from eemls.interfaces.data_hub_api import ImageMetadata
from eemls.interfaces.inference_service_api import PredictionAnswer

ASKED = {
    "consumer_id": "c1_navigation",
    "image_id": "img-7",
    "reply_topic": "eemls/results/c1_navigation/client-1",
    "correlation_id": "corr-1",
}


def test_a_classify_request_asks_about_a_clear_image_unless_told_otherwise() -> None:
    assert ClassifyRequest.model_validate(ASKED).image_quality == "clear"


@pytest.mark.parametrize("model", [ClassifyRequest, ReturnedLabel, ImageMetadata, PredictionAnswer])
def test_a_field_the_interface_does_not_name_is_refused(model: type[ClassifyRequest]) -> None:
    with pytest.raises(ValidationError, match="unicorn"):
        model.model_validate({**ASKED, "unicorn": 1})

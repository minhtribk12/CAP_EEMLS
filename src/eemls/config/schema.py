from __future__ import annotations

from typing import Annotated

from pydantic import ConfigDict, Field

from eemls.domain.types import Digest as Digest
from eemls.domain.types import FrozenMap as FrozenMap
from eemls.domain.types import FrozenModel as FrozenModel
from eemls.domain.types import NonNegativeCount as NonNegativeCount
from eemls.domain.types import Number as Number
from eemls.domain.types import PositiveCount as PositiveCount
from eemls.domain.types import PositiveNumber as PositiveNumber
from eemls.domain.types import Ratio as Ratio
from eemls.domain.types import Weight as Weight

Port = Annotated[int, Field(ge=1, le=65535, strict=True)]
Source = Annotated[str, Field(min_length=1)]
UrlPath = Annotated[str, Field(pattern=r"^/[A-Za-z0-9/_.-]*$")]


class ExtraFieldsIgnored(FrozenModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

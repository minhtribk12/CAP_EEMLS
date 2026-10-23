from __future__ import annotations

import logging
from collections.abc import Mapping

from eemls.domain.consumer_defined_metrics import CdmSet
from eemls.domain.contract import ConsumerContract
from eemls.evaluation_engine.aggregation_functions import Value
from eemls.evaluation_engine.contract_scoring import NotScored, ScoreParts
from eemls.evaluation_engine.ensemble_scoring import scored_window
from eemls.evaluation_engine.window import Window

logger = logging.getLogger(__name__)


def window_size_for(cdms: CdmSet, *, at_least: int) -> int:
    declared = [one.window for one in cdms.definitions.values() if one.window]
    return max([*declared, at_least])


def evaluated_window(
    contract: ConsumerContract, cdms: CdmSet, window: Window
) -> tuple[Mapping[str, Value], ScoreParts | None]:
    if not window or not any(row["cost_per_1k"] is not None for row in window):
        return {}, None
    values, parts = scored_window(contract, cdms, window)
    if isinstance(parts, NotScored):
        logger.info(
            f"stage_tag=decide process_tag=orchestrator error_type=window_not_scored "
            f"consumer={contract.consumer_id} missing={list(parts.missing)} "
            f"recovery_action=the_decision_is_stored_without_a_score detail={parts.reason}"
        )
        return values, None
    return values, parts

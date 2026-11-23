import numpy as np
import logging

from core.config import MLSWeightConfig
from core.anchor import fit_1d_anchor
from core.flows import build_domain_adapted_signal_flow
from core.extractor import MLSWeightExtractor
from core.signal_models import MCSpline1D

logger = logging.getLogger(__name__)


class SequentialMLSWeightExtractor:
    """
    Executes N-Dimensional background subtraction by chaining 1D ML-sWeights sequentially.
    At step k, only not-yet-unfolded discriminating variables (j > k) are appended to X_ctrl,
    while already-unfolded variables (j < k) are marginalized over so their positive peak
    and negative sideband sWeights cancel in expectation inside every control cell.
    """
    def __init__(self, config: MLSWeightConfig = None):
        self.config = config if config is not None else MLSWeightConfig()
        self.extractors = []
        self.stage_weights = []
        self.stage_prob_weights = []

    def extract_weights(self, X_ctrl, Y_disc_nd, Y_bounds_list, mc_Y_nd=None,
                        f_sig_list=None, f_bkg_base_list=None, bkg_samplers=None,
                        anchor_orders=None, init_yields_list=None,
                        w_in=None, validation_data=None):
        """
        Sequential 1D Unfolding across D discriminating variables.
        """
        N_events, D = Y_disc_nd.shape
        w_curr = np.ones(N_events, dtype=np.float64) if w_in is None else np.asarray(w_in, dtype=np.float64).copy()
        w_prob_curr = w_curr.copy()

        if anchor_orders is None:
            anchor_orders = [2] * D

        self.extractors = []
        self.stage_weights = []
        self.stage_prob_weights = []

        for k in range(D):
            logger.info(f"\n=======================================================================")
            logger.info(f"=== SEQUENTIAL UNFOLDING: STEP {k + 1} / {D} (Discriminant y_{k + 1}) ===")
            logger.info(f"=======================================================================")

            y_k = Y_disc_nd[:, k]
            y_min, y_max = Y_bounds_list[k]

            # Append ONLY not-yet-unfolded discriminating variables (j > k) to X_ctrl.
            # Already-unfolded variables (j < k) MUST be marginalized over so their
            # negative sideband sWeights cancel against positive peak sWeights.
            remaining_Idx = [j for j in range(k + 1, D)]
            if len(remaining_Idx) > 0:
                X_ctrl_k = np.column_stack([X_ctrl, Y_disc_nd[:, remaining_Idx]])
            else:
                X_ctrl_k = X_ctrl

            # 1. Dynamic 1D Anchor Fit weighted by w_curr
            if f_bkg_base_list is not None and f_bkg_base_list[k] is not None:
                f_bkg_k = f_bkg_base_list[k]
                sampler_k = bkg_samplers[k]
                ns_k, nb_k = init_yields_list[k]
                d_mu_k, d_sig_k = None, None
            else:
                mc_y_k = mc_Y_nd[:, k] if mc_Y_nd is not None else None
                f_bkg_k, sampler_k, ns_k, nb_k, d_mu_k, d_sig_k = fit_1d_anchor(
                    y_k, y_min, y_max, mc_y=mc_y_k, order=anchor_orders[k], w_in=w_curr
                )

            # 2. Build 1D Signal Model
            if f_sig_list is not None and f_sig_list[k] is not None:
                f_sig_k = f_sig_list[k]
            else:
                mc_y_k = mc_Y_nd[:, k]
                if getattr(self.config, "signal_model_type", "spline").lower() == "flow":
                    f_sig_k = build_domain_adapted_signal_flow(
                        mc_y_k, y_k, y_min, y_max,
                        data_peak_Pos=d_mu_k, data_peak_sigma=d_sig_k,
                        device=self.config.device,
                        burn_in_iters=self.config.domain_burn_in_iters,
                        freeze_domain=self.config.freeze_domain_adaptation
                    )
                else:
                    f_sig_k = MCSpline1D(
                        mc_y_k, y_min, y_max,
                        burn_in_iters=self.config.domain_burn_in_iters,
                        freeze_domain=self.config.freeze_domain_adaptation
                    )

            # 3. Run 1D Extractor with w_in = w_curr
            ext_k = MLSWeightExtractor(
                f_sig_1d_mc=f_sig_k,
                f_bkg_1d_base=f_bkg_k,
                bkg_1d_sampler=sampler_k,
                y_min=y_min,
                y_max=y_max,
                config=self.config
            )

            w_curr, w_s_k = ext_k.extract_weights(
                X_ctrl_k, y_k,
                init_yields=(ns_k, nb_k),
                validation_data=validation_data if k == D - 1 else None,
                w_in=w_curr
            )

            w_prob_curr = w_prob_curr * w_s_k
            self.extractors.append(ext_k)
            self.stage_weights.append(w_curr.copy())
            self.stage_prob_weights.append(w_prob_curr.copy())

        return w_curr, w_prob_curr

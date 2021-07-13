import numpy as np
import airshower_template_generator as atg


class ShowerModel:
    def __init__(
        self,
        main_axis_azimuth,
        main_axis_support_cx,
        main_axis_support_cy,
        light_field_cx,
        light_field_cy,
        light_field_x,
        light_field_y,
    ):
        self.main_axis_azimuth = main_axis_azimuth
        self.cx = light_field_cx
        self.cy = light_field_cy
        self.x = light_field_x
        self.y = light_field_y
        self.main_axis_support_cx = main_axis_support_cx
        self.main_axis_support_cy = main_axis_support_cy

    def _source_direction_cx_cy(self, c_para):
        source_cx = (
            self.main_axis_support_cx + np.cos(self.main_axis_azimuth) * c_para
        )
        source_cy = (
            self.main_axis_support_cy + np.sin(self.main_axis_azimuth) * c_para
        )
        return source_cx, source_cy

    def _core_position_x_y(self, r_para):
        core_x = 0.0 + np.cos(self.main_axis_azimuth) * r_para
        core_y = 0.0 + np.sin(self.main_axis_azimuth) * r_para
        return core_x, core_y

    def project_light_field_on_para_perp(self, c_para, r_para):
        source_cx, source_cy = self._source_direction_cx_cy(c_para=c_para)
        core_x, core_y = self._core_position_x_y(r_para=r_para)

        WRT_DOWNWARDS = -1.0
        c_para, c_perp = atg.projection.project_light_field_onto_source_image(
            cer_cx_rad=WRT_DOWNWARDS * self.cx,
            cer_cy_rad=WRT_DOWNWARDS * self.cy,
            cer_x_m=self.x,
            cer_y_m=self.y,
            primary_cx_rad=WRT_DOWNWARDS * source_cx,
            primary_cy_rad=WRT_DOWNWARDS * source_cy,
            primary_core_x_m=core_x,
            primary_core_y_m=core_y,
        )
        return c_para, c_perp

    def response(self, c_para, r_para, cer_perp_distance_threshold):
        cer_c_para, cer_c_perp = self.project_light_field_on_para_perp(
            c_para, r_para
        )

        num = len(cer_c_perp)

        l_trans_max = atg.model.lorentz_transversal(
            c_deg=0.0, peak_deg=0.0, width_deg=cer_perp_distance_threshold
        )
        l_trans = atg.model.lorentz_transversal(
            c_deg=cer_c_perp,
            peak_deg=0.0,
            width_deg=cer_perp_distance_threshold,
        )
        l_trans /= l_trans_max

        perp_weight = np.sum(l_trans) / num

        return perp_weight

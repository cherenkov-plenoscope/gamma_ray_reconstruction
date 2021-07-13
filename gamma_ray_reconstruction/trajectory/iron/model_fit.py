import numpy as np
from . import core_radius_search


class MainAxisToCoreFinder:
    def __init__(
        self,
        light_field_cx,
        light_field_cy,
        light_field_x,
        light_field_y,
        shower_maximum_cx,
        shower_maximum_cy,
        shower_maximum_object_distance,
        config,
    ):
        self.config = config
        self.shower_maximum_cx = shower_maximum_cx
        self.shower_maximum_cy = shower_maximum_cy
        self.shower_maximum_object_distance = shower_maximum_object_distance
        self.light_field_cx = light_field_cx
        self.light_field_cy = light_field_cy
        self.light_field_x = light_field_x
        self.light_field_y = light_field_y
        self.final_result = None

    def _support(self, main_axis_azimuth, main_axis_support_perp_offset):
        perp_azimuth_rad = main_axis_azimuth + 0.5 * np.pi
        offset_rad = main_axis_support_perp_offset
        cx = self.shower_maximum_cx + offset_rad * np.cos(perp_azimuth_rad)
        cy = self.shower_maximum_cy + offset_rad * np.sin(perp_azimuth_rad)
        return cx, cy

    def evaluate_shower_model(
        self, main_axis_azimuth, main_axis_support_perp_offset
    ):
        main_axis_support_cx, main_axis_support_cy = self._support(
            main_axis_azimuth=main_axis_azimuth,
            main_axis_support_perp_offset=main_axis_support_perp_offset,
        )

        result, _ = core_radius_search.estimate_core_radius_using_shower_model(
            main_axis_support_cx=main_axis_support_cx,
            main_axis_support_cy=main_axis_support_cy,
            main_axis_azimuth=main_axis_azimuth,
            light_field_cx=self.light_field_cx,
            light_field_cy=self.light_field_cy,
            light_field_x=self.light_field_x,
            light_field_y=self.light_field_y,
            shower_maximum_cx=self.shower_maximum_cx,
            shower_maximum_cy=self.shower_maximum_cy,
            shower_maximum_object_distance=self.shower_maximum_object_distance,
            config=self.config,
        )
        self.final_result = result
        self.final_result["main_axis_azimuth"] = main_axis_azimuth
        self.final_result["main_axis_support_cx"] = main_axis_support_cx
        self.final_result["main_axis_support_cy"] = main_axis_support_cy

        return 1.0 - self.final_result["shower_model_response"]

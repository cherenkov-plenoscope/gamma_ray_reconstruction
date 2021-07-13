import numpy as np
import airshower_template_generator as atg


def _source_direction_cx_cy(
    main_axis_azimuth,
    main_axis_support_cx,
    main_axis_support_cy,
    c_para
):
    source_cx = main_axis_support_cx + np.cos(main_axis_azimuth) * c_para
    source_cy = main_axis_support_cy + np.sin(main_axis_azimuth) * c_para
    return source_cx, source_cy


def _core_position_x_y(main_axis_azimuth, r_para):
    core_x = 0.0 + np.cos(main_axis_azimuth) * r_para
    core_y = 0.0 + np.sin(main_axis_azimuth) * r_para
    return core_x, core_y


def _project_light_field_on_para_perp(
    main_axis_azimuth,
    main_axis_support_cx,
    main_axis_support_cy,
    light_field_cx,
    light_field_cy,
    light_field_x,
    light_field_y,
    c_para,
    r_para,
):
    source_cx, source_cy = _source_direction_cx_cy(
        main_axis_azimuth=main_axis_azimuth,
        main_axis_support_cx=main_axis_support_cx,
        main_axis_support_c=main_axis_support_cy,
        c_para=c_para
    )
    core_x, core_y = _core_position_x_y(
        main_axis_azimuth=main_axis_azimuth,
        r_para=r_para
    )

    WRT_DOWNWARDS = -1.0
    c_para, c_perp = atg.projection.project_light_field_onto_source_image(
        cer_cx_rad=WRT_DOWNWARDS * light_field_cx,
        cer_cy_rad=WRT_DOWNWARDS * light_field_cy,
        cer_x_m=light_field_x,
        cer_y_m=light_field_y,
        primary_cx_rad=WRT_DOWNWARDS * source_cx,
        primary_cy_rad=WRT_DOWNWARDS * source_cy,
        primary_core_x_m=core_x,
        primary_core_y_m=core_y,
    )
    return c_para, c_perp


def response(
    main_axis_azimuth,
    main_axis_support_cx,
    main_axis_support_cy,
    light_field_cx,
    light_field_cy,
    light_field_x,
    light_field_y,
    c_para,
    r_para,
    cer_perp_distance_threshold,
):
    """
    Returns the matching response for a light-field to be the result of a
    specific primarie's trajectory.

    The model only consideres the transversal spread of
    Cherenkov-photons in image-space perpendicular to the shower-axis in
    the image-space.

    The primary particle's trajectory is defined by its main-axis projected
    into the image-space (azimuth, support_cx, support_cy), and by the
    source-angle c_para on this main-axis, as well as by the primarie's
    core-radius r_para in the direction of the main-axis.
    """
    cer_c_para, cer_c_perp = _project_light_field_on_para_perp(
        main_axis_azimuth=main_axis_azimuth,
        main_axis_support_cx=main_axis_support_cx,
        main_axis_support_cy=main_axis_support_cy,
        light_field_cx=light_field_cx,
        light_field_cy=light_field_cy,
        light_field_x=light_field_x,
        light_field_y=light_field_y,
        c_para=c_para,
        r_para=r_para,
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

def mask_border(image_tensor, border_fraction=0.08):
    masked_image = image_tensor.clone()

    _, height, width = masked_image.shape

    border_height = round(height * border_fraction)
    border_width = round(width * border_fraction)

    masked_image[:, :border_height, :] = 0
    masked_image[:, -border_height:, :] = 0
    masked_image[:, :, :border_width] = 0
    masked_image[:, :, -border_width:] = 0

    return masked_image


def mask_control_region(image_tensor, border_fraction=0.08):
    masked_image = image_tensor.clone()

    _, height, width = masked_image.shape

    border_area_fraction = (
        1 - (1 - 2 * border_fraction) ** 2
    )

    control_side_fraction = border_area_fraction ** 0.5

    control_height = round(height * control_side_fraction)
    control_width = round(width * control_side_fraction)

    top = (height - control_height) // 2
    left = (width - control_width) // 2

    masked_image[
        :,
        top:top + control_height,
        left:left + control_width
    ] = 0

    return masked_image
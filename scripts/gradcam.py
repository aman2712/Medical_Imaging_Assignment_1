import torch

def generate_gradcam(model, image_tensor, target_layer):
    activations = []
    gradients = []

    def save_activations(module, inputs, output):
        activations.append(output)

    def save_gradients(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    forward_handle = target_layer.register_forward_hook(save_activations)
    backward_handle = target_layer.register_full_backward_hook(save_gradients)

    model.zero_grad()

    image_batch = image_tensor.float().unsqueeze(0)
    logit = model(image_batch).squeeze()

    logit.backward()

    activation = activations[0].detach()
    gradient = gradients[0].detach()

    weights = gradient.mean(
        dim=(2, 3),
        keepdim=True
    )

    cam = (weights * activation).sum(dim=1).squeeze(0)
    cam = torch.relu(cam)

    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)

    cam = torch.nn.functional.interpolate(
        cam.unsqueeze(0).unsqueeze(0),
        size=image_tensor.shape[-2:],
        mode="bilinear",
        align_corners=False
    ).squeeze()

    forward_handle.remove()
    backward_handle.remove()

    return cam.numpy(), torch.sigmoid(logit).item()
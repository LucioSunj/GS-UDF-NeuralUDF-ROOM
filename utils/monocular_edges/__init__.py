import torch
import torch.nn.functional as F
def gradient_map(image):
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).float().unsqueeze(0).unsqueeze(0).cuda() / 4
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).float().unsqueeze(0).unsqueeze(0).cuda() / 4

    image = F.pad(image, (1, 1, 1, 1), 'reflect')

    grad_x = torch.cat([F.conv2d(image[i].unsqueeze(0), sobel_x, padding=0) for i in range(image.shape[0])])
    grad_y = torch.cat([F.conv2d(image[i].unsqueeze(0), sobel_y, padding=0) for i in range(image.shape[0])])
    edge = torch.sqrt(grad_x ** 2 + grad_y ** 2)
    edge = edge.norm(dim=0, keepdim=True)

    return edge


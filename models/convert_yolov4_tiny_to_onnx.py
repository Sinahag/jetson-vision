import torch
import torch.nn as nn
import torchvision.transforms as transforms
import argparse

def create_model(cfgfile, num_classes=80):
    # Example of a simple YOLOv4 model in PyTorch
    class YOLOv4(nn.Module):
        def __init__(self, cfgfile, num_classes=80):
            super(YOLOv4, self).__init__()
            # Define layers according to your YOLOv4 configuration
            # This is a simplified example, adjust according to your cfgfile
            # Example layers (you need to implement based on yolov4.cfg)
            self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
            self.fc1 = nn.Linear(32 * 56 * 56, 256)
            self.fc2 = nn.Linear(256, num_classes)

        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = self.pool(x)
            x = torch.relu(self.conv2(x))
            x = x.view(-1, 32 * 56 * 56)  # Flatten
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    model = YOLOv4(cfgfile, num_classes=num_classes)
    return model

def export_onnx(cfgfile, weightfile, output_onnx):
    # Create the model
    model = create_model(cfgfile)

    # Load weights
    model.load_state_dict(torch.load(weightfile))

    # Set the model to evaluation mode
    model.eval()

    # Example input (adjust according to your YOLOv4 configuration)
    input_shape = (3, 416, 416)
    dummy_input = torch.randn(1, *input_shape)

    # Export to ONNX
    torch.onnx.export(model, dummy_input, output_onnx, verbose=True, opset_version=11,
                      input_names=['input'], output_names=['output'])

    print(f"Model successfully exported to {output_onnx}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert YOLOv4 to ONNX format')
    parser.add_argument('--cfg', type=str, help='YOLOv4 configuration file (.cfg)')
    parser.add_argument('--weights', type=str, help='YOLOv4 weights file (.weights)')
    parser.add_argument('--output', type=str, default='yolov4.onnx', help='Output ONNX file path')
    args = parser.parse_args()

    export_onnx(args.cfg, args.weights, args.output)


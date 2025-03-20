import torch
import torchvision.transforms as transforms
from PIL import Image


def print_examples(model, device, dataset):
    transform = transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    model.eval()
    test_img1 = transform(Image.open("test_examples/cat.jpg").convert("RGB")).unsqueeze(
        0
    )
    print("CORRECT: A cat lying on the seil")
    print(
        "OUTPUT: "
        + " ".join(model.generate_caption(test_img1.to(device), dataset.vocab))
    )
    model.train()


def save_checkpoint(state, filename="my_checkpoint2.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)




def load_checkpoint(filename, model, optimizer):
    print(f"=> Loading checkpoint: {filename}")
    checkpoint = torch.load(filename, map_location="cpu")

    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    step = checkpoint.get("step", 0)
    print(f"✅ Loaded checkpoint (Step {step})")
    return step

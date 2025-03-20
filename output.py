import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from model import CNNtoRNN
from get_loader import Vocabulary

def load_model(checkpoint_path, embed_size, hidden_size, vocab_size, num_layers, device):
    model = CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    print(f"✅ Loaded checkpoint: {checkpoint_path}")
    return model

def predict_caption(model, image_path, vocabulary, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    caption = model.generate_caption(image, vocabulary)
    return " ".join(caption)

def show_image_with_caption(image_path, caption):
    image = Image.open(image_path)
    plt.imshow(image)
    plt.axis("off")
    plt.title(caption, fontsize=14)
    plt.show()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    embed_size = 256
    hidden_size = 256
    vocab_size = 2996
    num_layers = 1

    vocabulary = Vocabulary(freq_threshold=5)
    vocabulary.stoi = torch.load("vocab_stoi.pth")
    vocabulary.itos = {idx: word for word, idx in vocabulary.stoi.items()}

    checkpoint_path = "my_checkpoint2.pth.tar"
    model = load_model(checkpoint_path, embed_size, hidden_size, vocab_size, num_layers, device)

    test_image_path = "test_examples/family.jpg"

    caption = predict_caption(model, test_image_path, vocabulary, device)
    print(caption)

    show_image_with_caption(test_image_path, caption)

from .network import CNN


net = CNN()
net.training('./save/', save_path='./models', epochs=10)

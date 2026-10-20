

import torch
import torch.nn as nn
import torch.nn.functional as F


class MiniBrain(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))   # gizli katman 1
        x = F.relu(self.fc2(x))   # gizli katman 2
        x = self.fc3(x)           # çıkış katmanı (aktivasyon yok)
        return x

if __name__ == "__main__":
    batch_size, input_size, output_size = 32, 20, 10

    # Model ve mimari özeti
    model = MiniBrain(input_size, output_size)
    print(model)
    print("Toplam parametre:", sum(p.numel() for p in model.parameters()))

    # Örnek girdi-çıktı testi
    x = torch.randn(batch_size, input_size)
    y = model(x)
    print("\nGirdi boyutu :", x.shape)
    print("Çıktı boyutu :", y.shape)



# DEĞERLENDİRME: forward metodunun önemi

# __init__ modelin hangi katmanlara sahip olduğunu (parçaları), forward ise bu
# parçaların hangi sırayla ve nasıl bağlanacağını (veri akışını) belirler.
# Katmanlar tanımlı olsa bile forward yazılmadan model çalışmaz.
# forward çalışırken PyTorch yapılan her işlemi kaydeder (hesaplama grafiği);
# loss.backward() bu kayıt üzerinden türevleri hesaplar. Yani modelin hem tahmin
# yapması hem de öğrenebilmesi forward'a bağlıdır. ReLU gibi aktivasyonların
# nereye konacağı da burada seçilir; ReLU olmasaydı üç Linear katman tek bir
# Linear katmana eşdeğer olurdu ve model doğrusal olmayan ilişkileri öğrenemezdi.
# Modeli çağırırken model.forward(x) yerine model(x) yazılır; bu, forward'ı
# PyTorch'un ek mekanizmalarıyla (hook'lar vb.) birlikte çalıştırır.

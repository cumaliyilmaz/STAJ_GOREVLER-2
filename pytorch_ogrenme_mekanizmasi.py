import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# 1) Sentetik veri
X = torch.randn(100, 1)
Y = 3 * X + 2 + torch.randn(100, 1) * 0.1

# 2) Model, kayıp, optimizer (+ eğitim öncesi tahmin)
model = nn.Linear(1, 1)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

with torch.no_grad():
    tahmin_once = model(X)

# 3) Eğitim döngüsü
for epoch in range(1000):
    loss = criterion(model(X), Y)   # forward pass
    optimizer.zero_grad()           # eski gradyanları sıfırla
    loss.backward()                 # backward pass
    optimizer.step()                # parametre güncelleme

# 4) Sonuç ve grafik
print("weight:", model.weight.item(), "(hedef ~3)")
print("bias  :", model.bias.item(), "(hedef ~2)")

with torch.no_grad():
    tahmin_sonra = model(X)

plt.scatter(X, Y, label="Veri")
plt.scatter(X, tahmin_once, label="Eğitim öncesi")
plt.scatter(X, tahmin_sonra, label="Eğitim sonrası")
plt.legend()
plt.show()





for X, target in train_labeled_loader:
    break

X = X * 0.3081 + 0.1307

X.resize_(train_batch_size, X_dim)
X, target = Variable(X), Variable(target)

if cuda:
    X, target = X.cuda(), target.cuda()

_, z_gauss = Q(X)

z_cat = np.arange(0, 10)
z_cat = np.eye(n_classes)[z_cat].astype('float32')
z_cat = torch.from_numpy(z_cat)
z_cat = Variable(z_cat)
if cuda:
    z_cat = z_cat.cuda()

z_gauss = z_gauss[0].resize(1,z_dim)
z_gauss0 = z_gauss[0].resize(1, z_dim)

for i in range(9):
    z_gauss = torch.cat((z_gauss, z_gauss0), 0)
    
z = torch.cat((z_cat, z_gauss ), 1)

x = P(z)

img = np.array(x[0].data.tolist()).reshape(28,28)

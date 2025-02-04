import cv2
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve2d
from scipy import interpolate

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import numpy as np

tensor_precision = np.float32

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1.0

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        y_pred = y_pred[:, 0].contiguous().view(-1)
        y_true = y_true[:, 0].contiguous().view(-1)
        intersection = (y_pred * y_true).sum()
        dsc = (2. * intersection + self.smooth) / (
            y_pred.sum() + y_true.sum() + self.smooth
        )
        return 1. - dsc

class atrous(nn.Module): # this is the lateral geniculate nucleus :)
    def __init__(self, in_channels, features):
        super().__init__()
        assert (features % 4) == 0
        self.conv1 = atrous.filter(in_channels, int(features/4), 1, 1)
        self.conv2 = atrous.filter(in_channels, int(features/4), 3, 1)
        self.conv3 = atrous.filter(in_channels, int(features/4), 3, 2)
        self.conv4 = atrous.filter(in_channels, int(features/4), 3, 3)
        self.norm = nn.Sequential(nn.BatchNorm2d(num_features=features), nn.ReLU(inplace=True))

    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2(x)
        c3 = self.conv3(x)
        c4 = self.conv4(x)
        c = torch.cat((c1, c2, c3, c4), dim=1)
        return self.norm(c)

    @staticmethod
    def filter(in_channels, features, kernel, dilation):
        return nn.Conv2d(
            in_channels=in_channels,
            out_channels=features,
            kernel_size=kernel,
            stride=1,
            padding=int(kernel != 1) * dilation,
            dilation=dilation,
            groups=in_channels,
            bias=False,
        )

class UNet(nn.Module):
    def __init__(self, in_channels=2, features=32):
        super().__init__()
        # self.enc1 = UNet.block(in_channels, features)

        self.enc1 = nn.Sequential(
            atrous(in_channels=in_channels, features=features),
            UNet.block(features, features)
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc2 = UNet.block(features, features * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc3 = UNet.block(features * 2, features * 4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet.block(features * 4, features * 8)

        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.dec3 = UNet.block(features * 8, features * 4)
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.dec2 = UNet.block(features * 4, features * 2)
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.dec1 = UNet.block(features * 2, features)

        self.conv = nn.Conv2d(in_channels=features, out_channels=1, kernel_size=1)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        bottleneck = self.bottleneck(self.pool3(enc3))
        dec3 = self.upconv3(bottleneck)
        dec3 = self.dec3(torch.cat((dec3, enc3), dim=1))
        dec2 = self.upconv2(dec3)
        dec2 = self.dec2(torch.cat((dec2, enc2), dim=1))
        dec1 = self.upconv1(dec2)
        dec1 = self.dec1(torch.cat((dec1, enc1), dim=1))
        return torch.sigmoid(self.conv(dec1))

    @staticmethod
    def block(in_channels, features):
        stack = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=features,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(num_features=features),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=features,
                out_channels=features,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(num_features=features),
            nn.ReLU(inplace=True),
        )
        return stack

class ROIdata(Dataset):
    def __init__(self, stack):
        stack = _normstack(stack)
        idx = int(np.floor(stack.shape[0] / 3))
        np.random.shuffle(stack) # some data augmentation
        stack = np.concatenate((stack, np.flip(stack[:idx, :, :, :], axis=2)), axis=0) # flip data augmentation
        stack = np.concatenate((stack, np.flip(stack[idx:idx*2, :, :, :], axis=3)), axis=0) # flip data augmentation
        stack = np.concatenate((stack, np.rot90(stack[idx*2:, :, :, :], axes=(2, 3))), axis=0) # rotate data augmentation
        self.stack = torch.from_numpy(stack.astype(tensor_precision))

    def __len__(self):
        return self.stack.shape[0]

    def __getitem__(self, idx):
        img = self.stack[idx, :2, :, :]
        mask = self.stack[idx, 2, :, :]
        return img, mask[None, :, :]

def _normstack(stack):
    stack = stack.copy()
    while stack.ndim < 4:
        stack = np.expand_dims(stack, axis=0)
    idx = stack == 0
    stack = (stack - np.min(stack, axis=(2, 3), keepdims=True)) / np.ptp(stack, axis=(2, 3), keepdims=True)
    stack[np.isnan(stack)] = 0
    stack[idx] = 0
    return stack

def _train(md, dataloader, lr=1e-3):
    optimizer = torch.optim.Adam(md.parameters(), lr=lr)
    lossfunc = DiceLoss()
    device = torch.cuda.current_device()

    size = len(dataloader.dataset)
    md.train()

    for batch, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)

        pred = md(x)
        loss = lossfunc(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    loss, current = loss.item(), (batch + 1) * len(x)
    return loss

def _test(md, dataloader):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    lossfunc = DiceLoss()
    device = torch.cuda.current_device()
    md.eval()
    loss = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = md(x)
            loss += lossfunc(pred, y).item()
    loss /= num_batches
    return loss

def train(data_path, epochs=500, batch_size=64, holdout=.1):
    data = [os.path.join(data_path, file) for file in os.listdir(data_path) if file.endswith('.npy')]
    stack = [np.load(d, allow_pickle=True) for d in data]
    stack = np.concatenate(stack, axis=0)

    data = ROIdata(stack)
    train_data, test_data = torch.utils.data.random_split(data, (1 - holdout, holdout), torch.Generator())
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    model = UNet().to(torch.cuda.current_device())

    train_loss = []
    test_loss = []
    print('----- Training LGN UNet model -----')
    for t in tqdm(range(epochs)):
        train_loss.append(_train(model, train_dataloader))
        test_loss.append(_test(model, test_dataloader))
    print("Done!")

    return model, train_loss, test_loss


def detect(masks, mimg, model_path, diameter=None, dilation=3):
    """
    The masks coming out of Cellpose look like shit (too dilated, no clear edges).
    This method implements a custom Canny edge detection algorithm to refine the
    ROI masks. Roughly, it works as follow (see wikipedia article on Canny):
        1. Conduct a top-hat morphological transform on the mean image,
        2. Gaussian smooth that bitch with (3, 3) kernel,
        3. Compute the edge gradient and direction by Sobel filter,
        4. For each ROI, apply a 75th percentile threshold on the gradient to
        find the edges,
        5. Thin the edges by lower bound thresholding,
        6. Use a custom UNet model to refine the edges,
        7. Detect and fill in the contours,
        8. Et voila!

    Parameters
    ----------
    masks (array)
        straight out of roi_detect()
    mimg (array)
        mean stack image (or max projection) without registration boundaries
    diameter (int / float)
        Cellpose extimated diameter of the ROIs
    dilation (int)
        number of pixels to extend past the borders of the Cellpose ROI for
        gradient computation
    model_path (str)
        path to pth model (e.g. /home/loulou/model.pth)

    Returns
    -------
    refined_masks (array)
    """
    diameter = np.round(diameter).astype(int)
    # top-hat transform mean stack
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, ) * 2)
    mimg = cv2.morphologyEx(mimg, cv2.MORPH_TOPHAT, kernel)

    # smooth that bitch
    smooth = cv2.GaussianBlur(mimg, (3, 3), sigmaX=0, sigmaY=0)
    # apply Sobel filter and compute gradient magnitudes
    x = cv2.Sobel(src=smooth, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=3)
    y = cv2.Sobel(src=smooth, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=3)
    gradient = np.sqrt(x**2 + y**2)
    direction = np.arctan2(y, x)

    # load UNet model
    model = UNet().to(torch.cuda.current_device())
    model.load_state_dict(torch.load(model_path, weights_only=True))
    device = torch.cuda.current_device()
    model.eval()

    refined_masks = np.zeros_like(masks)
    counter = 1
    sz = 64
    # Canny edge detection for individual ROIs
    for n in range(1, np.max(masks) + 1):
        # obtain local gradient and direction for ROI
        ry, rx = np.nonzero(masks == n)
        ry = [np.min(ry), np.max(ry)]
        rx = [np.min(rx), np.max(rx)]
        g = gradient[np.ix_(np.arange(np.max([ry[0] - dilation, 0]), np.min([ry[1] + dilation + 1, masks.shape[0]])), \
                            np.arange(np.max([rx[0] - dilation, 0]), np.min([rx[1] + dilation + 1, masks.shape[1]])))]
        d = direction[np.ix_(np.arange(np.max([ry[0] - dilation, 0]), np.min([ry[1] + dilation + 1, masks.shape[0]])), \
                            np.arange(np.max([rx[0] - dilation, 0]), np.min([rx[1] + dilation + 1, masks.shape[1]])))]
        m = masks[np.ix_(np.arange(np.max([ry[0] - dilation, 0]), np.min([ry[1] + dilation + 1, masks.shape[0]])), \
                            np.arange(np.max([rx[0] - dilation, 0]), np.min([rx[1] + dilation + 1, masks.shape[1]])))] == n
        a = mimg[np.ix_(np.arange(np.max([ry[0] - dilation, 0]), np.min([ry[1] + dilation + 1, masks.shape[0]])), \
                            np.arange(np.max([rx[0] - dilation, 0]), np.min([rx[1] + dilation + 1, masks.shape[1]])))]

        # thresholding to find edges
        prct, _ = cv2.threshold(g.astype(np.uint16), np.min(g), np.max(g), cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        d[d < 0] = np.pi + d[d < 0]
        d = np.digitize(d, [np.pi, np.pi * 7 / 8, np.pi * 5 / 8, np.pi * 3 / 8, np.pi / 8, 0])
        d[d == 5] = 1
        thinned = (g > prct / 2) & m
        d = d * thinned

        # edge thinning by lower bound thresholding
        comp = np.zeros((2, d.shape[0], d.shape[1]))
        comp[0, :, :] += np.roll(np.roll(d == 1, shift=1, axis=1) * g, shift=-1, axis=1)
        comp[1, :, :] += np.roll(np.roll(d == 1, shift=-1, axis=1) * g, shift=1, axis=1)
        comp[0, :, :] += np.roll(np.roll(d == 2, shift=(1, -1), axis=(0, 1)) * g, shift=(-1, 1), axis=(0, 1))
        comp[1, :, :] += np.roll(np.roll(d == 2, shift=(-1, 1), axis=(0, 1)) * g, shift=(1, -1), axis=(0, 1))
        comp[0, :, :] += np.roll(np.roll(d == 3, shift=1, axis=0) * g, shift=-1, axis=0)
        comp[1, :, :] += np.roll(np.roll(d == 3, shift=-1, axis=0) * g, shift=1, axis=0)
        comp[0, :, :] += np.roll(np.roll(d == 4, shift=(1, 1), axis=(0, 1)) * g, shift=(-1, -1), axis=(0, 1))
        comp[1, :, :] += np.roll(np.roll(d == 4, shift=(-1, -1), axis=(0, 1)) * g, shift=(1, 1), axis=(0, 1))

        thinned = np.all(thinned & (g > comp), axis=0).astype(np.uint8)

        # edge tracking by hysteresis
        strong = g * thinned > prct
        weak = g * thinned > prct / 2
        weak ^= strong
        previous = weak
        kernel = np.ones((2, 2))
        while True:
            neighbours = convolve2d(strong, kernel, mode='same').astype(bool)
            strong |= weak & neighbours
            previous = weak
            weak &= ~neighbours
            if np.all(weak == previous):
                break
        thinned &= m

        if np.sum(thinned) == 0:
            continue

        if a.shape[0] > sz:
            a = a[np.floor((a.shape[0] - sz) / 2).astype(int) : np.floor((a.shape[0] - sz) / 2).astype(int) + sz, :]
            thinned = thinned[np.floor((a.shape[0] - sz) / 2).astype(int) : np.floor((a.shape[0] - sz) / 2).astype(int) + sz, :]
            m = m[np.floor((a.shape[0] - sz) / 2).astype(int) : np.floor((a.shape[0] - sz) / 2).astype(int) + sz, :]
        if a.shape[1] > sz:
            a = a[:, np.floor((a.shape[1] - sz) / 2).astype(int) : np.floor((a.shape[1] - sz) / 2).astype(int) + sz]
            thinned = thinned[:, np.floor((a.shape[1] - sz) / 2).astype(int) : np.floor((a.shape[1] - sz) / 2).astype(int) + sz]
            m = m[:, np.floor((a.shape[1] - sz) / 2).astype(int) : np.floor((a.shape[1] - sz) / 2).astype(int) + sz]

        a = np.pad(a, ((np.floor((sz - a.shape[0]) / 2).astype(int), np.ceil((sz - a.shape[0]) / 2).astype(int)), \
                       (np.floor((sz - a.shape[1]) / 2).astype(int), np.ceil((sz - a.shape[1]) / 2).astype(int))))
        thinned = np.pad(thinned, ((np.floor((sz - thinned.shape[0]) / 2).astype(int), np.ceil((sz - thinned.shape[0]) / 2).astype(int)), \
                       (np.floor((sz - thinned.shape[1]) / 2).astype(int), np.ceil((sz - thinned.shape[1]) / 2).astype(int))))
        stack = _normstack(np.array([a, thinned]))
        stack = torch.from_numpy(stack.astype(tensor_precision)).to(device)
        with torch.no_grad():
            refined = model(stack)
        refined = np.round(refined.cpu().detach().numpy()[0, 0, :, :]).astype(bool)
        refined = refined[np.floor((sz - m.shape[0]) / 2).astype(int) : np.floor((sz - m.shape[0]) / 2).astype(int) + m.shape[0], \
                            np.floor((sz - m.shape[1]) / 2).astype(int) : np.floor((sz - m.shape[1]) / 2).astype(int) + m.shape[1]]

        refined &= m
        if np.sum(refined) == 0:
            continue

        if g.shape[0] > sz or g.shape[1] > sz:
            refined = np.pad(refined, ((np.floor((g.shape[0] - a.shape[0]) / 2).astype(int), np.ceil((g.shape[0] - a.shape[0]) / 2).astype(int)), \
                       (np.floor((g.shape[1] - a.shape[1]) / 2).astype(int), np.ceil((g.shape[1] - a.shape[1]) / 2).astype(int))))

        refined_masks[np.ix_(np.arange(np.max([ry[0] - dilation, 0]), np.min([ry[1] + dilation + 1, masks.shape[0]])), \
                            np.arange(np.max([rx[0] - dilation, 0]), np.min([rx[1] + dilation + 1, masks.shape[1]])))] \
                            += refined * counter
        counter += 1

    print('Finished Uncanny edge refinement. ' + str(counter - 1) + ' ROIs remaining out of ' + \
          str(np.max(masks).astype(int)) + ' original masks.')

    return refined_masks
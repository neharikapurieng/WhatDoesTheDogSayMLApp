# To test loading and evaluating inputs
# and also test processing

import sys
sys.path.insert(1, 'script')
from model import EfficientNetNotRGB
from AudioUtil import AudioUtil
import torchvision
import torch
import numpy as np

if __name__=='__main__':
    # Initialize variables
    num_classes = 9
    model_name = 'dog_audio_effB1_ep150/dog_audio_effB1_ep150val_acc_0.7142857142857143val_loss_0.10887178033590317'
    duration = 5000 # 5 sec
    sr = 44100
    channel = 2
    shift_pct = 0.4
    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    
    # create model object
    deepNet = EfficientNetNotRGB.from_pretrained('efficientnet-b1',num_classes=num_classes)
    
    # load generator
    states = torch.load('../models/'+model_name)
    deepNet.load_state_dict(states['model_state_dict'])
    deepNet.eval()
    # process image
    aud = AudioUtil.open('../data/dog/c/dog_33c.wav')
    reaud = AudioUtil.resample(aud, sr)
    rechan = AudioUtil.rechannel(reaud, channel)
    dur_aud = AudioUtil.pad_trunc(rechan, duration)
    shift_aud = AudioUtil.time_shift(dur_aud, shift_pct)
    sgram = AudioUtil.spectro_gram(shift_aud, n_mels=64, n_fft=1024, hop_len=None)
    aug_sgram = AudioUtil.spectro_augment(sgram, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2)
    use_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    aug_sgram = use_transform(np.array(aug_sgram))
    aug_sgram = torch.reshape(aug_sgram, (1,aug_sgram.shape[1],aug_sgram.shape[0],aug_sgram.shape[2]))
    print(aug_sgram.shape)
    #aug_sgram = aug_sgram.to(device, dtype=torch.float)
    output = deepNet(aug_sgram)
    _, preds = torch.max(output, 1)
    print(str(int(preds[0])))
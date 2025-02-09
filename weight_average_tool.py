import torch
import os
import numpy as np
import scipy.stats as stats
ckpt = '/data3/yuyating/CKPT'
raw_clip = '/home/yuyating/.cache/clip/ViT-B-16.pt'
source_dir = ckpt + '/STDD/vitb16_8x16/checkpoints'
output_dir = ckpt + '/STDD/vitb16_8x16/wa_checkpoints'
ensemble = ['swa']

wa_start = 2
wa_end = 22

def get_gauss(mu, sigma):
    gauss = lambda x: 1/(sigma * (2 * np.pi)**0.5) * np.exp(-0.5 * ((x - mu) / sigma)**2)
    return gauss


def average_checkpoint(checkpoint_list, ensemble=None):
    ckpt_list = []

    # raw clip
    raw_clip_weight = {}
    clip_ori_state = torch.jit.load(raw_clip, map_location='cpu').state_dict()
    _ = [clip_ori_state.pop(i) for i in ['input_resolution', 'context_length', 'vocab_size']]
    for key in clip_ori_state:
        raw_clip_weight['model.' + key] = clip_ori_state[key]

    ckpt_list.append((0, raw_clip_weight))
    for name, ckpt_id in checkpoint_list:
        ckpt_list.append((ckpt_id, torch.load(name, map_location='cpu')['model_state']))

    # threshold filter
    new_ckpt_list = []
    ckpt_id_list = []
    for i in ckpt_list:
        if int(i[0]) >= wa_start and int(i[0]) <= wa_end:
            new_ckpt_list.append(i)
            ckpt_id_list.append(int(i[0]))

    print("Files with the following paths will participate in the parameter averaging")
    print(ckpt_id_list)

    state_dict = {}
    for key in raw_clip_weight:
        state_dict[key] = []
        for ckpt in new_ckpt_list:
            state_dict[key].append(ckpt[1][key])
    swa_state_dict = {}
    if 'swa' in ensemble:

        for key in state_dict:
            try:
                swa_state_dict[key] = torch.mean(torch.stack(state_dict[key], 0), 0)
            except:
                print(key)


    gwa_state_dict = {}
    if 'gwa' in ensemble:
        mu, sigma = 15, 10
        gauss = get_gauss(mu, sigma)
        gauss = np.array([gauss(i) for i in range(wa_start, wa_end + 1)])
        ckpt_weight = gauss / gauss.sum()

        for key in state_dict:
            try:
                for i, ckpt in enumerate(state_dict[key]):
                    if i == 0:
                        gwa_state_dict[key] = ckpt * ckpt_weight[i]
                    else:
                        gwa_state_dict[key] += ckpt * ckpt_weight[i]
            except:
                print(key)

    bwa_state_dict = {}
    if 'bwa' in ensemble:
        a, b = 0.5, 0.5
        x = np.arange(0, wa_end + 1 -wa_start)
        x = (x+0.5)/(wa_end+1-wa_start)
        beta = stats.beta.pdf(x, a, b)
        ckpt_weight = beta / beta.sum()
        for key in state_dict:
            try:
                for i, ckpt in enumerate(state_dict[key]):
                    if i == 0:
                        bwa_state_dict[key] = ckpt * ckpt_weight[i]
                    else:
                        bwa_state_dict[key] += ckpt * ckpt_weight[i]
            except:
                print(key)

    # exponential weighted average
    ewa_state_dict={}
    if 'ewa' in ensemble:
        decay = 0.9
        for key in state_dict:
            try:
                for i, ckpt in enumerate(state_dict[key]):
                    if i==0:
                        ewa_state_dict[key] = ckpt
                    else:
                        ewa_state_dict[key] = decay * ewa_state_dict[key] + (1-decay) * ckpt
            except:
                print


    return swa_state_dict, gwa_state_dict, bwa_state_dict, ewa_state_dict


os.makedirs(output_dir, exist_ok=True)
checkpoint_list = os.listdir(source_dir)
checkpoint_list = [(os.path.join(source_dir, i), int(i.split('.')[0].split('_')[-1])) for i in checkpoint_list]
checkpoint_list = sorted(checkpoint_list, key=lambda d: d[1])

swa_state_dict, gwa_state_dict, bwa_state_dict, ewa_state_dict = average_checkpoint(checkpoint_list, ensemble = ensemble)
if swa_state_dict:
    torch.save({'model_state': swa_state_dict}, os.path.join(output_dir, 'swa_%d_%d.pth'%(wa_start, wa_end)))
    print('swa saved')
if gwa_state_dict:
    torch.save({'model_state': gwa_state_dict}, os.path.join(output_dir, 'gwa_%d_%d.pth'%(wa_start, wa_end)))
    print('gwa saved')
if bwa_state_dict:
    torch.save({'model_state': bwa_state_dict}, os.path.join(output_dir, 'bwa_%d_%d.pth'%(wa_start, wa_end)))
    print('bwa saved')
if ewa_state_dict:
    torch.save({'model_state': ewa_state_dict}, os.path.join(output_dir, 'ewa_%d_%d.pth'%(wa_start, wa_end)))
    print('ewa saved')
# torch.save({'model_state': swa_state_dict}, os.path.join(output_dir, 'swa_%d_%d.pth'%(wa_start, wa_end)))


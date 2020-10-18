import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def weightedLoss(batch, outputs, weight):
    loss = 0
    center_loss, dis_loss, line_loss = 0, 0, 0
    ce_loss = nn.BCELoss(reduce=False)
    sl1_loss = nn.SmoothL1Loss(reduce=False)
    stack = len(outputs)
    # B = batch['kp_mask'].shape[0]
    K_num = batch['kp_mask'].sum(-1).sum(-1).sum(-1)
    for s in range(stack):
        output = outputs[s]
        pred_center = output['center']
        pred_line = output['line']
        pred_dis = output['dis']
        kp_mask = batch['kp_mask']
        center_mask = batch['center_mask']
        line_mask = batch['line_mask']

        center_t = ce_loss(pred_center, batch['center'])
        center_t *= center_mask
        center_loss += center_t.mean() / (stack)

        line_t = ce_loss(pred_line, batch['line'])
        line_t *= line_mask
        line_loss += line_t.mean() / (stack)

        dis_t = sl1_loss(pred_dis * kp_mask, batch['kp_displacement'])
        dis_loss = (dis_t.sum(-1).sum(-1).sum(-1) / K_num).mean() / (stack) / 4.

    loss += weight['center'] * center_loss + weight['dis'] * dis_loss + weight['line'] * line_loss
    loss_stats = {'loss': loss, 'center': weight['center'] * center_loss,
                  'dis': weight['dis'] * dis_loss, 'line': weight['line'] * line_loss}
    return loss, loss_stats

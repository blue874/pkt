import torch
import torch.nn as nn
import torch.nn.functional as F

def pdist(e, squared=False, eps=1e-12):
    e_square = e.pow(2).sum(dim=1)
    prod = e @ e.t()
    res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)

    if not squared:
        res = res.sqrt()

    res = res.clone()
    res[range(len(e)), range(len(e))] = 0
    return res

def Hadamardloss(ft_student,ft_teacher):
    dot_product=(ft_student*ft_teacher).sum(dim=1)
    loss=dot_product.sum()
    return loss

def LogitsDiv(logit_a,logit_b,logit_c):
    #a/b*c,
    logit_b=torch.where(logit_b == 0, torch.tensor(1e-8), logit_b)#去除可能出现的0导致的无穷大
    res_div=logit_a/logit_b
    logit=logit_c*res_div
    return logit

#KL散度
def KLloss(logit_s,logit_t,t):
    loss=  F.kl_div(
                    F.log_softmax(logit_s / t, dim=1),  # 学生模型的蒸馏输出
                    F.log_softmax(logit_t/ t, dim=1),  # 教师模型的输出
                    reduction='sum',  # 损失计算方式
                    log_target=True  # 使用log_target=True
                    ) * (t*t) / logit_s.numel()
    return loss
    
#rkd二元组距离，欧几里得距离
class RkdDistance(nn.Module):
    def forward(self, student, teacher):
        with torch.no_grad():
            t_d = pdist(teacher, squared=False)
            mean_td = t_d[t_d>0].mean()
            t_d = t_d / mean_td

        d = pdist(student, squared=False)
        mean_d = d[d>0].mean()
        d = d / mean_d

        loss = F.smooth_l1_loss(d, t_d, reduction='mean')
        return loss
#rkd三元组距离，角度损失。   
class RKdAngle(nn.Module):
    def forward(self, student, teacher):
        # N x C
        # N x N x C

        with torch.no_grad():
            td = (teacher.unsqueeze(0) - teacher.unsqueeze(1))
            norm_td = F.normalize(td, p=2, dim=2)
            t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)

        sd = (student.unsqueeze(0) - student.unsqueeze(1))
        norm_sd = F.normalize(sd, p=2, dim=2)
        s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)

        loss = F.smooth_l1_loss(s_angle, t_angle, reduction='elementwise_mean')
        return loss

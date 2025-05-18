import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
# 完全改成了CADP的self，但是实际上也没有改什么本质上的东西
# 实际上就是加了个self.dot和self.values
class SelfAttention(nn.Module):
    def __init__(self, input_size, heads, embed_size):
        super().__init__()
        self.input_size = input_size
        self.heads = heads
        self.emb_size = embed_size

        self.tokeys = nn.Linear(self.input_size, self.emb_size * heads, bias = False)
        self.toqueries = nn.Linear(self.input_size, self.emb_size * heads, bias = False)
        self.tovalues = nn.Linear(self.input_size, self.emb_size * heads, bias = False)

    def forward(self, x):
        b, t, hin = x.size()
        assert hin == self.input_size, 'Input size {} should match {}'.format(hin, self.input_size)
        
        h = self.heads 
        e = self.emb_size
        
        keys = self.tokeys(x).view(b, t, h, e)
        queries = self.toqueries(x).view(b, t, h, e)
        values = self.tovalues(x).view(b, t, h, e)

        # dot-product attention
        # folding heads to batch dimensions
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, e)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, e)
        values = values.transpose(1, 2).contiguous().view(b * h, t, e)

        queries = queries / (e ** (1/4))
        keys = keys / (e ** (1/4))
        # 计算两个tensor的矩阵乘法，torch.bmm(a,b),tensor a 的size为(b,h,w),tensor b的size为(b,w,m) 
        # 也就是说两个tensor的第一维是相等的，然后第一个数组的第三维和第二个数组的第二维度要求一样，对于剩下的则不做要求，输出维度 （b,h,m)
        # 只能做三维乘法，矩阵是二维或者四维都会报错
        dot = torch.bmm(queries, keys.transpose(1, 2))
        assert dot.size() == (b*h, t, t)

        # row wise self attention probabilities
        dot = F.softmax(dot, dim=2)
        # dot就是权重c
        self.dot = dot
        out = torch.bmm(dot, values).view(b, h, t, e)
        out = out.transpose(1, 2).contiguous().view(b, t, h * e)
        values = values.view(b, h, t, e)
        values = values.transpose(1, 2).contiguous().view(b, t, h * e)
        self.values = values
        return out



class Merger(nn.Module):
    def __init__(self, head, fea_dim):
        super(Merger, self).__init__()
        self.head = head
        if head > 1:
            self.weight = Parameter(torch.Tensor(1, head, fea_dim).fill_(1.))
            self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        """
        :param x: [bs, n_head, fea_dim]
        :return: [bs, fea_dim]
        """
        if self.head > 1:
            return torch.sum(self.softmax(self.weight) * x, dim=1, keepdim=False)
        else:
            return torch.squeeze(x, dim=1)

class Observe_State_Attention(nn.Module):
    def __init__(self, observe_size, state_size, heads, embed_size):
        super().__init__()
        self.observe_size = observe_size
        self.state_size = state_size
        self.heads = heads
        self.emb_size = embed_size
        # Q是输入信息，K是待匹配的信息，V是信息本身，V只是单纯表达了输入特征的信息
        self.toqueries = nn.Linear(self.observe_size, self.emb_size * heads, bias = False)
        self.tokeys = nn.Linear(self.state_size, self.emb_size * heads, bias = False)
        self.tovalues = nn.Linear(self.state_size, self.emb_size * heads, bias = False)
        self.unify_all_heads = Merger(heads, self.emb_size)
    def forward(self, observe, state):
        b, hin = observe.size()
        assert hin == self.observe_size, 'Input size {} should match {}'.format(hin, self.input_size)
        
        h = self.heads 
        e = self.emb_size
        
        # if next(self.toqueries.parameters()).device != observe.device:
        #     observe = observe.to(next(self.toqueries.parameters()).device)
        #     state = state.to(next(self.tokeys.parameters()).device)

        queries = self.toqueries(observe).view(b, h, e).reshape(b*h,1,-1)
        keys = self.tokeys(state).view(b, h, e).reshape(b*h,1,-1)
        values = self.tovalues(state).view(b, h, e).reshape(b*h,1,-1)

        # dot-product attention
        # folding heads to batch dimensions

        # 计算两个tensor的矩阵乘法，torch.bmm(a,b),tensor a 的size为(b,h,w),tensor b的size为(b,w,m) 
        # 也就是说两个tensor的第一维是相等的，然后第一个数组的第三维和第二个数组的第二维度要求一样，对于剩下的则不做要求，输出维度 （b,h,m)
        # 只能做三维乘法，矩阵是二维或者四维都会报错
        dot = torch.bmm(queries, keys.transpose(1, 2)) / (e ** (1/2))


        # row wise self attention probabilities
        # dot = F.softmax(dot, dim=-1)
        # dot就是权重c
        self.dot = dot
        out = torch.bmm(dot, values).transpose(1, 2).contiguous().view(b, h, e)
        out = self.unify_all_heads(out)
        # values = values.view(b, h, e)
        # values = values.transpose(1, 2).contiguous().view(b, h * e)
        # self.values = values
        return out

















class Observe_State_Attention1(nn.Module):
    def __init__(self, observe_size, state_size, heads, embed_size):
        super().__init__()
        self.observe_size = observe_size
        self.state_size = state_size
        self.heads = heads
        self.emb_size = embed_size
        # Q是输入信息，K是待匹配的信息，V是信息本身，V只是单纯表达了输入特征的信息
        self.toqueries = nn.Linear(self.observe_size, self.emb_size * heads, bias = False)
        self.tokeys = nn.Linear(self.state_size, self.emb_size * heads, bias = False)
        self.tovalues = nn.Linear(self.state_size, self.emb_size * heads, bias = False)
        self.unify_all_heads = Merger(heads, self.emb_size)
    def forward(self, observe, state):
        b, t, hin = observe.shape
        assert hin == self.observe_size, 'Input size {} should match {}'.format(hin, self.observe_size)
        
        h = self.heads 
        e = self.emb_size
        
        # if next(self.toqueries.parameters()).device != observe.device:
        #     observe = observe.to(next(self.toqueries.parameters()).device)
        #     state = state.to(next(self.tokeys.parameters()).device)

        queries = self.toqueries(observe).view(b, t, h, e)
        keys = self.tokeys(state).view(b, t, h, e)
        values = self.tovalues(state).view(b, t, h, e)

        # dot-product attention
        # folding heads to batch dimensions
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, e)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, e)
        values = values.transpose(1, 2).contiguous().view(b * h, t, e)

        queries = queries / (e ** (1/4))
        keys = keys / (e ** (1/4))
        # 计算两个tensor的矩阵乘法，torch.bmm(a,b),tensor a 的size为(b,h,w),tensor b的size为(b,w,m) 
        # 也就是说两个tensor的第一维是相等的，然后第一个数组的第三维和第二个数组的第二维度要求一样，对于剩下的则不做要求，输出维度 （b,h,m)
        # 只能做三维乘法，矩阵是二维或者四维都会报错
# 不会是dot这里有错误吧，dot的维度需要调转一下吗？？？？？？？
# dot没有错，就应该是query在key里面查询
        dot = torch.bmm(queries, keys.transpose(1, 2))
        assert dot.size() == (b*h, t, t)

        # row wise self attention probabilities
        dot = F.softmax(dot, dim=2)
        # dot就是权重c
        self.dot = dot
        out = torch.bmm(dot, values).view(b, h, t, e)
        out = out.transpose(1, 2).contiguous().view(b*t, h, e)
        out = self.unify_all_heads(out)
        # values = values.view(b, h, t, e)
        # values = values.transpose(1, 2).contiguous().view(b, t, h * e)
        # self.values = values
        return out





class Observe_State_Attention2(nn.Module):
    def __init__(self, observe_size, state_size, heads, embed_size):
        super().__init__()
        self.observe_size = observe_size
        self.state_size = state_size
        self.heads = heads
        self.emb_size = embed_size
        # Q是输入信息，K是待匹配的信息，V是信息本身，V只是单纯表达了输入特征的信息
        self.toqueries1 = nn.Sequential(
                nn.Linear(self.observe_size, self.emb_size * heads, bias = False),
                nn.ReLU(inplace=True),
                nn.Linear(self.emb_size * heads,self.emb_size * 4 * heads, bias = False),
                nn.ReLU(inplace=True),
                nn.Linear(self.emb_size * 4 * heads,self.emb_size * heads, bias = False)
            )
        self.tokeys1 = nn.Sequential(
                nn.Linear(self.state_size, self.emb_size * heads, bias = False),
                nn.ReLU(inplace=True),
                nn.Linear(self.emb_size * heads,self.emb_size * 4 * heads, bias = False),
                nn.ReLU(inplace=True),
                nn.Linear(self.emb_size * 4 * heads,self.emb_size * heads, bias = False)
            )
        self.tovalues1 = nn.Sequential(
                nn.Linear(self.state_size, self.emb_size * heads, bias = False),
                nn.ReLU(inplace=True),
                nn.Linear(self.emb_size * heads,self.emb_size * 4 * heads, bias = False),
                nn.ReLU(inplace=True),
                nn.Linear(self.emb_size * 4 * heads,self.emb_size * heads, bias = False)
            )
        self.toqueries2 = nn.Sequential(
                nn.Linear(self.observe_size, self.emb_size * heads, bias = False),
                nn.ReLU(inplace=True),
                nn.Linear(self.emb_size * heads,self.emb_size * 2 * heads, bias = False),
                nn.ReLU(inplace=True),
                nn.Linear(self.emb_size * 2 * heads,self.emb_size * heads, bias = False)
            )
        self.tokeys2 = nn.Sequential(
                nn.Linear(self.state_size, self.emb_size * heads, bias = False),
                nn.ReLU(inplace=True),
                nn.Linear(self.emb_size * heads,self.emb_size * 2 * heads, bias = False),
                nn.ReLU(inplace=True),
                nn.Linear(self.emb_size * 2 * heads,self.emb_size * heads, bias = False)
            )
        self.tovalues2 = nn.Sequential(
                nn.Linear(self.state_size, self.emb_size * heads, bias = False),
                nn.ReLU(inplace=True),
                nn.Linear(self.emb_size * heads,self.emb_size * 2 * heads, bias = False),
                nn.ReLU(inplace=True),
                nn.Linear(self.emb_size * 2 * heads,self.emb_size * heads, bias = False)
            )
        
        self.unify_queries_heads = Merger(2, self.emb_size * heads)
        self.unify_keys_heads = Merger(2, self.emb_size * heads)
        self.unify_values_heads = Merger(2, self.emb_size * heads)

        self.unify_all_heads = Merger(heads, self.emb_size)

        self.fc = nn.Linear(self.heads, 1)



    def forward(self, observe, state):
        b, t, hin = observe.size()
        assert hin == self.observe_size, 'Input size {} should match {}'.format(hin, self.observe_size)
        
        h = self.heads 
        e = self.emb_size

        queries1 = self.toqueries1(observe).view(b*t,-1).unsqueeze(1)
        keys1 = self.tokeys1(state).view(b*t, -1).unsqueeze(1)
        values1 = self.tovalues1(state).view(b*t, -1).unsqueeze(1)

        queries2 = self.toqueries2(observe).view(b*t, -1).unsqueeze(1)
        keys2 = self.tokeys2(state).view(b*t, -1).unsqueeze(1)
        values2 = self.tovalues2(state).view(b*t, -1).unsqueeze(1)

        queries = self.unify_queries_heads(torch.cat([queries1, queries2], dim=1)).view(b, t, h, e)
        keys = self.unify_queries_heads(torch.cat([keys1, keys2], dim=1)).view(b, t, h, e)
        values = self.unify_queries_heads(torch.cat([values1, values2], dim=1)).view(b, t, h, e)

        # dot-product attention
        # folding heads to batch dimensions
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, e)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, e)
        values = values.transpose(1, 2).contiguous().view(b * h, t, e)

        queries = queries / (e ** (1/4))
        keys = keys / (e ** (1/4))
        # 计算两个tensor的矩阵乘法，torch.bmm(a,b),tensor a 的size为(b,h,w),tensor b的size为(b,w,m) 
        # 也就是说两个tensor的第一维是相等的，然后第一个数组的第三维和第二个数组的第二维度要求一样，对于剩下的则不做要求，输出维度 （b,h,m)
        # 只能做三维乘法，矩阵是二维或者四维都会报错
        dot = torch.bmm(queries, keys.transpose(1, 2))
        assert dot.size() == (b*h, t, t)
        # row wise self attention probabilities
        dot = F.softmax(dot, dim=2)
        # dot就是权重c
        self.dot = dot
        out = torch.bmm(dot, values).view(b, h, t, e)
        out = out.transpose(1, 2).contiguous().view(b*t, h, e)
        out = self.unify_all_heads(out)
        values = values.view(b, h, t, e)
        values = values.transpose(1, 2).contiguous().view(b, t, h * e)
        self.values = values
        return out
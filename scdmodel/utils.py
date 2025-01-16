import math
from einops import rearrange

def features_transfer(x, data_format='NCWH'):
        S = x.shape[1]
        s = int(math.sqrt(S))
        if data_format == 'NWHC':
            x = rearrange(x, 'b (h w) c-> b h w c', h=s, w=s)
        elif data_format == 'NCWH':
            x = rearrange(x, 'b (h w) c-> b c h w', h=s, w=s)
        return x

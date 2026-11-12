"""Visible-input v1. Ragged P/E/N/S, objective room cells, independent local patches.

The inline transport budget is bytes, not an inventory/slot token limit. A whole
oversized frame fails before publication. No legacy field is an actor input.
"""
import numpy as np
import torch
from functools import lru_cache
from torch import nn
from torch.nn import functional as F

from .schema_v2 import (SCHEMA, VERSION, FRAME_BYTES, P_FIELDS, STAGE_INDEX, CURSE_START, OVERFLOW_START,
                        ENTITY_CLASSES, PF, EF, NF, SF, RC, HEADER, FIXED, RAGGED, ROOM_BITS, ROOM_SCALES)
# (the producer half, package C's since 2026-11-12: re-exported here, the existing imports unchanged)
from .visible_producer import (pack_canvas, pack_room, pack_frame, write_frame, observation_frame,
                               prioritize_entities)


def unpack_canvas(x,bits,scales):
    # Python-int shifts/masks/scales: no host tensor constants (their host->device copy is not
    # allowed inside a CUDA graph capture); Inductor fuses the per-channel stack (B42 glue 2).
    w=x.long()
    return torch.stack([((w>>sum(bits[:i]))&(2**bits[i]-1)).float()/float(scales[i]) for i in range(len(bits))],-1)


def unpack_room(x):
    x=x.reshape(*x.shape[:-1],-1,4)
    return torch.cat((unpack_canvas(x[...,0],ROOM_BITS,ROOM_SCALES),x[...,1:]),-1)


# B42 glue: fixed token buckets. Every ragged width is one of a few shapes chosen on the host
# from the record headers (never by reading device memory): bodies/local patches {1,2,4},
# entities {16,32,64,128} (pack_frame caps at 128), inventory/slots one upper bucket. Counts
# past a fixed set double (never truncate); the published schema keeps them that small.
BUCKETS = {'body':(1,2,4),'local':(1,2,4),'entity':(16,32,64,128),'inventory':(32,),'slots':(16,)}


def bucket_width(name, m):
    for w in BUCKETS[name]:
        if m <= w:
            return w
    w = BUCKETS[name][-1]
    while w < m:
        w *= 2
    return w


def frame_counts(frames):
    """Host (p,e,n,s) [n,4] int64 from the header of complete visible-v2 frames (uint8 [n,>=24])."""
    return np.ascontiguousarray(frames[:, :HEADER*4]).view('<f4')[:, 2:6].astype(np.int64)


def decode_frame(raw, sizes, versions, counts=None):
    """Device-native typed gather. counts (B42): the host (p,e,n,s) of the rows (frame_counts of
    the same records, checked on the host when written/decoded): fixed buckets, no device->host
    synchronisation and no device-side checks. None: the checked legacy path (exact buckets)."""
    f = raw.contiguous().view(torch.float32)
    if counts is None:
        if not bool((versions == VERSION).all()):
            raise ValueError('missing/old visible input schema; legacy observations are not accepted')
        counts_d = f[:, 2:6].long()
        p,e,n,s = counts_d.unbind(1)
        expected = FIXED + p*(PF+324) + e*EF+n*NF+s*SF
        if not bool(((f[:,0] == 61107) & (f[:,1] == expected) & (sizes == expected*4) &
                     (p > 0) & (e <= 128) & (counts_d >= 0).all(1) & (sizes <= FRAME_BYTES)).all()):
            raise ValueError('invalid visible-v2 complete frame length/counts')
        widths = [max(1, (int(c.max())+3)//4*4) for c in (p,p,e,n,s)]
    else:
        counts = np.asarray(counts, np.int64)
        widths = [bucket_width(k, int(counts[:, j].max()) if len(counts) else 1)
                  for k, j in zip(RAGGED, (0,0,1,2,3))]
        counts_d = torch.from_numpy(counts)
        if f.device.type == 'cuda':
            counts_d = counts_d.pin_memory()
        counts_d = counts_d.to(f.device, non_blocking=True)
        p,e,n,s = counts_d.unbind(1)
    b = {}
    offset = torch.full_like(p, FIXED)
    for name, count, bucket in zip(RAGGED, (p,p,e,n,s), widths):
        width = RAGGED[name]
        at = torch.arange(bucket,device=f.device)[None,:]
        valid = at < count[:,None]
        idx = offset[:,None,None]+at[:,:,None]*width+torch.arange(width,device=f.device)
        values = f.gather(1, idx.clamp_max(f.shape[1]-1).reshape(len(f),-1)).reshape(len(f),bucket,width)
        b['v_'+name] = values.masked_fill(~valid[:,:,None],0)
        b['v_'+name+'_mask'] = valid
        offset = offset+count*width
    # B42: the same fused per-bucket bitplane expansion as the host-decoded path (from_host).
    with torch.inference_mode(False),torch.no_grad(),torch.autocast(f.device.type,enabled=False):
        b['v_room'],b['v_local'],b['v_map'] = canvas_decoder()(
            f[:,HEADER:HEADER+1792].contiguous(),b['v_local'],f[:,HEADER+1792:FIXED].contiguous())
    return b


def from_rows(rows, device, pending=None):
    # NumPy views decode the host wire without copying its 128-KiB padding or
    # dispatching hundreds of tiny Torch CPU operations for every history batch.
    device=torch.device(device)
    b=decode_frame_host(rows,packed=device.type=='cuda')
    if pending is not None:
        b['pending']=np.asarray(pending,dtype=np.int64)
        b['first']=rows['first']
    return from_host(b,device)


# B42 glue 4: attention layout buckets by a row's total real tokens (30 + 2 per body + entities
# + items + slots). Early-training rows have 36-67 tokens (B42 prof-f0/f1 dumps), late-game
# rows up to ~200. Past the set: doubling (never truncation).
T_BUCKETS=(40,48,64,96,128,192,256,512)
PACKED_KEYS=('room','map','pending','body','body_row','local','entity','inventory','slots','d_body','d_room',
             'd_local','d_map','d_entity','d_inventory','d_slots','d_global','keep','rows_valid')


def t_bucket(m):
    for w in T_BUCKETS:
        if m<=w:return w
    w=T_BUCKETS[-1]
    while w<m:w*=2
    return w


def packed_rows_batch(batch,rows,counts,device):
    """A row-packed StructuredSpatial batch (StructuredSpatial.forward_rows) of `rows` of a
    decoded batch. batch: a from_host dict (typed, left-aligned ragged slots); rows: host int64
    row indices; counts: their host real counts [n,5] (host_counts order). All index arithmetic
    is on the host (one H2D of all indices); the device only gathers. Lists are padded to >= 2
    entries (destination: the trash row) and a 1-row batch to 2 rows (a duplicate), so no
    dimension is 0 or 1 (dynamo would specialise it); keep the first len(rows) outputs."""
    idx,T=packed_rows_plan({k:batch['v_'+k].shape[1] for k in RAGGED},rows,counts)
    names=list(idx);sizes=[len(idx[k]) for k in names]
    flat=torch.from_numpy(np.concatenate([idx[k] for k in names]).astype(np.int64))
    device=torch.device(device)
    if device.type=='cuda':flat=flat.pin_memory()
    flat=flat.to(device,non_blocking=True)
    return packed_rows_apply(batch,dict(zip(names,flat.split(sizes))),T)


def packed_rows_plan(widths,rows,counts,starts=None):
    """packed_rows_batch's host part: (index arrays (host int64), T). widths: the decoded batch's
    ragged slot widths {name: width}. starts (B44, compact lists, decode_frame_compact): {name:
    (first entry of each batch row [n], index of the zero entry)}; then sources index the compact
    [K+1, width] arrays."""
    rows=np.asarray(rows,np.int64);counts=np.asarray(counts,np.int64)
    if len(rows)==1:rows,counts=np.repeat(rows,2),np.repeat(counts,2,0)
    n=len(rows);p,pl,e,ni,sl=counts.T
    if np.any(p!=pl):raise ValueError('visible-v2 has one local patch per body')
    tot=30+p+pl+e+ni+sl;T=t_bucket(int(tot.max()));trash=n*T
    base=np.arange(n,dtype=np.int64)*T
    off={'body':np.zeros(n,np.int64),'local':p+28,'entity':p+pl+29,'inventory':p+pl+29+e,'slots':p+pl+29+e+ni}
    idx,src={},{}
    for name,c in (('body',p),('local',pl),('entity',e),('inventory',ni),('slots',sl)):
        W=widths[name] if starts is None else 0;k=int(c.sum())
        rr=np.repeat(np.arange(n,dtype=np.int64),c);j=np.arange(k)-np.repeat(np.cumsum(c)-c,c)
        if starts is None:
            sv=np.repeat(rows,c)*W+j;pad=0
        else:
            sv=np.repeat(starts[name][0][rows],c)+j;pad=starts[name][1]
        dv=base[rr]+off[name][rr]+j
        if k<2:
            sv=np.concatenate((sv,np.full(2-k,pad,np.int64)));dv=np.concatenate((dv,np.full(2-k,trash)))
            rr=np.concatenate((rr,np.zeros(2-k,np.int64)))
        src[name]=sv;idx['d_'+name]=dv
        if name=='body':idx['body_row']=rr
    idx['d_room']=((base+p)[:,None]+np.arange(28)).reshape(-1)
    idx['d_map']=base+2*p+28
    idx['d_global']=base+tot-1
    at=np.arange(T)[None,:]
    idx['keep']=(base[:,None]+at)[at<tot[:,None]]
    idx['rows']=rows;idx['tot']=tot
    for name in src:idx['s_'+name]=src[name]
    return idx,T


def packed_rows_apply(batch,d,T):
    """packed_rows_batch's device part: d holds packed_rows_plan's arrays on the device."""
    out={k:d[k] for k in ('body_row','d_body','d_local','d_entity','d_inventory','d_slots','d_room','d_map',
                          'd_global','keep')}
    out['rows_valid']=torch.arange(T,device=d['tot'].device)[None,:]<d['tot'][:,None]
    rd=d['rows']
    out['room']=batch['v_room'].index_select(0,rd);out['map']=batch['v_map'].index_select(0,rd)
    out['pending']=batch['pending'].index_select(0,rd)
    for name in RAGGED:
        v=batch['v_'+name];out[name]=v.reshape(-1,v.shape[-1]).index_select(0,d['s_'+name])
    return out


def host_counts(b):
    """Real (body, local, entity, inventory, slots) token counts [n,5] of a host-decoded dict."""
    return np.stack([b['v_'+k+'_mask'].sum(1) for k in RAGGED],1).astype(np.int64)


def merge_host(parts):
    """Concatenate decode_frame_host dicts (same packing), padding ragged slots."""
    if len(parts)==1:return dict(parts[0])
    out={}
    for k in parts[0]:
        xs=[q[k] for q in parts]
        base=k[2:-5] if k.endswith('_mask') else k[2:]
        if k.startswith('v_') and base in RAGGED and len({x.shape[1] for x in xs})>1:
            w=max(x.shape[1] for x in xs)
            xs=[np.pad(x,[(0,0),(0,w-x.shape[1])]+[(0,0)]*(x.ndim-2)) for x in xs]
        out[k]=np.concatenate(xs)
    return out


def from_host(b, device):
    """Upload a decode_frame_host dict (plus pending/first) and finish canvas decoding."""
    device=torch.device(device)
    if device.type=='cpu':return {k:torch.from_numpy(v) for k,v in b.items()}
    # A dtype-preserving byte slab needs one H2D and no GPU float->mask/int casts.
    layout={};offset=0
    for name,value in b.items():
        offset=(offset+7)//8*8
        layout[name]=(offset,value.nbytes,value.shape,torch.from_numpy(np.empty((),dtype=value.dtype)).dtype)
        offset+=value.nbytes
    slab=np.empty(offset,dtype=np.uint8)
    for name,value in b.items():
        np.ndarray(value.shape,dtype=value.dtype,buffer=slab,offset=layout[name][0])[...]=value
    # Data decoding never needs autograd or autocast. A single execution context
    # avoids recompiling identical bitfields for actor/learner and FP32/BF16.
    with torch.inference_mode(False),torch.no_grad(),torch.autocast('cuda',enabled=False):
        flat=torch.from_numpy(slab).pin_memory().to(device,non_blocking=True)
        b={name:flat.narrow(0,start,size).view(dtype).reshape(shape)
           for name,(start,size,shape,dtype) in layout.items()}
        b['v_room'],b['v_local'],b['v_map']=canvas_decoder()(b['v_room'],b['v_local'],b['v_map'])
    return b


def expand_canvases(room,local,minimap):
    """Tensor-only expansion: upload packed cells, fuse their bitfield decoding."""
    room=unpack_room(room).transpose(1,2).reshape(-1,RC,16,28)
    local=unpack_room(local).transpose(2,3).flatten(2)
    minimap=unpack_canvas(minimap,(1,)*8,(1,)*8).transpose(1,2).reshape(-1,8,13,13)
    return room,local,minimap


@lru_cache(maxsize=1)
def canvas_decoder():
    # B42: one fused Inductor kernel set per body bucket {1,2,4} (static token shapes, batch
    # symbolic): the room/local/minimap bitplane unpack of every decoder. No CUDA-graph-owned
    # output buffers. Off the GPU (CPU tests) the eager expansion.
    from .block_compile import compile_batch_dynamic
    compiled=compile_batch_dynamic(expand_canvases)
    def call(room,local,minimap):
        if room.device.type!='cuda':return expand_canvases(room,local,minimap)
        return compiled(room,local,minimap)
    return call


def _unpack_room_host(x):
    x=x.reshape(*x.shape[:-1],-1,4)
    shifts=np.cumsum((0,)+ROOM_BITS[:-1],dtype=np.uint32)[:,None]
    masks=((np.uint32(1)<<np.asarray(ROOM_BITS,dtype=np.uint32))-np.uint32(1))[:,None]
    bits=np.right_shift(x[...,0].astype(np.uint32)[...,None,:],shifts)
    np.bitwise_and(bits,masks,out=bits)
    result=np.empty((*x.shape[:-2],16,x.shape[-2]),dtype=np.float32)
    np.divide(bits,np.asarray(ROOM_SCALES,dtype=np.float32)[:,None],
              out=result[...,:13,:],dtype=np.float32)
    result[...,13:,:]=np.moveaxis(x[...,1:],-1,-2)
    return result


def check_frames(rows):
    """Host validity check of complete visible-v2 frames (ROW records); returns (p,e,n,s) [n,4].
    Reads only the 24-byte headers; the device decoders never check (or synchronise) again."""
    head=rows['visible_frame'].reshape(len(rows),-1)[:,:HEADER*4]
    return check_headers(rows['visible_schema'],head,rows['visible_bytes'])


def check_headers(schema,head,size):
    """check_frames on the record fields only: schema [n], frame header bytes [n,24], sizes [n]."""
    if not np.all(np.asarray(schema)==VERSION):
        raise ValueError('missing/old visible input schema; legacy observations are not accepted')
    h=np.ascontiguousarray(head).view('<f4')
    counts=h[:,2:6].astype(np.int64);p,e,n,s=counts.T
    expected=FIXED+p*(PF+324)+e*EF+n*NF+s*SF
    size=np.asarray(size).reshape(-1)
    if not np.all((h[:,0]==61107)&(h[:,1]==expected)&(size==expected*4)&
                  (p>0)&(e<=128)&(counts>=0).all(1)&(size<=FRAME_BYTES)):
        raise ValueError('invalid visible-v2 complete frame length/counts')
    return counts


def decode_frame_host(rows,packed=False):
    """Same contract as decode_frame; strided ROW views remain zero-copy on input."""
    f=rows['visible_frame'].view('<f4')
    counts=check_frames(rows);p,e,n,s=counts.T
    room=f[:,HEADER:HEADER+1792]
    minimap=f[:,HEADER+1792:FIXED]
    if not packed:
        room=_unpack_room_host(room).reshape(-1,RC,16,28)
        minimap=((minimap.astype(np.int64)[...,None]>>np.arange(8))&1).astype(np.float32)
        minimap=minimap.transpose(0,2,1).reshape(-1,8,13,13)
    b={'v_room':room,'v_map':minimap}
    offset=np.full_like(p,FIXED)
    for name,count in zip(RAGGED,(p,p,e,n,s)):
        width=RAGGED[name];bucket=bucket_width(name,int(count.max()))
        at=np.arange(bucket)[None,:];valid=at<count[:,None]
        idx=offset[:,None,None]+at[:,:,None]*width+np.arange(width)
        values=f[np.arange(len(rows))[:,None,None],np.minimum(idx,f.shape[1]-1)]
        values[~valid]=0
        b['v_'+name]=values;b['v_'+name+'_mask']=valid
        offset+=count*width
    if not packed:b['v_local']=_unpack_room_host(b['v_local']).reshape(len(rows),-1,RC*81)
    return b


def decode_frame_compact(rows,packed=False):
    """decode_frame_host's values without bucket padding (B44 packed learner): v_room / v_map as
    decode_frame_host; each ragged list as 'c_<name>' [K, width]: the K real entries of all rows in
    row order (row r's from the cumulative count of the rows before it); 'counts' [n,5] (body,
    local, entity, inventory, slots). Host work and staged bytes follow the real entries."""
    return decode_frames_compact([rows],packed)[0]


class _Frames:
    """Float32 reader of a part's frames: ROW-like records (visible_frame inline) or (B45) compact records with
    their frame arena ((rows, arena bytes): the frame of row r starts at byte rows['visible_off'][r])."""

    def __init__(self, part):
        if isinstance(part, tuple):
            rows, buf = part
            buf = np.asarray(buf, np.uint8)
            self.f = buf[:len(buf)//4*4].view('<f4')
            self.base = np.asarray(rows['visible_off'], np.int64)//4
            head = buf[np.asarray(rows['visible_off'], np.int64)[:, None]+np.arange(HEADER*4)]
            self.counts = check_headers(rows['visible_schema'], head, rows['visible_bytes'])
            self.n = len(rows)
        else:
            self.f = part['visible_frame'].view('<f4')
            self.base = None
            self.counts = check_frames(part)
            self.n = len(part)

    def take(self, r, col):
        """f[r, col] (r: rows of this part [k], col [k, w])."""
        if self.base is None:
            return self.f[r[:, None] if col.ndim == 2 else r, col]
        return self.f[self.base[r][:, None]+col if col.ndim == 2 else self.base[r]+col]

    def block(self, lo, hi):
        """f[:, lo:hi] of every row."""
        if self.base is None:
            return self.f[:, lo:hi]
        return self.f[self.base[:, None]+np.arange(lo, hi)]


def decode_frames_compact(parts,packed=False):
    """decode_frame_compact of several record arrays at once (one set of index computations, one
    fancy-index gather per array and list); returns one dict per array (views of shared arrays).
    A part is a ROW-like record array or (B45) a (compact records, frame arena) pair."""
    fr=[_Frames(r) for r in parts]
    lens=np.array([x.n for x in fr],np.int64)
    m=int(lens.sum())
    counts=np.concatenate([x.counts for x in fr]) if fr else np.zeros((0,4),np.int64)
    p,e,n,s_=counts.T
    counts=np.stack((p,p,e,n,s_),1).astype(np.int64)
    row_at=np.concatenate(([0],np.cumsum(lens)))
    room=np.empty((m,1792),np.float32);minimap=np.empty((m,169),np.float32)
    for i,x in enumerate(fr):
        room[row_at[i]:row_at[i+1]]=x.block(HEADER,HEADER+1792);minimap[row_at[i]:row_at[i+1]]=x.block(HEADER+1792,FIXED)
    if not packed:
        room=_unpack_room_host(room).reshape(-1,RC,16,28)
        minimap=((minimap.astype(np.int64)[...,None]>>np.arange(8))&1).astype(np.float32)
        minimap=minimap.transpose(0,2,1).reshape(-1,8,13,13)
    offset=np.full(m,FIXED,np.int64)
    lists,ent_at={},{}
    for k,name in enumerate(RAGGED):
        width=RAGGED[name];c=counts[:,k];K=int(c.sum())
        out=np.empty((K,width),np.float32)
        cum=np.concatenate(([0],np.cumsum(c)))
        if K:
            rr=np.repeat(np.arange(m),c);j=np.arange(K)-np.repeat(cum[:-1],c)
            col=(offset[rr]+j*width)[:,None]+np.arange(width)
            for i,x in enumerate(fr):
                a,b=cum[row_at[i]],cum[row_at[i+1]]
                if b>a:out[a:b]=x.take(rr[a:b]-row_at[i],col[a:b])
        if name=='local' and not packed:out=_unpack_room_host(out).reshape(K,RC*81)
        lists[name]=out;ent_at[name]=cum
        offset+=c*width
    res=[]
    for i in range(len(parts)):
        lo,hi=row_at[i],row_at[i+1]
        d={'v_room':room[lo:hi],'v_map':minimap[lo:hi],'counts':counts[lo:hi]}
        for name in RAGGED:
            cum=ent_at[name];d['c_'+name]=lists[name][cum[lo]:cum[hi]]
        res.append(d)
    return res


def compile_inputs(b,keys):
    """The compiled Spatial's inputs: contiguous (static strides), offset-0, normal tensors (dynamo
    guards on dispatch keys and storage offsets: an inference-mode tensor or a slab view
    recompiled every variant)."""
    out={}
    with torch.inference_mode(False):
        for k in keys:
            v=b[k]
            # a view's storage offset is guarded too (0/1-specialised): fresh offset-0 storage
            out[k]=(v.clone(memory_format=torch.contiguous_format) if v.is_inference() or v.storage_offset()
                    else v.contiguous())
    return out


SPATIAL_KEYS=('v_body','v_body_mask','v_room','v_local','v_local_mask','v_map','v_entity','v_entity_mask',
              'v_inventory','v_inventory_mask','v_slots','v_slots_mask','pending')
SPATIAL_COMPILE=__import__('os').environ.get('ISAAC_SPATIAL_COMPILE','1')=='1'


class StructuredSpatial(nn.Module):
    def __init__(self,width,layers,heads):
        super().__init__()
        from .tok_policy import Block
        self.body=nn.Sequential(nn.Linear(PF,width),nn.GELU(),nn.Linear(width,width))
        self.room=nn.Sequential(nn.Conv2d(RC,32,3,2,1),nn.GELU(),nn.Conv2d(32,64,3,2,1),nn.GELU())
        self.room_project=nn.Linear(64,width)
        self.position=nn.Linear(2,width,bias=False)
        self.local=nn.Sequential(nn.Conv2d(RC,32,3,1,1),nn.GELU(),nn.Conv2d(32,32,3,1,1),
                                 nn.GELU(),nn.Flatten(),nn.Linear(32*9*9,width))
        self.minimap=nn.Sequential(nn.Conv2d(8,32,3,2,1),nn.GELU(),nn.Flatten(),nn.Linear(32*7*7,width))
        self.identity=nn.Embedding(65536,24,padding_idx=0)
        self.entity=nn.Linear(EF+72,width)
        self.inventory=nn.Linear(NF+24,width)
        self.slots=nn.Linear(SF+24,width)
        self.owner=nn.Linear(1,width,bias=False)
        self.pending=nn.Linear(20,width,bias=False)
        self.kind=nn.Parameter(torch.randn(8,width)*.02)
        self.global_query=nn.Parameter(torch.randn(1,1,width)*.02)
        self.blocks=nn.ModuleList(Block(width,heads) for _ in range(layers))
        y,x=torch.meshgrid(torch.linspace(-1,1,4),torch.linspace(-1,1,7),indexing='ij')
        self.register_buffer('coordinates',torch.stack((x,y),-1).reshape(28,2))

    def forward(self,b):
        if 'rows_valid' in b:
            # B42 glue 4 (learner): row-packed batch (packed_rows_batch): every per-token part on
            # real tokens only, attention per row-total bucket T. One compiled graph per T.
            x=compile_inputs(b,PACKED_KEYS)
            if SPATIAL_COMPILE and x['room'].is_cuda:
                from .block_compile import spatial_rows_compiled
                return spatial_rows_compiled()(self,x)
            return self.forward_rows(x)
        if 'v_body' not in b:raise ValueError('new Spatial requires visible-v2 input; legacy input rejected')
        x=compile_inputs(b,SPATIAL_KEYS)
        # B42 glue 3: on the GPU the whole forward is one compiled graph per (entity, body)
        # bucket and execution mode (grad/no-grad, autocast): the clone/masked_fill/cat/
        # one_hot/log1p/kind-bias/mask glue fuses in Inductor, linear layers stay cuBLAS.
        if SPATIAL_COMPILE and x['v_body'].is_cuda:
            from .block_compile import spatial_compiled
            return spatial_compiled()(self,x)
        return self.forward_eager(x)

    def forward_eager(self,b):
        b=dict(b)
        for name in RAGGED:
            b['v_'+name]=b['v_'+name].masked_fill(~b['v_'+name+'_mask'][:,:,None],0)
        bodies=b['v_body'].clone();batch=len(bodies)
        bodies[:,:,5:12]=bodies[:,:,5:12].masked_fill(bodies[:,:,40:41]==0,0)
        bodies[:,:,38]=bodies[:,:,38].masked_fill(bodies[:,:,39]==0,0)
        bodies[:,:,36]=bodies[:,:,36].masked_fill(bodies[:,:,37]==0,-1)
        bodies[:,:,30:33]=0  # no hidden engine cooldown/charge counters
        pending=torch.cat([F.one_hot(b['pending'][:,i].long(),n) for i,n in enumerate((9,5,2,2,2))],-1).float()
        p=self.body(bodies)+self.pending(pending)[:,None,:]*(bodies[:,:,36:37]==0)
        r=self.room_project(self.room(b['v_room']).flatten(2).transpose(1,2))+self.position(self.coordinates)
        l=b['v_local'];l=self.local(l.reshape(-1,RC,9,9)).reshape(batch,l.shape[1],-1)
        l=l+self.owner(bodies[:,:,35:36])
        e=b['v_entity'];eknown=e[:,:,47:48]>0
        ids=e[:,:,:3].long();ids=torch.cat((ids[:,:,:2],ids[:,:,2:].masked_fill(~eknown,0)),2)
        e=e.clone();e[:,:,2:3]=e[:,:,2:3].masked_fill(~eknown,0)
        e=self.entity(torch.cat((e/64,self.identity(ids).flatten(2)),2))
        n=b['v_inventory'].clone();n[:,:,1]=n[:,:,1].masked_fill(n[:,:,3]==0,0)
        nt=self.inventory(torch.cat((torch.sign(n)*torch.log1p(n.abs()),self.identity(n[:,:,1].long())),2))
        s=b['v_slots'].clone();s[:,:,3]=s[:,:,3].masked_fill(s[:,:,8]==0,0)
        s[:,:,5]=s[:,:,5].masked_fill(s[:,:,9]==0,0)
        s[:,:,6:8]=s[:,:,6:8].masked_fill(s[:,:,10:11]==0,0)
        s[:,:,12]=s[:,:,12].masked_fill(s[:,:,10]==0,0)
        st=self.slots(torch.cat((torch.sign(s)*torch.log1p(s.abs()),self.identity(s[:,:,3].long())),2))
        chunks=(p,r,l,self.minimap(b['v_map'])[:,None],e,nt,st,self.global_query.expand(batch,-1,-1))
        x=torch.cat([v+self.kind[i] for i,v in enumerate(chunks)],1)
        valid=torch.cat((b['v_body_mask'],torch.ones((batch,28),device=x.device,dtype=torch.bool),
                         b['v_local_mask'],torch.ones((batch,1),device=x.device,dtype=torch.bool),
                         b['v_entity_mask'],b['v_inventory_mask'],b['v_slots_mask'],
                         torch.ones((batch,1),device=x.device,dtype=torch.bool)),1)
        bias=x.new_zeros((batch,1,1,x.shape[1])).masked_fill(~valid[:,None,None,:],float('-inf'))
        for block in self.blocks:x=block(x,bias)
        return x,~valid

    def forward_rows(self,x):
        """forward_eager on a row-packed batch (packed_rows_batch): the same per-token functions
        applied to real bodies/local patches/entities/items/slots only (each list padded to >=2
        entries whose destination is a trash row), each row's tokens compacted to the front of
        a [n, T] layout (T: its row-total bucket), the Transformer's per-token layers on the real
        tokens (Block.forward_packed), attention in [n, T] with the padded keys masked. The
        attention and the resampler do not depend on token order: same values up to summation
        order. Returns ([n, T, width], padding mask)."""
        valid=x['rows_valid'];n,T=valid.shape
        bodies=x['body'].clone()
        bodies[:,5:12]=bodies[:,5:12].masked_fill(bodies[:,40:41]==0,0)
        bodies[:,38]=bodies[:,38].masked_fill(bodies[:,39]==0,0)
        bodies[:,36]=bodies[:,36].masked_fill(bodies[:,37]==0,-1)
        bodies[:,30:33]=0
        pending=torch.cat([F.one_hot(x['pending'][:,i].long(),k) for i,k in enumerate((9,5,2,2,2))],-1).float()
        p=self.body(bodies)+self.pending(pending).index_select(0,x['body_row'])*(bodies[:,36:37]==0)
        r=self.room_project(self.room(x['room']).flatten(2).transpose(1,2))+self.position(self.coordinates)
        l=self.local(x['local'].reshape(-1,RC,9,9))+self.owner(bodies[:,35:36])
        e=x['entity'];eknown=e[:,47:48]>0
        ids=e[:,:3].long();ids=torch.cat((ids[:,:2],ids[:,2:].masked_fill(~eknown,0)),1)
        e=e.clone();e[:,2:3]=e[:,2:3].masked_fill(~eknown,0)
        e=self.entity(torch.cat((e/64,self.identity(ids).flatten(1)),1))
        q=x['inventory'].clone();q[:,1]=q[:,1].masked_fill(q[:,3]==0,0)
        nt=self.inventory(torch.cat((torch.sign(q)*torch.log1p(q.abs()),self.identity(q[:,1].long())),1))
        s=x['slots'].clone();s[:,3]=s[:,3].masked_fill(s[:,8]==0,0)
        s[:,5]=s[:,5].masked_fill(s[:,9]==0,0)
        s[:,6:8]=s[:,6:8].masked_fill(s[:,10:11]==0,0)
        s[:,12]=s[:,12].masked_fill(s[:,10]==0,0)
        st=self.slots(torch.cat((torch.sign(s)*torch.log1p(s.abs()),self.identity(s[:,3].long())),1))
        pieces=(p,r.flatten(0,1),l,self.minimap(x['map']),e,nt,st,self.global_query[0].expand(n,-1))
        vals=[v+self.kind[i] for i,v in enumerate(pieces)]
        flat=vals[0].new_zeros((n*T+1,vals[0].shape[-1]))      # last row: trash
        for v,k in zip(vals,('d_body','d_room','d_local','d_map','d_entity','d_inventory','d_slots','d_global')):
            flat=flat.index_copy(0,x[k],v.to(flat.dtype))
        flat=flat[:n*T]
        bias=flat.new_zeros((n,1,1,T)).masked_fill(~valid[:,None,None,:],float('-inf'))
        keep=x['keep']
        xp=flat.index_select(0,keep)
        for block in self.blocks:xp=block.forward_packed(xp,bias,keep,(n,T))
        return xp.new_zeros((n*T,xp.shape[-1])).index_copy(0,keep,xp).view(n,T,-1),~valid


def action_gate(b):
    # Bomb/active/pocket availability and readiness are observations, not rules.
    x=b['v_body'] if 'v_body' in b else b['v_gate']
    return torch.ones((len(x),3),dtype=torch.bool,device=x.device)

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
from config import ClipperConfig

# ══════════════════════════════════════════════════════════════════════
#  1. Motion Enhancement Module
# ══════════════════════════════════════════════════════════════════════
class MotionEnhancementModule(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.temporal_conv = nn.Conv1d(
            embed_dim, embed_dim,
            kernel_size=3, padding=1,
            groups=embed_dim
        )
        self.gate = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, frame_embeds: torch.Tensor) -> torch.Tensor:
        x      = frame_embeds.transpose(1, 2)                    
        motion = self.temporal_conv(x).transpose(1, 2)           
        gate   = torch.sigmoid(self.gate(frame_embeds))          
        return self.norm(frame_embeds + gate * motion)           

# ══════════════════════════════════════════════════════════════════════
#  2. Text-Guided Excitation Module
# ══════════════════════════════════════════════════════════════════════
class TextGuidedExcitationModule(nn.Module):
    def __init__(self, embed_dim: int, reduction: int = 4):
        super().__init__()
        r = embed_dim // reduction            
        self.text_proj  = nn.Linear(embed_dim, r)
        self.frame_proj = nn.Linear(embed_dim, r)
        self.weight_out = nn.Linear(r, 1)

    def forward(self, frame_embeds: torch.Tensor,
                text_embed: torch.Tensor = None):
        f = self.frame_proj(frame_embeds)                        

        if text_embed is not None:
            t           = self.text_proj(text_embed).unsqueeze(1)  
            interaction = f + t.expand_as(f)                       
        else:
            interaction = f                                        

        weights         = torch.sigmoid(
            self.weight_out(torch.tanh(interaction))
        ).squeeze(-1)                                              
        weighted_frames = frame_embeds * weights.unsqueeze(-1)    
        return weighted_frames, weights

# ══════════════════════════════════════════════════════════════════════
#  3. Softmax Aggregation Module
# ══════════════════════════════════════════════════════════════════════
class SoftmaxAggregationModule(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.attn_proj = nn.Linear(embed_dim, 1)

    def forward(self, weighted_frames: torch.Tensor) -> torch.Tensor:
        scores = self.attn_proj(weighted_frames).squeeze(-1)     
        attn   = F.softmax(scores, dim=-1)                       
        return (weighted_frames * attn.unsqueeze(-1)).sum(1)     

# ══════════════════════════════════════════════════════════════════════
#  4. Query Module
# ══════════════════════════════════════════════════════════════════════
class QueryModule(nn.Module):
    def __init__(self, embed_dim: int, num_queries: int = 8,
                 num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.num_queries = num_queries
        self.embed_dim   = embed_dim

        self.slots = nn.Parameter(torch.empty(num_queries, embed_dim))
        nn.init.xavier_uniform_(self.slots.unsqueeze(0))

        self.cross_attn = nn.MultiheadAttention(
            embed_dim   = embed_dim,
            num_heads   = num_heads,
            dropout     = dropout,
            batch_first = True
        )

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, text_embed: torch.Tensor) -> torch.Tensor:
        B = text_embed.shape[0]
        queries = self.slots.unsqueeze(0).expand(B, -1, -1)   
        kv = text_embed.unsqueeze(1)                             
        attn_out, _ = self.cross_attn(
            query = queries,
            key   = kv,
            value = kv
        )                                                      
        x = self.norm1(attn_out + queries)                     
        x = self.norm2(x + self.ffn(x))                        
        return x

# ══════════════════════════════════════════════════════════════════════
#  5. Similarity Module
# ══════════════════════════════════════════════════════════════════════
class SimilarityModule(nn.Module):
    def __init__(self, alpha: float = 0.7):
        super().__init__()
        self.alpha = alpha
        self.logit_scale = nn.Parameter(
            torch.ones([]) * math.log(1.0 / 0.07)
        )

    def global_similarity(self, text_embed: torch.Tensor,
                          clip_embed: torch.Tensor) -> torch.Tensor:
        scale = self.logit_scale.exp().clamp(max=100.0)
        return scale * (text_embed @ clip_embed.T)

    def local_similarity(self, concept_embeds: torch.Tensor,
                         frame_embeds: torch.Tensor) -> torch.Tensor:
        scale = self.logit_scale.exp().clamp(max=100.0)
        nc = F.normalize(concept_embeds.float(), dim=-1)  
        nf = F.normalize(frame_embeds.float(),   dim=-1)  
        scores = torch.einsum('tqd, vfd -> tvqf', nc, nf) 
        max_scores = scores.max(dim=-1).values             
        local_sim  = max_scores.mean(dim=-1)               
        return scale * local_sim

    def forward(self, text_embed: torch.Tensor,
                clip_embed: torch.Tensor,
                concept_embeds: torch.Tensor = None,
                frame_embeds:   torch.Tensor = None) -> torch.Tensor:
        g_sim = self.global_similarity(text_embed, clip_embed)

        if (concept_embeds is None
                or frame_embeds is None
                or self.alpha == 1.0):
            return g_sim

        l_sim = self.local_similarity(concept_embeds, frame_embeds)
        return self.alpha * g_sim + (1.0 - self.alpha) * l_sim

# ══════════════════════════════════════════════════════════════════════
#  MAIN CLIPPER MODEL
# ══════════════════════════════════════════════════════════════════════
class ClipperModel(nn.Module):
    def __init__(self, config: ClipperConfig):
        super().__init__()
        self.config = config

        # Frozen CLIP backbone
        self.clip_model, _, self.preprocess = \
            open_clip.create_model_and_transforms(
                config.clip_model, pretrained="openai"
            )
        self.tokenizer = open_clip.get_tokenizer(config.clip_model)

        for p in self.clip_model.parameters():
            p.requires_grad = False
        self.clip_model.eval()

        # Trainable custom modules
        D = config.embed_dim

        self.motion      = MotionEnhancementModule(D)
        self.excitation  = TextGuidedExcitationModule(
            D, config.excitation_reduction
        )
        self.aggregation = SoftmaxAggregationModule(D)
        self.query_mod   = QueryModule(
            D,
            num_queries = config.num_query_vectors,
            num_heads   = config.num_attn_heads,
            dropout     = config.query_dropout
        )
        self.similarity  = SimilarityModule(
            alpha = config.similarity_alpha
        )

    def encode_text(self, tokens: torch.Tensor):
        with torch.no_grad():
            raw = self.clip_model.encode_text(tokens).float()  

        if self.config.use_custom_modules:
            concept_embeds = self.query_mod(raw)               
        else:
            concept_embeds = raw.unsqueeze(1).expand(
                -1, self.config.num_query_vectors, -1
            ).clone()

        text_embed     = F.normalize(raw,             dim=-1)
        concept_embeds = F.normalize(concept_embeds, dim=-1)
        return text_embed, concept_embeds

    def encode_video(self, frames: torch.Tensor,
                     text_embed: torch.Tensor = None):
        B, T, C, H, W = frames.shape

        with torch.no_grad():
            raw = self.clip_model.encode_image(
                frames.view(B * T, C, H, W)
            ).float()                                          
        frame_embeds = raw.view(B, T, -1)                      

        if self.config.use_custom_modules:
            frame_embeds = self.motion(frame_embeds)
            frame_embeds, _ = self.excitation(frame_embeds, text_embed)
            clip_embed = self.aggregation(frame_embeds)
        else:
            clip_embed = frame_embeds.mean(dim=1)              

        clip_embed = F.normalize(clip_embed, dim=-1)
        return clip_embed, frame_embeds

    def forward(self, frames: torch.Tensor, tokens: torch.Tensor):
        text_embed, concept_embeds = self.encode_text(tokens)
        clip_embed, frame_embeds   = self.encode_video(frames, text_embed)

        sim = self.similarity(
            text_embed, clip_embed, concept_embeds, frame_embeds
        )                                                      

        B      = sim.shape[0]
        labels = torch.arange(B, device=sim.device)

        loss = (
            F.cross_entropy(sim,   labels) +
            F.cross_entropy(sim.T, labels)
        ) / 2.0

        return {
            "loss":        loss,
            "similarity":  sim,
            "logit_scale": self.similarity.logit_scale.exp().item()
        }
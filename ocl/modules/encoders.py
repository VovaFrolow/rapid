from typing import Any, Dict, List, Optional, Union

import einops
import timm
import torch
import torchvision
from torch import nn

from ocl.modules import utils
from ocl.utils import config_as_kwargs, make_build_fn


@make_build_fn(__name__, "encoder")
def build(config, name: str):
    if name == "FrameEncoder":
        pos_embed = None
        if config.get("pos_embed"):
            pos_embed = utils.build_module(config.pos_embed)

        output_transform = None
        if config.get("output_transform").name == "networks.two_layer_mlp":
            output_transform = utils.build_module(config.output_transform)
        else:
            output_transform = config.get("output_transform")

        return FrameEncoder(
            backbone=utils.build_module(config.backbone, default_group="encoders"),
            pos_embed=pos_embed,
            output_transform=output_transform,
            **config_as_kwargs(config, ("backbone", "pos_embed", "output_transform")),
        )
    elif name == "HCEncoder":
        pos_embed = None
        if config.get("pos_embed"):
            pos_embed = utils.build_module(config.pos_embed)
        output_transform = None
        return HCEncoder(
            backbone=utils.build_module(config.backbone, default_group="encoders"),
            pos_embed=pos_embed,
            output_transform=output_transform,
            **config_as_kwargs(config, ("backbone", "pos_embed", "output_transform")),
        )
    else:
        return None


class FrameEncoder(nn.Module):
    """Module reducing image to set of features."""

    def __init__(
        self,
        backbone: nn.Module,
        pos_embed: Optional[nn.Module] = None,
        output_transform: Optional[nn.Module] = None,
        spatial_flatten: bool = False,
        main_features_key: str = "vit_block12",
    ):
        super().__init__()
        self.backbone = backbone
        self.pos_embed = pos_embed
        self.output_transform = output_transform
        self.spatial_flatten = spatial_flatten
        self.main_features_key = main_features_key

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        # images: batch x n_channels x height x width
        backbone_features = self.backbone(images)
        if isinstance(backbone_features, dict):
            features = backbone_features[self.main_features_key].clone()
        else:
            features = backbone_features.clone()

        if self.pos_embed:
            features = self.pos_embed(features)

        if self.spatial_flatten:
            features = einops.rearrange(features, "b c h w -> b (h w) c")
        if isinstance(self.output_transform, torch.nn.Module):
            features = self.output_transform(features)

        assert (
            features.ndim == 3
        ), f"Expect output shape (batch, tokens, dims), but got {features.shape}"
        if isinstance(backbone_features, dict):
            for k, backbone_feature in backbone_features.items():
                if self.spatial_flatten:
                    backbone_features[k] = einops.rearrange(backbone_feature, "b c h w -> b (h w) c")
                assert (
                    backbone_feature.ndim == 3
                ), f"Expect output shape (batch, tokens, dims), but got {backbone_feature.shape}"
            main_backbone_features = backbone_features[self.main_features_key]

            return {
                "features": features,
                "backbone_features": main_backbone_features,
                **backbone_features,
            }
        else:
            if self.spatial_flatten:
                backbone_features = einops.rearrange(backbone_features, "b c h w -> b (h w) c")
            assert (
                backbone_features.ndim == 3
            ), f"Expect output shape (batch, tokens, dims), but got {backbone_features.shape}"

            return {
                "features": features,
                "backbone_features": backbone_features,
            }


class TimmExtractor(nn.Module):
    """Feature extractor utilizing models from timm library."""

    # Convenience aliases for feature keys
    FEATURE_ALIASES = {
        **{f"resnet_block{i}": f"layer{i}" for i in range(1, 5)},
        **{f"vit_block{i + 1}": f"blocks.{i}" for i in range(12)},
        **{f"vit_block_values{i + 1}": f"blocks.{i}.attn.qkv" for i in range(12)},
        **{f"vit_block_queries{i + 1}": f"blocks.{i}.attn.qkv" for i in range(12)},
        **{f"vit_block_keys{i + 1}": f"blocks.{i}.attn.qkv" for i in range(12)},
        "vit_output": "norm",
    }
    FEATURE_MAPPING = {
        **{f"layer{i}": f"resnet_block{i}" for i in range(1, 5)},
        **{f"blocks.{i}": f"vit_block{i + 1}" for i in range(12)},
        **{f"blocks.{i}.attn.qkv": f"vit_block_keys{i + 1}" for i in range(12)},
        "norm": "vit_output",
    }

    def __init__(
        self,
        model: str,
        pretrained: bool = False,
        frozen: bool = False,
        features: Optional[Union[str, List[str]]] = None,
        checkpoint_path: Optional[str] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        model_name = model
        self.frozen = frozen
        self.features = [features] if isinstance(features, str) else features
        self.is_vit = model_name.startswith("vit")

        model = TimmExtractor._create_model(model_name, pretrained, checkpoint_path, model_kwargs)

        if self.features is not None:
            nodes = torchvision.models.feature_extraction.get_graph_node_names(model)[0]

            features = []
            for name in self.features:
                if name in TimmExtractor.FEATURE_ALIASES:
                    name = TimmExtractor.FEATURE_ALIASES[name]

                if not any(node.startswith(name) for node in nodes):
                    raise ValueError(
                        f"Requested features under node {name}, but this node does "
                        f"not exist in model {model_name}. Available nodes: {nodes}"
                    )

                features.append(name)

            model = torchvision.models.feature_extraction.create_feature_extractor(model, features)

        self.model = model

        if self.frozen:
            self.requires_grad_(False)

    @staticmethod
    def _create_model(
        model_name: str,
        pretrained: bool,
        checkpoint_path: Optional[str],
        model_kwargs: Optional[Dict[str, Any]],
        trials: int = 0,
    ) -> nn.Module:
        if model_kwargs is None:
            model_kwargs = {}

        try:
            model = timm.create_model(
                model_name, pretrained=pretrained, checkpoint_path=checkpoint_path, **model_kwargs
            )
        except (FileExistsError, FileNotFoundError):
            # Timm uses Hugginface hub for loading the files, which does some symlinking in the
            # background when loading the checkpoint. When multiple concurrent jobs attempt to
            # load the checkpoint, this can create conflicts, because the symlink is first removed,
            # then created again by each job. We attempt to catch the resulting errors here, and
            # retry creating the model, up to 3 times.
            if trials == 2:
                raise
            else:
                model = None

        if model is None:
            model = TimmExtractor._create_model(
                model_name, pretrained, checkpoint_path, model_kwargs, trials=trials + 1
            )

        return model

    def forward(self, inp):
        if self.frozen:
            with torch.no_grad():
                outputs = self.model(inp)
        else:
            outputs = self.model(inp)

        if self.features is not None:
            if self.is_vit:
                outputs = {k: v[:, 1:] for k, v in outputs.items()}  # Remove CLS token
            outputs = {self.FEATURE_MAPPING[key]: value for key, value in outputs.items()}
            for name in self.features:
                if ("keys" in name) or ("queries" in name) or ("values" in name):
                    feature_name = name.replace("queries", "keys").replace("values", "keys")
                    B, N, C = outputs[feature_name].shape
                    qkv = outputs[feature_name].reshape(
                        B, N, 3, C // 3
                    )  # outp has shape B, N, 3 * H * (C // H)
                    q, k, v = qkv.unbind(2)
                    if "keys" in name:
                        outputs[name] = k
                    elif "queries" in name:
                        outputs[name] = q
                    elif "values" in name:
                        outputs[name] = v
                    else:
                        raise ValueError(f"Unknown feature name {name}.")

            if len(outputs) == 1:
                # Unpack single output for now
                return next(iter(outputs.values()))
            else:
                return outputs
        else:
            return outputs

class HCEncoder(nn.Module):
    """Module reducing image to set of features."""

    def __init__(
        self,
        backbone: nn.Module,
        pos_embed: Optional[nn.Module] = None,
        output_transform: Optional[nn.Module] = None,
        spatial_flatten: bool = False,
        main_features_key: str = "vit_block12",
    ):
        super().__init__()
        self.backbone = backbone
        self.pos_embed = pos_embed
        self.output_transform = output_transform
        self.spatial_flatten = spatial_flatten
        self.main_features_key = main_features_key

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        # images: batch x n_channels x height x width
        backbone_features = self.backbone(images)
        # if isinstance(backbone_features, dict):
        features = backbone_features[f'{self.main_features_key}_hc'].clone()
        # else:
        #     features = backbone_features.clone()
        if self.pos_embed:
            features = self.pos_embed(features)

        if self.spatial_flatten:
            features = einops.rearrange(features, "b c h w -> b (h w) c")
        if self.output_transform:
            features = self.output_transform(features)

        assert (
            features.ndim == 3
        ), f"Expect output shape (batch, tokens, dims), but got {features.shape}"
        if isinstance(backbone_features, dict):
            for k, backbone_feature in backbone_features.items():
                if self.spatial_flatten:
                    backbone_features[k] = einops.rearrange(backbone_feature, "b c h w -> b (h w) c")
                assert (
                    backbone_feature.ndim == 3
                ), f"Expect output shape (batch, tokens, dims), but got {backbone_feature.shape}"
            # main_backbone_features = backbone_features[self.main_features_key]
            main_backbone_features = backbone_features[self.main_features_key]

            if "image_kk" in backbone_features.keys():
                return {
                    "features": features,
                    "backbone_features": main_backbone_features,
                    "adjacency": backbone_features["image_kk"],
                    **backbone_features,
                }
            else:
                return {
                    "features": features,
                    "backbone_features": main_backbone_features,
                    **backbone_features,
                }
        else:
            if self.spatial_flatten:
                backbone_features = einops.rearrange(backbone_features, "b c h w -> b (h w) c")
            assert (
                backbone_features.ndim == 3
            ), f"Expect output shape (batch, tokens, dims), but got {backbone_features.shape}"

            return {
                "features": features,
                "backbone_features": backbone_features,
            }


class TimmExtractorv2(nn.Module):
    """Feature extractor utilizing models from timm library."""

    # Convenience aliases for feature keys
    FEATURE_ALIASESv2 = {
        **{f"resnet_block{i}": f"layer{i}" for i in range(1, 5)},
        **{f"vit_block{j + 1}": [f"blocks.{i}" for i in range(12)] for j in range(12)},
        **{f"vit_block_values{i + 1}": f"blocks.{i}.attn.qkv" for i in range(12)},
        **{f"vit_block_queries{i + 1}": f"blocks.{i}.attn.qkv" for i in range(12)},
        **{f"vit_block_keys{j + 1}": [f"blocks.{i}.attn.qkv" for i in range(12)] for j in range(12)},
        **{f"vit_block_attn{j + 1}": [f"blocks.{i}.attn.proj_drop" for i in range(12)] for j in range(12)},
        "vit_output": "norm",
    }
    FEATURE_MAPPINGv2 = {
        **{f"layer{i}": f"resnet_block{i}" for i in range(1, 5)},
        **{f"blocks.{i}": f"vit_block{i + 1}" for i in range(12)},
        **{f"blocks.{i}.attn.qkv": f"vit_block_keys{i + 1}" for i in range(12)},
        **{f"blocks.{i}.attn.proj_drop": f"vit_block_attn{i + 1}" for i in range(12)},
        "norm": "vit_output",
    }

    def __init__(
        self,
        model: str,
        pretrained: bool = False,
        frozen: bool = False,
        drop: bool = True,
        dim: int = 256,
        mode: str = "default",
        proj_type: str = None,
        features: Optional[Union[str, List[str]]] = None,
        checkpoint_path: Optional[str] = None,
        last_blocks: int = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        model_name = model
        self.frozen = frozen
        self.last_blocks = last_blocks
        self.dim = dim
        self.drop = drop
        self.mode = mode
        self.features = [features] if isinstance(features, str) else features
        self.is_vit = model_name.startswith("vit")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = TimmExtractorv2._create_model(model_name, pretrained, checkpoint_path, model_kwargs).to(device)
        if self.features is not None:
            nodes = torchvision.models.feature_extraction.get_graph_node_names(model)[0]
            features = {}
            for name in self.features:
                if name in TimmExtractorv2.FEATURE_ALIASESv2:
                    name = TimmExtractorv2.FEATURE_ALIASESv2[name]
                if type(name) is not str:
                    for n in name:
                        if not any(node.startswith(n) for node in nodes):
                            raise ValueError(
                                f"Requested features under node {n}, but this node does "
                                f"not exist in model {model_name}. Available nodes: {nodes}"
                            )
                else:
                    if not any(node.startswith(name) for node in nodes):
                            raise ValueError(
                                f"Requested features under node {name}, but this node does "
                                f"not exist in model {model_name}. Available nodes: {nodes}"
                            )

                # features.append(name)
                for i, n in enumerate(name):
                    if "attn.qkv" in n:
                        features[n] = f"vit_block_keys{i+1}"
                    elif "attn.proj_drop" in n:
                        features[n] = f"vit_block_attn{i+1}"
                    else:
                        features[n] = f"vit_block{i+1}"
            features = {k: v for k, v in features.items() if v in self.features}
            # self.levels = []
            # for feats in features[-last_blocks:]:
            #     level = torchvision.models.feature_extraction.create_feature_extractor(model, feats)
            #     self.levels.append(level)
            model = torchvision.models.feature_extraction.create_feature_extractor(model, features)
        
        self.model = model

        if self.frozen:
            self.requires_grad_(False)
        
        self.dropout = torch.nn.Dropout2d(p=0.1)
        self.n_feats = 768 * last_blocks
        self.cluster1 = self.make_clusterer(self.n_feats, self.dim)
        self.proj_cluster1 = self.make_clusterer(self.n_feats, 768)
        self.proj_type = proj_type
        self.normalization = torch.nn.LayerNorm(self.n_feats)
        if self.proj_type == "nonlinear":
            self.cluster2 = self.make_nonlinear_clusterer(self.n_feats, self.dim)
            self.proj_cluster2 = self.make_nonlinear_clusterer(self.n_feats, 768)
        self.project_head = nn.Linear(self.dim, self.dim)
        self.num_heads = 12

    def make_clusterer(self, in_channels, out):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out, (1, 1))) 

    def make_nonlinear_clusterer(self, in_channels, out):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, in_channels, (1, 1)),
            torch.nn.ReLU(), # torch.nn.GELU()
            torch.nn.Conv2d(in_channels, out, (1, 1)))

    @staticmethod
    def _create_model(
        model_name: str,
        pretrained: bool,
        checkpoint_path: Optional[str],
        model_kwargs: Optional[Dict[str, Any]],
        trials: int = 0,
    ) -> nn.Module:
        if model_kwargs is None:
            model_kwargs = {}

        try:
            model = timm.create_model(
                model_name, pretrained=pretrained, checkpoint_path=checkpoint_path, **model_kwargs
            )
        except (FileExistsError, FileNotFoundError):
            # Timm uses Hugginface hub for loading the files, which does some symlinking in the
            # background when loading the checkpoint. When multiple concurrent jobs attempt to
            # load the checkpoint, this can create conflicts, because the symlink is first removed,
            # then created again by each job. We attempt to catch the resulting errors here, and
            # retry creating the model, up to 3 times.
            if trials == 2:
                raise
            else:
                model = None

        if model is None:
            model = TimmExtractorv2._create_model(
                model_name, pretrained, checkpoint_path, model_kwargs, trials=trials + 1
            )

        return model

    def forward(self, inp):
        # outputs = []
        # for level in self.levels:
        #     if self.frozen:
        #         with torch.no_grad():
        #             outputs.append(level(inp))
        #     else:
        #         outputs.append(level(inp))
        if self.frozen:
            with torch.no_grad():
                outputs = self.model(inp)
        else:
            outputs = self.model(inp)

        if self.is_vit:
            outputs = {k: v[:, 1:] for k, v in outputs.items()}  # Remove CLS token
        if self.mode == "image_kk":
            feature_keys = []
            qkv_keys = []
            attn_keys = []
            for key in outputs.keys():
                if f"keys" in key:
                    qkv_keys.append(key)
                elif f"attn" in key:
                    attn_keys.append(key)
                else:
                    feature_keys.append(key)
            features = torch.cat([outputs[key] for idx, key in enumerate(feature_keys) if len(feature_keys) - (idx + 1) < self.last_blocks], 
                                dim=2)
            # qkv = [outputs[key][:, 1:, :] for idx, key in enumerate(qkv_keys) if len(qkv_keys) - (idx + 1) < self.last_blocks]
            # attn = [outputs[key][:, 1:, :] for idx, key in enumerate(attn_keys) if len(attn_keys) - (idx + 1) < self.last_blocks]
            # feat_h, feat_w = int(features.shape[1]**0.5), int(features.shape[1]**0.5) 
            # kk = [blk.reshape(features.shape[0], feat_h, feat_w, -1).permute(0, 3, 1, 2) for blk in qkv]
            # B, H, W, D = k[0].shape
            # image_kk = torch.cat(kk, dim=1)
            features = self.normalization(features)
            B, N, C = outputs[feature_keys[-1]][:, :, :].shape
            qkv = [outputs[key].reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) \
                   for idx, key in enumerate(qkv_keys) if len(qkv_keys) - (idx + 1) < self.last_blocks]
            feat_h, feat_w = int(features.shape[1]**0.5), int(features.shape[1]**0.5)
            image_k = [k[1, :, :, :, :].reshape(B, qkv[0].shape[2], feat_h, feat_w, -1) for k in qkv]
            B, H, I, J, D = image_k[0].shape
            image_kk = [k.permute(0, 1, 4, 2, 3).reshape(B, H*D, I, J) for k in image_k]
            image_kk = torch.cat(image_kk, dim=1)
            image_feat = features.reshape(features.shape[0], feat_h, feat_w, -1).permute(0, 3, 1, 2)
            if self.proj_type is not None:
                if self.drop:
                    with torch.no_grad():
                        code = self.cluster1(self.dropout(image_feat))
                    code_kk = self.cluster1(self.dropout(image_kk))
                else:
                    with torch.no_grad():
                        code = self.cluster1(image_feat)
                    code_kk = self.cluster1(image_kk)
                if self.proj_type == "nonlinear":
                    if self.drop:
                        code += self.cluster2(self.dropout(image_feat))
                        code_kk += self.cluster2(self.dropout(image_kk))
                    else:
                        code += self.cluster2(image_feat)
                        code_kk += self.cluster2(image_kk)
            else:
                if self.drop:
                    code = self.dropout(image_feat)
                    code_kk = self.dropout(image_kk)
                else:
                    code = image_feat
                    code_kk = image_kk
            code_kk = code_kk.permute(0, 2, 3, 1).reshape(-1, self.dim)
            code_kk = self.project_head(code_kk)
            outputs = {self.features[0]: code.permute(0, 2, 3, 1).reshape(features.shape[0], feat_h*feat_w, -1),
                       f'{self.features[0]}_last': outputs[list(outputs.keys())[-1]][:, 1:, :],
                       "image_kk": code_kk.reshape(B, feat_h*feat_w, -1)}
            
            return outputs
        elif "hc" in self.mode:
            feature_keys = []
            proj_keys = []
            attn_keys = []
            for key in outputs.keys():
                if f"keys" in key or f"queries" in key or f"values" in key:
                    proj_keys.append(key)
                elif f"attn" in key:
                    attn_keys.append(key)
                else:
                    feature_keys.append(key)
            if self.last_blocks > 1:
                features = torch.cat([outputs[key] for key in feature_keys], dim=2)
            else:
                features = outputs[feature_keys[-1]]
            features = self.normalization(features)
            feat_h, feat_w = int(features.shape[1]**0.5), int(features.shape[1]**0.5) 
            image_feat = features.reshape(features.shape[0], feat_h, feat_w, -1).permute(0, 3, 1, 2)
            if self.proj_type is not None:
                with torch.no_grad():
                    if self.drop:
                        code = self.cluster1(self.dropout(image_feat))
                    else:
                        code = self.cluster1(image_feat)
                if self.proj_type == "nonlinear":
                    if self.drop:
                        code += self.cluster2(self.dropout(image_feat))
                    else:
                        code += self.cluster2(image_feat)
            else:
                if self.drop:
                    code = self.dropout(image_feat)
                else:
                    code = image_feat
            print("123")
            # code = self.project_head(code.flatten(-2, -1))
            if self.mode == "hc_video":
                for name in proj_keys:
                    if ("keys" in name) or ("queries" in name) or ("values" in name):
                        feature_name = name.replace("queries", "keys").replace("values", "keys")
                        B, N, C = outputs[feature_name].shape
                        qkv = outputs[feature_name].reshape(
                            B, N, 3, C // 3
                        )  # outp has shape B, N, 3 * H * (C // H)
                        q, k, v = qkv.unbind(2)
                        if "keys" in name:
                            outputs[name] = k
                        elif "queries" in name:
                            outputs[name] = q
                        elif "values" in name:
                            outputs[name] = v
                        else:
                            raise ValueError(f"Unknown feature name {name}.")
                # if self.last_blocks > 1:
                #     projections = torch.cat([outputs[key] for key in proj_keys], dim=2)
                # else:
                #     projections = outputs[proj_keys[-1]]
                # projections = self.normalization(projections)
                # proj_h, proj_w = int(projections.shape[1]**0.5), int(projections.shape[1]**0.5) 
                # proj = projections.reshape(projections.shape[0], proj_h, proj_w, -1).permute(0, 3, 1, 2)
                # if self.proj_type is not None:
                #     with torch.no_grad():
                #         if self.drop:
                #             code_proj = self.proj_cluster1(self.dropout(proj))
                #         else:
                #             code_proj = self.proj_cluster1(proj)
                #     if self.proj_type == "nonlinear":
                #         if self.drop:
                #             code_proj += self.proj_cluster2(self.dropout(proj))
                #         else:
                #             code_proj += self.proj_cluster2(proj)
                # else:
                #     if self.drop:
                #         code_proj = self.dropout(proj)
                #     else:
                #         code_proj = proj
                return {f"{feature_keys[-1]}_hc": code.permute(0, 2, 3, 1).reshape(features.shape[0], feat_h*feat_w, -1), # code.permute(0, 2, 1),
                        # f"{proj_keys[-1]}_hc": code_proj.permute(0, 2, 3, 1).reshape(code_proj.shape[0], proj_h*proj_w, -1), # outputs[proj_keys[-1]], 
                        feature_keys[-1]: outputs[feature_keys[-1]],
                        proj_keys[-1]: outputs[proj_keys[-1]]}
            
            return {f"{feature_keys[-1]}_hc": code.permute(0, 2, 3, 1).reshape(features.shape[0], feat_h*feat_w, -1), # code.permute(0, 2, 1),
                    feature_keys[-1]: outputs[feature_keys[-1]]}
            
            # outputs = {self.features[0]: features,
            #            f'{self.features[0]}_last': outputs[list(outputs.keys())[-1]][:, 1:, :]}
            # outputs = {self.features[0]: features[:, 1:, :]}
            
            # outputs = {feature_keys[-1]: code.permute(0, 2, 3, 1).reshape(features.shape[0], feat_h*feat_w, -1), # code.permute(0, 2, 1),
            #         f'{self.features[0]}_last': outputs[list(outputs.keys())[-1]][:, 1:, :]}
            
            # return outputs
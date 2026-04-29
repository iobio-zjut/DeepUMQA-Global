import os
import csv
from functools import partial
from os.path import abspath, dirname, exists, join
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

try:
    import pytorch_lightning as pl
except ImportError:  # inference does not require Lightning
    class _LightningDataModule:
        pass

    class _PLNamespace:
        LightningDataModule = _LightningDataModule

    pl = _PLNamespace()


class DecoyDataset(Dataset):
    def __init__(self,
                 targets_fpath='',
                 interface_pdb_base_dir="",
                 feature_root="",
                 interface_fe_base_dir="",
                 structure_profile_dir="",
                 af_orientation_dir="",
                 three_di_base_dir="",
                 voro_base_dir="",
                 mpnn_base_dir="",
                 process_feat=False,
                 max_length=9999,
                 pool_process=20,
                 missing_res_csv=None,
                 infer=False,
                 sample_num=1,
                 **kwargs):
        
        self.samples = []
        self.max_length = max_length
        self.infer = infer
        self.target_models_dict = {}
        self.target_list = []
        self.process_feat = process_feat
        self.sample_num = sample_num

        if process_feat:
            print(
                "[WARN] DecoyDataset.process_feat is deprecated and ignored. "
                "Run structure_rank/utils/pipeline/feature_pipeline.py before inference."
            )

        # 路径赋值
        self.interface_pdb_base_dir = interface_pdb_base_dir
        self.af_orientation_dir = af_orientation_dir
        self.feature_root = self._resolve_root_dir(feature_root, interface_pdb_base_dir)
        self.interface_fe_base_dir = self._resolve_feature_dir(interface_fe_base_dir, "base")
        self.structure_profile_dir = self._resolve_feature_dir(structure_profile_dir, "ori")
        self.three_di_base_dir = self._resolve_feature_dir(three_di_base_dir, "3di")
        self.voro_base_dir = self._resolve_feature_dir(voro_base_dir, "voro")
        self.mpnn_base_dir = self._resolve_feature_dir(mpnn_base_dir, "mpnn")
        self._profile_bundle_cache = {}
        self._parse_targets(targets_fpath, missing_res_csv)

    def _resolve_root_dir(self, feature_root, interface_pdb_base_dir):
        if feature_root:
            return str(Path(feature_root).resolve())
        if interface_pdb_base_dir:
            parent_dir = Path(interface_pdb_base_dir).resolve().parent
            return str((parent_dir / "feature").resolve())
        return ""

    def _resolve_feature_dir(self, explicit_path, expected_name):
        candidates = []
        if explicit_path:
            candidates.append(Path(explicit_path).resolve())
        if self.feature_root:
            candidates.append((Path(self.feature_root) / expected_name).resolve())

        for candidate in candidates:
            direct_hit = candidate / expected_name
            if direct_hit.is_dir():
                return str(direct_hit)
            if candidate.exists() or explicit_path:
                return str(candidate)

        return str(candidates[0]) if candidates else ""

    def _parse_targets(self, targets_fpath, missing_res_csv):
        target_exclude = []
        if missing_res_csv and exists(missing_res_csv):
            with open(missing_res_csv, "r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                target_exclude = [row["pdb"] for row in reader if row.get("pdb")]

        self.interface_fpath_list = []
        with open(targets_fpath, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if not parts[0]: continue
                smp = parts[0]
                s_dockq = parts[1] if len(parts) > 1 else 0
                
                target, model = smp.split("/")[-2], smp.split("/")[-1]
                if target in target_exclude: continue

                if target not in self.target_models_dict:
                    self.target_models_dict[target] = []
                    self.target_list.append(target)
                
                self.target_models_dict[target].append([smp, float(s_dockq)])
                interface_pdb_fpath = join(self.interface_pdb_base_dir, target, model)
                self.interface_fpath_list.append((smp, interface_pdb_fpath))
                self.samples.append([smp, float(s_dockq)])

    def __len__(self):
        return len(self.samples) if self.infer else len(self.target_list)

    def __getitem__(self, idx):
        try:
            return self.load_data_(idx)
        except Exception as e:
            print(f"Error loading idx {idx}: {e}")
            import traceback
            traceback.print_exc()
            return {"error": 0}

    def _resolve_profile_path(self, target, model_base):
        model_profile_path = os.path.join(self.structure_profile_dir, target, model_base + '.npz')
        if os.path.exists(model_profile_path):
            return model_profile_path
        return os.path.join(self.structure_profile_dir, target + '.npz')

    def _load_profile_bundle(self, profile_path):
        cached = self._profile_bundle_cache.get(profile_path)
        if cached is not None:
            return cached

        with np.load(profile_path, encoding='bytes', allow_pickle=True) as profile_data:
            profile = profile_data["profile"].astype(np.float32)
            entropy = profile_data["entropy"].astype(np.float32)
            profile_mask = profile_data["mask"].astype(bool)
            if "orientation" in profile_data.files:
                af_orientation = profile_data["orientation"].astype(np.float32)
            else:
                length = profile.shape[0]
                af_orientation = np.zeros((length, length, 6), dtype=np.float32)

        bundle = {
            "profile": profile,
            "entropy": entropy,
            "mask_3d": np.expand_dims(profile_mask, axis=-1).astype(np.float32),
            "af_orientation": af_orientation,
        }
        self._profile_bundle_cache[profile_path] = bundle
        return bundle

    def load_data_(self, idx):
        # 1. 样本路径与基础解析
        if self.infer: 
            smp_fpath, quality = self.samples[idx] 
        else:
            target_name = self.target_list[idx]
            models = self.target_models_dict[target_name]
            smp_fpath, quality = random.sample(models, self.sample_num)[0]
        
        target = smp_fpath.split("/")[-2]
        model_name = smp_fpath.split("/")[-1]
        model_base = model_name.rsplit('.pdb', 1)[0]
        target_model = "/".join(smp_fpath.split("/")[-2:])

        # --- 路径定义 ---
        # 基础特征 (Base)
        feat_path = join(self.interface_fe_base_dir, target, model_base + '.npz')
        # 整合了 Profile 和 Orientation 的文件
        profile_path = self._resolve_profile_path(target, model_base)
        three_di_path = join(self.three_di_base_dir, target, model_base + '.npz')
        voro_path = join(self.voro_base_dir, target, model_base + '.npz')
        mpnn_path = join(self.mpnn_base_dir, target, model_base + '.npz')

        # 检查核心文件是否存在
        if not os.path.exists(feat_path) or not os.path.exists(profile_path):
            if not os.path.exists(feat_path): print(f"❌ Missing BaseFeat: {feat_path}")
            if not os.path.exists(profile_path): print(f"❌ Missing Profile: {profile_path}")
            return {"name": smp_fpath}
        atomic_feature_paths = {
            "3di": three_di_path,
            "voro": voro_path,
            "mpnn": mpnn_path,
        }
        missing_atomic = [name for name, path in atomic_feature_paths.items() if not os.path.exists(path)]
        if missing_atomic:
            for name in missing_atomic:
                print(f"❌ Missing {name}: {atomic_feature_paths[name]}")
            return {"name": smp_fpath}

        # --- 加载 Profile & Orientation (从同一个文件读取) ---
        target_profile = self._load_profile_bundle(profile_path)
        profile = target_profile["profile"]
        entropy = target_profile["entropy"]
        mask_3d = target_profile["mask_3d"]
        af_orientation = target_profile["af_orientation"]

        # 检查核心文件 (Debug)
        if not os.path.exists(feat_path) or not os.path.exists(profile_path):
            print(f"❌ Missing core files for {target}")
            return {"name": smp_fpath}

        # 2. 加载数据并处理维度 (严格参考你提供的代码逻辑)
        
        # 加载 Base 数据
        with np.load(feat_path, encoding='bytes', allow_pickle=True) as data:
            # 【关键修复】: 按照你提供的逻辑对 feat 进行切片，解决 permute 维度报错
            feat = data["feat"]
            if feat.ndim == 3:
                feat = feat[0, :, :]
            adj = data["adj"]
            obt = data["obt"]
            tbt = data["tbt"]
            vidx = data["idx"]
            val = data["val"]
            prop = data["prop"][:52]
            angles = np.stack([np.sin(data["phi"]), np.cos(data["phi"]),
                               np.sin(data["psi"]), np.cos(data["psi"])], axis=-1)
            orientations = np.stack([data["omega6d"], data["theta6d"], data["phi6d"]], axis=-1)
            orientations = np.concatenate([np.sin(orientations), np.cos(orientations)], axis=-1)
            euler = np.concatenate([np.sin(data["euler"]), np.cos(data["euler"])], axis=-1)
            maps = data["maps"]
        
        if obt.shape[1] > self.max_length:
            return {"name": smp_fpath}

        with np.load(mpnn_path, encoding='bytes', allow_pickle=True) as mpnn_data:
            mpnn = mpnn_data["h_ES_encoder"].astype(np.float32)
        with np.load(three_di_path, encoding='bytes', allow_pickle=True) as three_di_data:
            seq3di = three_di_data["seq3di"].astype(np.float32)
        with np.load(voro_path, encoding='bytes', allow_pickle=True) as voro_data:
            voro_area_raw = voro_data["voro_area"]
            voro_solvent_area_raw = voro_data["voro_solvent_area"]
            voro_normal_raw = voro_data["normal"]

        # Voronoi 几何特征处理
        voro_area = np.sum(voro_area_raw, axis=0, keepdims=True).T
        voro_solvent_area = np.expand_dims(voro_solvent_area_raw, axis=-1)
        
        # 维度对齐填充
        diff = voro_area.shape[0] - voro_solvent_area.shape[0]
        if diff != 0:
            zero_padding = np.zeros((abs(diff), voro_solvent_area.shape[1]))
            voro_solvent_area = np.vstack([voro_solvent_area, zero_padding]) if diff > 0 else np.vstack([voro_solvent_area, zero_padding])
        voro_features = np.concatenate([voro_area, voro_solvent_area], axis=-1).T
        
        # Voronoi Normal 处理: L*L*3 -> (3, L)
        voro_normal = voro_normal_raw
        voro_normal = np.sum(voro_normal, axis=0).T

        # 3. 特征拼接 (1D & 2D)
        _1d = np.concatenate([angles.transpose(1, 0), obt, prop, seq3di], axis=0)

        # 拼接 2D: (tbt_T, maps, euler, orientations)
        _2d = np.concatenate([tbt.transpose(1, 2, 0), maps, euler, orientations], axis=-1)

        # 4. 返回最终样本
        return {
            '_1d': _1d.astype(np.float32),
            'voro_features': voro_features.astype(np.float32),
            'voro_normal': voro_normal.astype(np.float32),
            'profile': profile,
            'entropy': entropy,
            'af_orientation': af_orientation,
            'mask_3d': mask_3d,
            'mpnn': mpnn,
            '_2d': _2d.astype(np.float32),
            'vidx': vidx.astype(np.int32),
            'val': val.astype(np.float32),
            'feat': feat.astype(np.float32),
            'adj': adj.astype(np.int64),
            'quality': quality,
            'name': smp_fpath,
        }

def load_plddt(self,smp_fpath):
        res_score_list = []
        res_score = []
        last_chain = ""
        complex_size = 0
        
        for line in open(smp_fpath[:-4], "r").readlines():
            if line.startswith('ATOM'):
                chain = line[20:22]
                atom = line[12:16].strip()

                if atom == "CA":
                    lddt = float(line[60:66])
                    res_score.append(float(lddt))
                    complex_size +=1    
        return np.array(res_score)[:,None].astype(np.float32)

def dict_multimap(fn, dicts):
    first = dicts[0]
    new_dict = {}
    for k, v in first.items():
        all_v = [d[k] for d in dicts]
        if type(v) is dict:
            new_dict[k] = dict_multimap(fn, all_v)
        else:
            new_dict[k] = fn(all_v)
    return new_dict

class BatchCollator:
    def __call__(self, prots):
        stack_fn = partial(torch.stack, dim=0)
        return dict_multimap(stack_fn, prots)

class ScoreDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super(ScoreDataModule, self).__init__()
        self.config = config
        self.batch_collator = BatchCollator()

    def setup(self, stage=None):
        self.train_dataset = DecoyDataset(targets_fpath=self.config.training_fpath,
        interface_pdb_base_dir = self.config.interface_pdb_base_dir,
        feature_root=getattr(self.config, "feature_root", ""),
        interface_fe_base_dir = self.config.interface_fe_base_dir,
        feat_adj_base_dir = self.config.feat_adj_base_dir,
        three_di_base_dir=getattr(self.config, "three_di_base_dir", ""),
        voro_base_dir=getattr(self.config, "voro_base_dir", ""),
        mpnn_base_dir=getattr(self.config, "mpnn_base_dir", ""),
        structure_profile_dir=self.config.structure_profile_dir,
        process_feat=self.config.process_feat,
        max_length=self.config.max_length,
        loss_type=self.config.loss_type,
        pool_process=self.config.pool_process,
        abag=self.config.abag,
        missing_res_csv=self.config.missing_res_csv
        )
        self.val_dataset = DecoyDataset(targets_fpath=self.config.val_fpath,
        interface_pdb_base_dir = self.config.interface_pdb_base_dir,
        feature_root=getattr(self.config, "feature_root", ""),
        interface_fe_base_dir = self.config.interface_fe_base_dir,
        three_di_base_dir=getattr(self.config, "three_di_base_dir", ""),
        voro_base_dir=getattr(self.config, "voro_base_dir", ""),
        mpnn_base_dir=getattr(self.config, "mpnn_base_dir", ""),
        structure_profile_dir=self.config.structure_profile_dir,
        process_feat=self.config.process_feat,
        max_length=self.config.max_length,
        loss_type=self.config.loss_type,
        pool_process=self.config.pool_process,
        
        )
        
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.train_batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=self.config.num_workers,
            # collate_fn=self.batch_collator,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.train_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.config.num_workers,
        )

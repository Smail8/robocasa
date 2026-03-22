import json
import os

import imageio
import robocasa
import numpy as np
import matplotlib.pyplot as plt
import robosuite
import robosuite.utils.transform_utils as T
from robosuite.controllers import load_composite_controller_config
from robosuite.utils.camera_utils import (
    get_camera_extrinsic_matrix,
    get_camera_intrinsic_matrix,
    get_real_depth_map,
)

class Simulation:
    def __init__(
            self, 
            env_name, 
            robots="PandaOmron", 
            split=None,
            render=False,
            offscreen_render=True,
            camera_obs=True,
            camera_names=None,
            camera_heights=None,
            camera_widths=None,
            ignore_done=True,
            camera_depths=None,
            camera_segmentations=None,
            seed=None,
        ):
        self.env_name = env_name
        self.robots = robots
        self.split = split
        self.render = render
        self.offscreen_render = offscreen_render
        self.camera_obs = camera_obs
        self.camera_names = camera_names
        self.camera_heights = camera_heights
        self.camera_widths = camera_widths
        self.ignore_done = ignore_done
        self.camera_depths = camera_depths
        self.camera_segmentations = camera_segmentations
        self.seed = seed
        self.env = None
        self.setup_environment()

    def _normalize_env_name(self):
        if self.env_name.startswith("robocasa/"):
            return self.env_name.split("/", 1)[1]
        return self.env_name

    def _build_env_kwargs(self):
        controller_config = (
            load_composite_controller_config(
                controller=None,
                robot=self.robots if isinstance(self.robots, str) else self.robots[0],
            )
            if self.robots
            else None
        )

        obj_instance_split = None
        layout_and_style_ids = None
        layout_ids = None
        style_ids = None
        if self.split == "target":
            obj_instance_split = "target"
            layout_and_style_ids = list(zip(range(1, 11), range(1, 11)))
        elif self.split == "pretrain":
            obj_instance_split = "pretrain"
            layout_ids = -2
            style_ids = -2
        elif self.split == "all":
            layout_ids = -3
            style_ids = -3
        elif self.split is not None:
            raise ValueError('split must be either {None, "all", "pretrain", "target"}')

        env_kwargs = dict(
            env_name=self._normalize_env_name(),
            robots=self.robots,
            controller_configs=controller_config,
            camera_names=self.camera_names,
            camera_heights=self.camera_heights,
            camera_widths=self.camera_widths,
            has_renderer=self.render,
            has_offscreen_renderer=self.offscreen_render,
            ignore_done=self.ignore_done,
            use_object_obs=True,
            use_camera_obs=self.camera_obs,
            camera_depths=(self.camera_depths if self.camera_depths is not None else False),
            seed=self.seed,
            obj_instance_split=obj_instance_split,
            layout_and_style_ids=layout_and_style_ids,
            layout_ids=layout_ids,
            style_ids=style_ids,
        )
        if self.camera_segmentations is not None:
            env_kwargs["camera_segmentations"] = self.camera_segmentations
        return env_kwargs

    def _make_env(self):
        env_kwargs = self._build_env_kwargs()
        try:
            self.env = robosuite.make(**env_kwargs)
        except TypeError as exc:
            if "camera_segmentations" not in str(exc):
                raise
            env_kwargs.pop("camera_segmentations", None)
            self.env = robosuite.make(**env_kwargs)

    def setup_environment(self):
        self._make_env()

    def _get_body_pose_in_world(self, body_name):
        body_id = self.env.sim.model.body_name2id(body_name)
        pos_world = np.array(self.env.sim.data.body_xpos[body_id])
        quat_world_xyzw = T.convert_quat(
            np.array(self.env.sim.data.body_xquat[body_id]), to="xyzw"
        )
        quat_world_wxyz = T.convert_quat(quat_world_xyzw, to="wxyz")
        return {
            "position": pos_world.tolist(),
            "orientation_wxyz": quat_world_wxyz.tolist(),
        }

    def _get_object_cfg_map(self):
        object_cfg_map = {}
        for cfg in getattr(self.env, "object_cfgs", []):
            alias = cfg.get("name")
            mjcf_path = cfg.get("info", {}).get("mjcf_path")
            if alias is None or mjcf_path is None:
                continue
            object_cfg_map[alias] = {
                "asset_name": os.path.basename(os.path.dirname(mjcf_path)),
                "mjcf_path": self._to_relative_model_path(mjcf_path),
            }
        print(f"Object config map: {object_cfg_map}")
        return object_cfg_map

    def _get_models_root(self):
        return os.path.join(os.path.dirname(robocasa.__file__), "models")

    def _to_relative_model_path(self, path):
        if path is None:
            return None

        abs_path = os.path.abspath(path)
        models_root = os.path.abspath(self._get_models_root())
        try:
            return os.path.relpath(abs_path, models_root)
        except ValueError:
            return abs_path

    def _get_fixture_cfg_map(self):
        fixture_cfg_map = {}
        for name, model in getattr(self.env, "fixtures", {}).items():
            mjcf_path = getattr(model, "file", None)
            fixture_cfg_map[name] = {
                "asset_name": (
                    os.path.basename(os.path.dirname(mjcf_path)) if mjcf_path is not None else None
                ),
                "mjcf_path": self._to_relative_model_path(mjcf_path),
            }
        return fixture_cfg_map

    def _get_camera_height_width(self, camera_idx):
        heights = self.env._input2list(self.camera_heights, len(self.camera_names))
        widths = self.env._input2list(self.camera_widths, len(self.camera_names))
        return int(heights[camera_idx]), int(widths[camera_idx])

    def _get_camera_segmentation_types(self, camera_idx):
        if self.camera_segmentations is None:
            return []

        segs = self.camera_segmentations
        if isinstance(segs, str):
            return [segs]

        if not isinstance(segs, (list, tuple)):
            return [segs]

        if len(segs) == 0:
            return []

        if all(isinstance(seg, str) for seg in segs):
            if self.camera_names is not None and len(segs) == len(self.camera_names):
                return [segs[camera_idx]]
            return list(segs)

        if camera_idx >= len(segs) or segs[camera_idx] is None:
            return []

        camera_segs = segs[camera_idx]
        if isinstance(camera_segs, str):
            return [camera_segs]
        if isinstance(camera_segs, (list, tuple)):
            return list(camera_segs)
        return [camera_segs]

    def _install_camera_segmentations(self):
        if self.camera_segmentations is None:
            return

        num_cameras = len(self.camera_names) if self.camera_names is not None else len(getattr(self.env, "camera_names", []))
        if num_cameras <= 0:
            return

        segs = self.camera_segmentations
        if isinstance(segs, str):
            camera_segmentations = [[segs] for _ in range(num_cameras)]
        elif isinstance(segs, (list, tuple)):
            if len(segs) == 0:
                camera_segmentations = [None for _ in range(num_cameras)]
            elif all(isinstance(seg, str) for seg in segs):
                # If length matches #cameras, interpret as one segmentation type per camera.
                # Otherwise interpret as a shared list of segmentation types for all cameras.
                if len(segs) == num_cameras:
                    camera_segmentations = [[seg] if seg is not None else None for seg in segs]
                else:
                    shared = [seg for seg in segs if seg is not None]
                    camera_segmentations = [list(shared) for _ in range(num_cameras)]
            else:
                camera_segmentations = []
                for i in range(num_cameras):
                    seg = segs[i] if i < len(segs) else None
                    if seg is None:
                        camera_segmentations.append(None)
                    elif isinstance(seg, str):
                        camera_segmentations.append([seg])
                    elif isinstance(seg, (list, tuple)):
                        camera_segmentations.append(list(seg))
                    else:
                        camera_segmentations.append([seg])
        else:
            camera_segmentations = [[segs] for _ in range(num_cameras)]

        self.env.camera_segmentations = camera_segmentations
        self._refresh_segmentation_registry()
        self.env._observables = self.env._setup_observables()

    def _refresh_segmentation_registry(self):
        if self.camera_segmentations is None or not hasattr(self.env, "model"):
            return

        if not hasattr(self.env.model, "mujoco_objects"):
            return

        existing_names = {obj.name for obj in self.env.model.mujoco_objects if hasattr(obj, "name")}
        for obj in getattr(self.env, "objects", {}).values():
            if obj.name not in existing_names:
                self.env.model.mujoco_objects.append(obj)
                existing_names.add(obj.name)

        if hasattr(self.env, "sim") and self.env.sim is not None:
            self.env.model.generate_id_mappings(sim=self.env.sim)

    def get_observations(self):
        self._refresh_segmentation_registry()
        obs = self.env.reset()
        if self.camera_segmentations is not None:
            # Reconfigure camera segmentation observables only after reset, once robot observables are initialized.
            self._install_camera_segmentations()
            obs = self.env._get_observations(force_update=True)
        images = [obs[f"{cam}_image"] for cam in self.camera_names]
        depth_maps = [get_real_depth_map(self.env.sim, obs[f"{v}_depth"]) for v in self.camera_names]
        segmentations = {}
        for i, cam in enumerate(self.camera_names):
            seg_types = self._get_camera_segmentation_types(i)
            if not seg_types:
                continue
            cam_segmentations = {}
            for seg_type in seg_types:
                obs_key = f"{cam}_segmentation_{seg_type}"
                if obs_key in obs:
                    cam_segmentations[seg_type] = obs[obs_key]
            if cam_segmentations:
                segmentations[cam] = cam_segmentations
        if segmentations:
            return images, depth_maps, segmentations
        return images, depth_maps

    def get_instance_id_to_name_map(self, include_background=False, objects_only=False, include_asset_name=False):
        self._refresh_segmentation_registry()

        if not hasattr(self.env, "model") or not hasattr(self.env.model, "instances_to_ids"):
            return {}

        instance_names = list(self.env.model.instances_to_ids.keys())
        instance_id_to_name = {i + 1: name for i, name in enumerate(instance_names)}
        entity_cfg_map = {}
        entity_cfg_map.update(self._get_object_cfg_map())
        entity_cfg_map.update(self._get_fixture_cfg_map())

        if objects_only:
            object_names = set(getattr(self.env, "objects", {}).keys())
            instance_id_to_name = {
                instance_id: name
                for instance_id, name in instance_id_to_name.items()
                if name in object_names
            }

        if include_background:
            instance_id_to_name[0] = "background"

        if include_asset_name:
            instance_id_to_name = {
                instance_id: {
                    "name": name,
                    "asset_name": entity_cfg_map.get(name, {}).get("asset_name"),
                }
                for instance_id, name in instance_id_to_name.items()
            }

        return instance_id_to_name

    def _get_entity_instance_id_map(self):
        instance_id_to_name = self.get_instance_id_to_name_map(include_background=False)
        entity_names = set(getattr(self.env, "objects", {}).keys()) | set(getattr(self.env, "fixtures", {}).keys())
        return {
            name: instance_id
            for instance_id, name in instance_id_to_name.items()
            if name in entity_names
        }

    def _get_visible_body_names(self, segmentations):
        if not segmentations:
            return None

        instance_id_to_name = self.get_instance_id_to_name_map(
            include_background=False,
        )
        if not instance_id_to_name:
            return None

        visible_body_names = set()
        for camera_segmentations in segmentations.values():
            if "instance" not in camera_segmentations:
                continue

            seg = np.squeeze(camera_segmentations["instance"])
            for label in np.unique(seg):
                if label <= 0:
                    continue
                body_name = instance_id_to_name.get(int(label))
                if body_name is not None:
                    visible_body_names.add(body_name)

        return visible_body_names

    def _build_entity_metadata(self, entities, cfg_map, visible_names=None):
        instance_id_map = self._get_entity_instance_id_map()
        metadata = {}

        for name, model in entities.items():
            if visible_names is not None and name not in visible_names:
                continue

            pose = self._get_body_pose_in_world(model.root_body)
            cfg = cfg_map.get(name, {})
            metadata[name] = {
                "name": name,
                "instance_id": instance_id_map.get(name),
                "position": pose["position"],
                "orientation_wxyz": pose["orientation_wxyz"],
                "asset_name": cfg.get("asset_name"),
                "mjcf_path": cfg.get("mjcf_path"),
            }

        return metadata
    
    def get_metadata(self, segmentations=None):
        visible_names = self._get_visible_body_names(segmentations)
        object_cfg_map = self._get_object_cfg_map()
        fixture_cfg_map = self._get_fixture_cfg_map()
        image_height, image_width = self._get_camera_height_width(0)

        metadata = {
            "env_name": self.env_name,
            "world_frame": "world",
            "image_size": {"height": image_height, "width": image_width},
            "near": float(self.env.sim.model.vis.map.znear * self.env.sim.model.stat.extent),
            "far": float(self.env.sim.model.vis.map.zfar * self.env.sim.model.stat.extent),
            "cameras": {},
            "objects": self._build_entity_metadata(
                self.env.objects,
                object_cfg_map,
                visible_names=visible_names,
            ),
            "fixtures": self._build_entity_metadata(
                self.env.fixtures,
                fixture_cfg_map,
                visible_names=visible_names,
            ),
        }

        for i, cam in enumerate(self.camera_names):
            cam_height, cam_width = self._get_camera_height_width(i)
            K = get_camera_intrinsic_matrix(self.env.sim, cam, cam_height, cam_width)
            T_w_cam = get_camera_extrinsic_matrix(self.env.sim, cam)
            quat_w_cam_xyzw = T.mat2quat(T_w_cam[:3, :3])
            quat_w_cam_wxyz = T.convert_quat(quat_w_cam_xyzw, to="wxyz")

            metadata["cameras"][cam] = {
                "height": cam_height,
                "width": cam_width,
                "intrinsics": {
                    "fx": float(K[0, 0]),
                    "fy": float(K[1, 1]),
                    "cx": float(K[0, 2]),
                    "cy": float(K[1, 2]),
                    "matrix": K.tolist(),
                },
                "position": T_w_cam[:3, 3].tolist(),
                "orientation_wxyz": quat_w_cam_wxyz.tolist(),
            }

        return metadata

    def save(self, save_dir, images=None, depth_maps=None, segmentations=None, metadata=None):
        if images is None or depth_maps is None:
            observations = self.get_observations()
            if len(observations) == 3:
                images, depth_maps, observed_segmentations = observations
            else:
                images, depth_maps = observations
                observed_segmentations = {}

            if segmentations is None:
                segmentations = observed_segmentations

        if segmentations is None:
            segmentations = {}

        if metadata is None:
            metadata = self.get_metadata(segmentations=segmentations)

        os.makedirs(save_dir, exist_ok=True)

        for cam, img, depth in zip(self.camera_names, images, depth_maps):
            cam_dir = os.path.join(save_dir, cam)
            os.makedirs(cam_dir, exist_ok=True)
            imageio.imwrite(os.path.join(cam_dir, "rgb.png"), img[::-1, ...])
            np.save(os.path.join(cam_dir, "depth.npy"), depth[::-1, ...])

            for seg_type, seg in segmentations.get(cam, {}).items():
                np.save(
                    os.path.join(cam_dir, f"segmentation_{seg_type}.npy"),
                    seg[::-1, ...],
                )

        with open(os.path.join(save_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=4)

        return metadata
    
    def visualize_rgb_depth(self, images, depth_maps):
        plt.figure(figsize=(12, 6))
        for i, (img, depth) in enumerate(zip(images, depth_maps)):
            plt.subplot(2, len(images), i + 1)
            plt.imshow(img, origin="lower")
            plt.title(f"{self.camera_names[i]} RGB")
            plt.axis("off")

            plt.subplot(2, len(images), len(images) + i + 1)
            plt.imshow(depth, cmap="viridis", origin="lower")
            plt.title(f"{self.camera_names[i]} Depth")
            plt.axis("off")
        plt.tight_layout()
        plt.show()

    def visualize_segmentation_masks(self, images, segmentations, view_id, segmentation_type=None):
        if isinstance(view_id, int):
            cam_idx = view_id
            cam_name = self.camera_names[view_id]
        else:
            cam_name = view_id
            cam_idx = self.camera_names.index(cam_name)

        if cam_name not in segmentations:
            raise KeyError(f"No segmentation found for camera '{cam_name}'")

        if isinstance(images, dict):
            if cam_name not in images:
                raise KeyError(f"No RGB image found for camera '{cam_name}'")
            rgb = np.array(images[cam_name]).copy()
        else:
            rgb = np.array(images[cam_idx]).copy()

        camera_segmentations = segmentations[cam_name]
        if not camera_segmentations:
            raise ValueError(f"Camera '{cam_name}' has no segmentation entries")

        if segmentation_type is None:
            if len(camera_segmentations) != 1:
                raise ValueError(
                    f"Camera '{cam_name}' has multiple segmentation types: {list(camera_segmentations.keys())}. "
                    "Specify segmentation_type."
                )
            segmentation_type = next(iter(camera_segmentations))

        if segmentation_type not in camera_segmentations:
            raise KeyError(
                f"Segmentation type '{segmentation_type}' not found for camera '{cam_name}'. "
                f"Available types: {list(camera_segmentations.keys())}"
            )

        seg = np.squeeze(camera_segmentations[segmentation_type])
        labels = [label for label in np.unique(seg) if label > 0]
        instance_id_to_name = (
            self.get_instance_id_to_name_map(include_background=False, include_asset_name=True)
            if segmentation_type == "instance"
            else {}
        )

        if len(labels) == 0:
            plt.figure(figsize=(4, 4))
            plt.imshow(rgb, origin="lower")
            plt.title(f"{cam_name} {segmentation_type}: no objects")
            plt.axis("off")
            plt.tight_layout()
            plt.show()
            return

        n_cols = min(4, len(labels))
        n_rows = int(np.ceil(len(labels) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
        axes = np.atleast_1d(axes).ravel()

        for ax, label in zip(axes, labels):
            mask = seg == label
            masked_rgb = rgb.copy()
            masked_rgb[~mask] = 0
            ax.imshow(masked_rgb, origin="lower")
            object_info = instance_id_to_name.get(int(label))
            object_name = None
            asset_name = None
            if isinstance(object_info, dict):
                object_name = object_info.get("name")
                asset_name = object_info.get("asset_name")
            else:
                object_name = object_info
            title = (
                f"{cam_name} {segmentation_type} id={int(label)}\n{object_name} ({asset_name})"
                if object_name is not None and asset_name is not None
                else f"{cam_name} {segmentation_type} id={int(label)}\n{object_name}"
                if object_name is not None
                else f"{cam_name} {segmentation_type} id={int(label)}"
            )
            ax.set_title(title)
            ax.axis("off")

        for ax in axes[len(labels):]:
            ax.axis("off")

        plt.tight_layout()
        plt.show()

    def close(self):
        if self.env is not None:
            self.env.close()
